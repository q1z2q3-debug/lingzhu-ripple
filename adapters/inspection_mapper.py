"""
测点 → 九维三态映射器（灵助侧唯一需要与内核约定的接口）
=========================================================
原则：
1. 每一维只吃少量核心测点，不做"什么都往里塞"的全量收敛；
2. 每个测点先归一化成 [-1, 1] 的"倾向分"（-1 最差，0 居中，1 最好/最强），
   再按权重聚合，最后过双阈值（滞回）判为 阴/和/阳；
3. 输出必须带"判定依据"列表，方便工程师复核，也方便幻觉中和时追溯。

维度语义（与 HexagramIndex.DIMS 严格一致）：
  体 本体结构（机械/结构健康）      用 功能性能（输出能力）
  变 变化趋势（劣化速度）            时 时序节奏（运行时长/延误）
  空 位置布局（装配位置/空间约束）   势 势能负载（负载率/电流）
  因 工艺成因（工艺参数偏离）        果 结果质量（缺陷/一次合格率）
  缘 外部关联（环境/上游供应/协同）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

DIMS = ["体", "用", "变", "时", "空", "势", "因", "果", "缘"]
DIM_FULL = {
    "体": "本体结构",
    "用": "功能性能",
    "变": "变化趋势",
    "时": "时序节奏",
    "空": "位置布局",
    "势": "势能负载",
    "因": "工艺成因",
    "果": "结果质量",
    "缘": "外部关联",
}

# 双阈值：进入阳/阴的分数门槛（留出滞回带，避免来回跳变）
T_HIGH = 0.42
T_LOW = -0.32


@dataclass
class Signal:
    """单测点规则"""

    key: str  # 读数里的字段名
    direction: str = "higher_worse"  # higher_worse / higher_better / lower_better
    warn: float = 0.0  # 开始偏弱的点
    crit: float = 0.0  # 已显著异常的点
    weight: float = 1.0
    unit: str = ""

    def score(self, value: float) -> float:
        """把原始读数折成 [-1, 1]：-1 越糟，+1 越健康"""
        if self.direction == "higher_worse":
            if value <= self.warn:
                return 0.0
            if value >= self.crit:
                return -1.0
            return -float((value - self.warn) / max(1e-9, self.crit - self.warn))
        if self.direction == "lower_better":
            # 越小越好（如附着力等级 1 级最佳）：crit 为最佳侧界限，warn 为开始变差的界限
            if value <= self.crit:
                return 1.0
            if value >= self.warn:
                return -1.0
            return float((self.warn - value) / max(1e-9, self.warn - self.crit))
        # higher_better
        if value >= self.crit:
            return 1.0
        if value <= self.warn:
            return -1.0
        return float((value - self.warn) / max(1e-9, self.crit - self.warn))


@dataclass
class DeviceProfile:
    """一类设备的映射规则；换设备只换 profile，不动内核"""

    name: str = "通用旋转机械"
    rules: Dict[str, List[Signal]] = field(default_factory=dict)

    @staticmethod
    def default_rotating() -> "DeviceProfile":
        """泵/风机/压缩机一类：最常用、也最容易验证的一组规则"""
        return DeviceProfile(
            name="旋转机械（泵/风机/压缩机）",
            rules={
                "体": [
                    Signal("vibration_rms", "higher_worse", 2.8, 7.1, 1.0, "mm/s"),
                    Signal("bearing_temp", "higher_worse", 65, 95, 0.6, "℃"),
                ],
                "用": [
                    Signal("flow_ratio", "higher_better", 0.6, 0.95, 0.8, "设计流量比"),
                    Signal("pressure_ratio", "higher_better", 0.6, 0.95, 0.6, "设计压比"),
                ],
                "变": [
                    Signal("vibration_trend", "higher_worse", 0.02, 0.15, 1.0, "相对劣化率"),
                ],
                "时": [
                    Signal("abnormal_duration_min", "higher_worse", 10, 240, 0.8, "min"),
                ],
                "空": [
                    Signal("alignment_dev", "higher_worse", 0.05, 0.25, 0.8, "mm"),
                ],
                "势": [
                    Signal("load_ratio", "higher_worse", 0.85, 1.05, 0.7, "额定负载比"),
                    Signal("motor_current", "higher_worse", 1.05, 1.3, 0.5, "额定电流比"),
                ],
                "因": [
                    Signal("lube_pressure_dev", "higher_worse", 0.1, 0.4, 1.0, "润滑压差"),
                    Signal("process_param_dev", "higher_worse", 0.05, 0.2, 0.6, "工艺参数偏离"),
                ],
                "果": [
                    Signal("defect_count", "higher_worse", 0, 3, 1.0, "件"),
                    Signal("first_pass_yield", "higher_better", 0.9, 0.99, 0.6, "一次合格率"),
                ],
                "缘": [
                    Signal("ambient_temp_dev", "higher_worse", 5, 20, 0.7, "℃"),
                    Signal("upstream_stability", "higher_better", 0.7, 0.95, 0.6, "上游稳定度"),
                ],
            },
        )

    @staticmethod
    def quality_line() -> "DeviceProfile":
        """
        质量域 profile（涂装/表面处理产线为样板，换线改阈值即可）
        注意：维度语义与运维域完全一致，只是测点不同——这是"一个内核多域复用"的关键。
        """
        return DeviceProfile(
            name="质量域·涂装/表面处理产线",
            rules={
                "体": [  # 产品本体/材料状态
                    Signal("coating_thickness", "higher_better", 18.0, 25.0, 0.9, "μm"),
                    Signal("adhesion_grade", "lower_better", 3.0, 1.0, 0.7, "附着力等级(1级最佳)"),
                ],
                "用": [
                    Signal("seal_test_ratio", "higher_better", 0.90, 0.99, 0.8, "密封合格率"),
                    Signal("conductivity_dev", "higher_worse", 0.05, 0.20, 0.6, "导电性偏离"),
                ],
                "变": [
                    Signal("param_drift_rate", "higher_worse", 0.03, 0.15, 1.0, "参数漂移率"),
                ],
                "时": [
                    Signal("process_interval_min", "higher_worse", 30, 180, 0.6, "工序间隔 min"),
                    Signal("storage_hours", "higher_worse", 24, 96, 0.5, "存放时长 h"),
                ],
                "空": [
                    Signal("coating_position_dev", "higher_worse", 0.02, 0.10, 0.7, "涂覆位置偏差"),
                ],
                "势": [
                    Signal("process_cpk", "higher_better", 1.0, 1.33, 0.8, "工序能力指数 Cpk"),
                ],
                "因": [
                    Signal("bath_concentration_dev", "higher_worse", 0.10, 0.30, 1.0, "槽液浓度偏离"),
                    Signal("bath_temp_dev", "higher_worse", 3, 8, 0.6, "槽液温度偏离 ℃"),
                ],
                "果": [
                    Signal("defect_rate", "higher_worse", 0.005, 0.02, 1.0, "缺陷率"),
                    Signal("first_pass_yield", "higher_better", 0.95, 0.995, 0.7, "一次合格率"),
                ],
                "缘": [
                    Signal("supplier_ppm", "higher_worse", 200, 600, 0.8, "供应商 ppm"),
                    Signal("humidity_rh", "higher_worse", 65, 80, 0.6, "%RH"),
                ],
            },
        )


class TritMapper:
    def __init__(self, profile: Optional[DeviceProfile] = None):
        self.profile = profile or DeviceProfile.default_rotating()

    def map(self, readings: Dict[str, float]) -> Dict[str, Any]:
        """returns {trits, dim_scores, evidence}"""
        trits: List[int] = []
        dim_scores: Dict[str, float] = {}
        evidence: List[Dict[str, Any]] = []

        for dim in DIMS:
            rules = self.profile.rules.get(dim, [])
            num = 0.0
            den = 0.0
            used: List[Dict[str, Any]] = []
            for sig in rules:
                if sig.key not in readings:
                    continue
                raw = readings[sig.key]
                try:
                    raw_f = float(raw)
                except (TypeError, ValueError):
                    continue
                s = sig.score(raw_f)
                num += s * sig.weight
                den += sig.weight
                used.append(
                    {
                        "signal": sig.key,
                        "value": raw_f,
                        "unit": sig.unit,
                        "score": round(s, 3),
                        "direction": sig.direction,
                        "warn": sig.warn,
                        "crit": sig.crit,
                    }
                )

            if den == 0:
                score = 0.0
            else:
                score = num / den
            dim_scores[dim] = round(score, 3)

            if score >= T_HIGH:
                t = 1
            elif score <= T_LOW:
                t = -1
            else:
                t = 0
            trits.append(t)

            evidence.append(
                {
                    "dim": dim,
                    "dim_full": DIM_FULL[dim],
                    "score": round(score, 3),
                    "trit": t,
                    "trit_char": {-1: "阴", 0: "和", 1: "阳"}[t],
                    "signals": used,
                }
            )

        return {
            "profile": self.profile.name,
            "trits": trits,
            "dim_scores": dim_scores,
            "evidence": evidence,
            "missing_signals": [k for k in self._all_keys() if k not in readings],
        }

    def _all_keys(self) -> List[str]:
        keys = []
        for rules in self.profile.rules.values():
            keys.extend(s.key for s in rules)
        return keys

    def describe(self, trits: List[int]) -> str:
        pos = [DIM_FULL[DIMS[i]] for i, v in enumerate(trits) if v == 1]
        neg = [DIM_FULL[DIMS[i]] for i, v in enumerate(trits) if v == -1]
        seg = []
        if pos:
            seg.append("偏强：" + "、".join(pos))
        if neg:
            seg.append("偏弱：" + "、".join(neg))
        return "；".join(seg) if seg else "各维度居中"


SIGNAL_LABEL = {
    "vibration_rms": ("振动烈度", "超标"),
    "bearing_temp": ("轴承温度", "超高"),
    "flow_ratio": ("输出流量", "不足"),
    "pressure_ratio": ("输出压比", "不足"),
    "vibration_trend": ("劣化速率", "过快"),
    "abnormal_duration_min": ("异常持续时长", "过长"),
    "alignment_dev": ("对中偏差", "超差"),
    "load_ratio": ("负载率", "过载"),
    "motor_current": ("电机电流", "偏高"),
    "lube_pressure_dev": ("润滑压差", "偏离"),
    "process_param_dev": ("工艺参数", "偏离"),
    "defect_count": ("缺陷数", "偏多"),
    "first_pass_yield": ("一次合格率", "偏低"),
    "ambient_temp_dev": ("环境温度", "偏离"),
    "upstream_stability": ("上游稳定度", "不足"),
    "coating_thickness": ("涂层厚度", "偏低"),
    "adhesion_grade": ("附着力等级", "下降"),
    "seal_test_ratio": ("密封合格率", "偏低"),
    "conductivity_dev": ("导电性", "偏离"),
    "param_drift_rate": ("参数漂移", "过快"),
    "process_interval_min": ("工序间隔", "过长"),
    "storage_hours": ("存放时长", "过长"),
    "coating_position_dev": ("涂覆位置", "偏移"),
    "process_cpk": ("工序能力", "不足"),
    "bath_concentration_dev": ("槽液浓度", "偏离"),
    "bath_temp_dev": ("槽液温度", "偏离"),
    "defect_rate": ("缺陷率", "偏高"),
    "supplier_ppm": ("供应商来料 ppm", "偏高"),
    "humidity_rh": ("环境湿度", "偏高"),
}


def _severity(score: float) -> str:
    if score <= -0.75:
        return "严重"
    if score <= -0.35:
        return "明显"
    return "轻度"


def symptom_from_evidence(evidence: List[Dict[str, Any]]) -> str:
    """
    自动生成"症状短语"，直接喂给因果层的相似度检索。
    只描述负向维度，且用语义化表述（不拼原始数值）——
    同一构型在不同工单里措辞不同、量程版本不同，只有语义短语才稳定可匹配。
    """
    parts = []
    for e in evidence:
        if e["trit"] != -1:
            continue
        hits = [s for s in e["signals"] if s["score"] <= T_LOW]
        if not hits:
            parts.append(f"{e['dim_full']}异常")
            continue
        worst = min(hits, key=lambda s: s["score"])
        label, verb = SIGNAL_LABEL.get(worst["signal"], (worst["signal"], "异常"))
        parts.append(f"{e['dim_full']}{_severity(worst['score'])}异常（{label}{verb}）")
    return "，".join(parts) if parts else "无显著异常"
