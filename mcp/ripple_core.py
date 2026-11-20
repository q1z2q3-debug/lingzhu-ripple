"""
涟漪认知场 · 灵助落地内核
=========================================================
定位：作为灵助「运维域 / 质量域」智能助手共用的判断内核（不是又一个问答助手）
职责：可解释判断 + 因果根因免疫 + 不确定性量化 + 反事实推演 + 工单可追溯
接口：由 ripple_mcp_server.py 通过 MCP 暴露给灵擎「MCP 工具管理」

分层（与 L0/L1/L4/L6/L7 对齐）：
  L0  三元九爻 19683 卦空间（HexagramIndex）
  L1  涟漪概率动力学（RippleField：漂移-扩散平衡，存在非平凡稳态）
  L4  因果链与免疫（CausalEngine：文本相似 + 拓扑距离双通道匹配）
  L6  七层记忆（SevenLayerMemory，JSON 持久化）
  L7  幻觉中和（HallucinationGuard）

L1 的数学说明（很重要，不是修辞）：
  纯扩散（只做传播核卷积）在有限格点上的唯一稳态是均匀分布，也就是"热寂"，
  场面信息会随时间被抹平，这也是早期版本场熵逼近满值、驻波比≈0 的根因。
  本实现引入漂移项：维护认知先验 prior（历史观测的叠加势），每一步做
      ψ ← normalize( K_λ * ψ · exp(β·log prior + γ·coupling) )
  即离散格点上的漂移-扩散平衡（Gibbs 形式）。其不动点非均匀：
  吸引子处概率高、邻域按 exp(-λ·hamming) 衰减，宽度由 λ 与 β 共同决定。
  稳态存在 ⇒ 驻波比可测、场熵可解释、反事实可比较。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ============================================================
# L0  三元九爻 · 19683 卦空间
# ============================================================

DIMS: List[str] = ["体", "用", "变", "时", "空", "势", "因", "果", "缘"]
DIM_FULL: Dict[str, str] = {
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
TRIT_CHAR = {-1: "阴", 0: "和", 1: "阳"}

N_DIMS = 9
N_STATES = 3
N_NODES = N_STATES ** N_DIMS  # 19683


def _build_trit_matrix() -> np.ndarray:
    """19683 × 9 的构型矩阵，模块级缓存一次，避免每步重算"""
    idx = np.arange(N_NODES)[:, None]
    basis = (N_STATES ** np.arange(N_DIMS))[None, :]
    digits = (idx // basis) % N_STATES
    return (digits - 1).astype(np.float32)


_TRIT_MATRIX: Optional[np.ndarray] = None


def trit_matrix() -> np.ndarray:
    global _TRIT_MATRIX
    if _TRIT_MATRIX is None:
        _TRIT_MATRIX = _build_trit_matrix()
    return _TRIT_MATRIX


class HexagramIndex:
    DIMS = DIMS

    @staticmethod
    def encode(trits: Iterable[int]) -> int:
        values = list(trits)
        if len(values) != N_DIMS:
            raise ValueError(f"需要 {N_DIMS} 个维度，收到 {len(values)}")
        idx = 0
        for i, v in enumerate(values):
            if v not in (-1, 0, 1):
                raise ValueError(f"无效三态值: {v}")
            idx += (v + 1) * (3 ** i)
        return idx

    @staticmethod
    def decode(idx: int) -> List[int]:
        if not 0 <= idx < N_NODES:
            raise ValueError(f"索引超出范围: {idx}")
        return [int(x) for x in trit_matrix()[idx]]

    @staticmethod
    def to_name(trits: Iterable[int]) -> str:
        values = list(trits)
        return " ".join(f"{DIMS[i]}{TRIT_CHAR[values[i]]}" for i in range(N_DIMS))

    @staticmethod
    def to_readable(trits: Iterable[int]) -> str:
        values = list(trits)
        pos = [DIM_FULL[DIMS[i]] for i in range(N_DIMS) if values[i] == 1]
        neg = [DIM_FULL[DIMS[i]] for i in range(N_DIMS) if values[i] == -1]
        seg = []
        if pos:
            seg.append("偏强：" + "、".join(pos))
        if neg:
            seg.append("偏弱：" + "、".join(neg))
        return "；".join(seg) if seg else "各维度居中，无显著偏置"


def hamming(a: Iterable[int], b: Iterable[int]) -> int:
    return sum(1 for x, y in zip(list(a), list(b)) if x != y)


# ============================================================
# L1  涟漪概率动力学
# ============================================================

class RippleField:
    """
    漂移-扩散平衡的离散认知场。

    参数
      lam            传播核衰减（越大涟漪越窄、场越集中）
      beta           先验势阱强度（越大越"记住"已观测到的构型）
      prior_lam      先验核衰减（历史观测的锐度）
      coupling       维度耦合矩阵，刻画维度间相互强化
    """

    def __init__(
        self,
        lam: float = 1.4,
        beta: float = 1.0,
        prior_lam: float = 3.0,
        coupling_scale: float = 0.35,
        seed: int = 7,
    ):
        self.lam = float(lam)
        self.beta = float(beta)
        self.prior_lam = float(prior_lam)
        self.coupling_scale = float(coupling_scale)
        self.psi = np.full(N_NODES, 1.0 / N_NODES, dtype=np.float64)
        self.prior = np.zeros(N_NODES, dtype=np.float64)
        rng = np.random.default_rng(seed)
        self.coupling = rng.normal(0, 0.1, size=(N_DIMS, N_DIMS))
        np.fill_diagonal(self.coupling, 0.0)
        self.steps = 0
        self.source_count = 0
        self.adapt = True

    # ---------- 传播核 ----------
    @staticmethod
    def _blur(vec: np.ndarray, lam: float) -> np.ndarray:
        """9 次沿轴卷积 = 汉明距离核 exp(-λ·d) 的可分离实现"""
        w = float(np.exp(-lam))
        arr = vec.reshape((N_STATES,) * N_DIMS).copy()
        for axis in range(N_DIMS):
            arr = RippleField._blur_axis(arr, axis, w)
        return arr.reshape(-1)

    @staticmethod
    def _blur_axis(arr: np.ndarray, axis: int, w: float) -> np.ndarray:
        """
        沿单轴的一步三邻域扩散，边界反射（把缺失邻居的质量还给自身）。

        为什么必须反射：阴(-1)/和(0)/阳(+1) 只是一条 3 格点路径，端点只有 1 个邻居。
        普通卷积把越界当 0，端点行和 < 1，转移核就不是行随机的：
        每一步都会丢质量，全阴/全阳这类角落构型被系统性削弱，概率不守恒，
        表现为熵读数失真、条件化后场莫名接近均匀。反射后每轴行和恒为 1 且矩阵对称，
        9 轴可分离、严格守恒 —— 这是"场是可计算的"最底层前提。
        """
        a = np.moveaxis(arr, axis, -1)
        prev = np.zeros_like(a)
        nxt = np.zeros_like(a)
        prev[..., 1:] = a[..., :-1]
        nxt[..., :-1] = a[..., 1:]
        out = a + w * prev + w * nxt
        out[..., 0] += w * a[..., 0]    # 端点反射：无邻居可去，留在原地
        out[..., -1] += w * a[..., -1]
        out /= (1.0 + 2.0 * w)
        return np.moveaxis(out, -1, axis)

    def _kernel_for(self, trits: Iterable[int], lam: Optional[float] = None) -> np.ndarray:
        """
        先验核 = exp(-λ·汉明距离)，沿每轴用 3×3 矩阵 [[1,w,w²],[w,1,w],[w²,w,1]] 精确可分离实现。

        注意它与 _blur 是两种东西，不能混用：
          _blur 是"一步转移算子"（只能走一格，行随机、守恒），用来做动力学；
          _kernel_for 是"静态分布形状"（可达全部 9 跳），用来立先验。
        早期版本用 _blur 生成先验，导致先验支撑只有半径 1 的邻域，
        一旦要把某维从"阴"锁到"阳"（需跨两格），支撑里根本没有概率，
        条件化就退化成均匀分布（熵冲到 8.7）——这是必须分开的根本原因。
        """
        delta = np.zeros(N_NODES, dtype=np.float64)
        delta[HexagramIndex.encode(trits)] = 1.0
        w = float(np.exp(-(self.lam if lam is None else lam)))
        arr = delta.reshape((N_STATES,) * N_DIMS)
        mat = np.array([[1.0, w, w * w], [w, 1.0, w], [w * w, w, 1.0]])
        for axis in range(N_DIMS):
            a = np.moveaxis(arr, axis, -1)
            arr = np.moveaxis(a @ mat.T, -1, axis)
        k = arr.reshape(-1)
        return k / k.sum()

    # ---------- 势阱权重 ----------
    def _potential_weight(self) -> np.ndarray:
        """
        w = exp(β·log prior + γ·coupling)
        prior 用"概率混合"而不是"对数平均"：概率混合对矛盾证据得到双峰先验，
        场会分裂到两个吸引子上、中间是低谷 —— 这对应"两种解释并存"，
        是可解释的；对数平均会得到一条横跨两者的"山脊"，主构型会落到
        谁都不像的混合态，工程上无法解读，必须避免。
        """
        prior_norm = self.prior / max(self.prior.sum(), 1e-15)

        main_idx = int(np.argmax(self.psi))
        main_trits = trit_matrix()[main_idx].astype(np.float64)
        dim_score = self.coupling @ main_trits
        node_score = trit_matrix() @ dim_score
        node_score -= node_score.max()

        log_w = self.beta * np.log(prior_norm + 1e-15) + self.coupling_scale * node_score
        w = np.exp(log_w - log_w.max())
        return w / w.sum()

    def _posterior_step(self, mix: float = 0.12) -> None:
        """
        一步更新：ψ ← (1-m)·normalize(K_λψ · w) + m·prior

        为什么必须带这个混合项 m：
          纯乘性更新（ψ ← Kψ·w）是"保守"的——某构型当前概率为 0，它永远拿不到概率，
          新故障就永远进不了视野，这在数学上叫不可约性缺失（链不连通、不可达）。
          加入对先验的弱混合后，状态空间连通、存在唯一不变分布，
          新证据一定能被"走进来"，同时靠 m 小（0.12）保证旧认知不会被瞬间冲掉。
          这一步同时补上了此前悬着的"收敛性/可达性"缺口。
        """
        prop = self._blur(self.psi, self.lam)
        post = prop * self._potential_weight()
        total = post.sum()
        post = post / total if total > 1e-15 else np.full(N_NODES, 1.0 / N_NODES)
        prior = self.prior / max(self.prior.sum(), 1e-15) if self.prior.sum() > 0 else post
        self.psi = (1.0 - mix) * post + mix * prior
        self.psi /= self.psi.sum()

    # ---------- 扰动注入 ----------
    def inject_source(self, trits: Iterable[int], strength: float = 1.0) -> int:
        """
        单源扰动：
          1) 累积到认知先验（历史证据，形成势阱）
          2) 立即做一次后验更新（保证新证据马上影响场，而不是等几十步）
          3) 按场熵自适应调节 λ，使场维持在可解释的弥散带
        """
        trits = list(trits)
        idx = HexagramIndex.encode(trits)
        prior_kernel = self._kernel_for(trits, lam=self.prior_lam)

        # 先验走概率混合（EMA），不累加：
        # 累加会让"观测越多越尖锐"，把置信度变成次数的函数，是过度自信的来源；
        # 混合后锐度上限由单次观测核决定，置信度只来自"证据彼此一致"。
        alpha = 1.0 if self.source_count == 0 else 0.4 * max(0.0, min(1.0, float(strength)))
        self.prior = (1.0 - alpha) * self.prior + alpha * prior_kernel
        self.prior /= self.prior.sum()
        self.source_count += 1

        # 新证据要"走进视野"，只做 1~2 步会滞后于现实（旧构型仍占峰）；
        # 6 步足以让新构型夺峰，又不至于把历史认知一次性抹平。
        for _ in range(6):
            self._posterior_step()
        self._adapt_lam()
        return idx

    def inject_multi_sources(
        self, sources: List[List[int]], strengths: Optional[List[float]] = None
    ) -> None:
        if strengths is None:
            strengths = [1.0] * len(sources)
        for trits, s in zip(sources, strengths):
            self.inject_source(trits, s)

    def _adapt_lam(self) -> None:
        """
        熵控：熵高于目标（场太弥散）→ λ↑ 锐化；熵低于目标（场太集中）→ λ↓ 放开。
        注意方向不能反：λ 越大传播核越窄、场越集中、熵越低；
        若写成 lam += k*(target-ent)，就会形成"越集中越锐化"的正反馈坍塌。
        """
        if not self.adapt:
            return
        target = 2.0
        self.lam = float(np.clip(self.lam + 0.04 * (self.entropy() - target), 0.6, 2.4))

    # ---------- 演化 ----------
    def evolve_step(self) -> None:
        self._posterior_step()
        self._adapt_lam()
        self.steps += 1

    def evolve(self, steps: int = 1) -> None:
        for _ in range(max(0, int(steps))):
            self.evolve_step()

    def settle(self, steps: int = 12, tol: float = 1e-3, adapt: bool = True) -> int:
        """收敛到稳态（用于反事实、长时推演），返回实际迭代步数"""
        old = self.adapt
        self.adapt = adapt
        prev = None
        for i in range(steps):
            self.evolve_step()
            ent = self.entropy()
            if prev is not None and abs(ent - prev) < tol:
                self.adapt = old
                return i + 1
            prev = ent
        self.adapt = old
        return steps

    # ---------- 读数 ----------
    def entropy(self) -> float:
        p = self.psi
        return float(-np.sum(p * np.log(p + 1e-15)))

    def dispersion(self) -> float:
        return float(1.0 - self.psi.max())

    def main(self) -> Dict[str, Any]:
        idx = int(np.argmax(self.psi))
        trits = HexagramIndex.decode(idx)
        return {
            "index": idx,
            "trits": trits,
            "name": HexagramIndex.to_name(trits),
            "readable": HexagramIndex.to_readable(trits),
            "prob": round(float(self.psi[idx]), 4),
        }

    def top_k(self, k: int = 5) -> List[Dict[str, Any]]:
        k = max(1, min(int(k), 20))
        idxs = np.argsort(self.psi)[-k:][::-1]
        out = []
        for i in idxs.tolist():
            trits = HexagramIndex.decode(i)
            out.append(
                {
                    "index": i,
                    "trits": trits,
                    "name": HexagramIndex.to_name(trits),
                    "prob": round(float(self.psi[i]), 4),
                }
            )
        return out

    def stationary_ratio(self, k: int = 10) -> float:
        """驻波比：前 k 个构型占全场的质量，越高越接近稳态"""
        return float(np.sort(self.psi)[-k:].sum())

    def conditional(self, locks: Dict[int, int], steps: int = 8) -> "RippleField":
        """
        条件化场：把若干维"锁定"在指定三态，再在剩余维度上重算分布。
        数学上是条件概率 ψ(x | x_i = v_i)，不是"再喂一条证据"。
        区别很重要：再喂证据会让旧构型继续占峰（干预被稀释），
        条件化则直接问"若该维确定如此，最可能的整体构型是什么"，这才是预案要的答案。
        """
        f = RippleField(
            lam=self.lam, beta=self.beta, prior_lam=self.prior_lam,
            coupling_scale=self.coupling_scale, seed=17,
        )
        f.coupling = self.coupling.copy()
        trits = trit_matrix()
        mask = np.ones(N_NODES, dtype=bool)
        for i, v in (locks or {}).items():
            mask &= trits[:, int(i)] == int(v)

        prior = self.prior.copy()
        prior[~mask] = 0.0
        f.prior = prior / prior.sum() if prior.sum() > 1e-15 else mask.astype(np.float64) / mask.sum()

        psi = self.psi.copy()
        psi[~mask] = 0.0
        f.psi = psi / psi.sum() if psi.sum() > 1e-15 else f.prior.copy()
        f.source_count = self.source_count
        f.steps = self.steps
        f.adapt = False
        f.settle(max(4, int(steps)), adapt=False)
        return f

    def reset(self, keep_prior: bool = False) -> None:
        self.psi[:] = 1.0 / N_NODES
        if not keep_prior:
            self.prior[:] = 0.0
            self.source_count = 0
        self.lam = 1.4
        self.steps = 0

    def snapshot(self, top_k: int = 5) -> Dict[str, Any]:
        return {
            "main": self.main(),
            "top": self.top_k(top_k),
            "entropy": round(self.entropy(), 4),
            "dispersion": round(self.dispersion(), 4),
            "stationary_ratio": round(self.stationary_ratio(), 4),
            "lam": round(self.lam, 4),
            "steps": self.steps,
            "sources": self.source_count,
        }

    # ---------- 影子场：用于"若…会怎样" ----------
    def shadow(self, trits: Iterable[int]) -> "RippleField":
        """
        影子场 = 主场状态的完整副本，再叠加一次"按干预后构型观测"的证据。
        为什么必须复制主场的 prior：只有一个源的场天然比多源场集中，
        若用"新建空场"做反事实，熵的比较就失去意义（源数不同，不可比）。
        """
        f = RippleField(
            lam=self.lam, beta=self.beta, prior_lam=self.prior_lam,
            coupling_scale=self.coupling_scale, seed=13,
        )
        f.coupling = self.coupling.copy()
        f.psi = self.psi.copy()
        f.prior = self.prior.copy()
        f.source_count = self.source_count
        f.steps = self.steps
        f.adapt = False
        f.inject_source(trits, strength=1.0)
        f.settle(10, adapt=False)
        return f


# ============================================================
# L4  因果链与免疫
# ============================================================

@dataclass
class CausalChain:
    chain_id: str
    symptom: str
    immediate_cause: str
    root_cause: str
    fix_action: str
    method: str
    confidence: float
    hexagram_index: int
    hexagram_name: str
    trits: List[int] = field(default_factory=list)
    verified: bool = False
    device_type: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class ImmunityPattern:
    pattern_id: str
    root_cause: str
    fix_action: str
    symptom: str = ""
    trits: List[int] = field(default_factory=list)
    hit_count: int = 1
    confidence: float = 0.65
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    verified: bool = False


def _grams(text: str) -> set:
    """中文无空格：字二元组 + 空白切词，双通道保证召回"""
    text = (text or "").strip().lower()
    g = {text[i : i + 2] for i in range(max(0, len(text) - 1))}
    g.update(w for w in text.replace("，", " ").replace("、", " ").replace("；", " ").split() if w)
    return g


def text_similarity(a: str, b: str) -> float:
    sa, sb = _grams(a), _grams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def topo_similarity(a: Iterable[int], b: Iterable[int]) -> float:
    if not a or not b:
        return 0.0
    return 1.0 - hamming(a, b) / N_DIMS


def match_score(query_text: str, query_trits: Iterable[int], case_text: str, case_trits: Iterable[int]) -> float:
    """
    双通道匹配：文本语义 0.55 + 拓扑构型 0.45。
    工业场景里同一故障的表述千差万别，但测点构型高度一致，
    因此拓扑通道是主证据，文本只做消歧——这也是与纯 LLM 检索的关键差别。
    """
    ts = text_similarity(query_text, case_text)
    tp = topo_similarity(query_trits, case_trits)
    return round(0.55 * ts + 0.45 * tp, 4)


class CausalEngine:
    """三层查询：免疫记忆（ms 级）→ 相似匹配（含拓扑）→ 推理待定（交上层 Agent）"""

    def __init__(self, state_path: Optional[Path] = None):
        self.chains: List[CausalChain] = []
        self.immunity: Dict[str, ImmunityPattern] = {}
        self.state_path = Path(state_path) if state_path else None
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.chains = [CausalChain(**c) for c in data.get("chains", [])]
            self.immunity = {k: ImmunityPattern(**v) for k, v in data.get("immunity", {}).items()}
        except Exception:
            self.chains, self.immunity = [], {}

    def save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chains": [asdict(c) for c in self.chains],
            "immunity": {k: asdict(v) for k, v in self.immunity.items()},
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 写入 ----------
    def add_chain(
        self,
        symptom: str,
        immediate_cause: str,
        root_cause: str,
        fix_action: str,
        trits: Iterable[int],
        method: str = "inference",
        confidence: float = 0.55,
        verified: bool = False,
        device_type: str = "",
    ) -> CausalChain:
        trits = list(trits)
        idx = HexagramIndex.encode(trits)
        chain = CausalChain(
            chain_id=f"cc_{int(time.time()*1000)}_{len(self.chains)}",
            symptom=symptom,
            immediate_cause=immediate_cause,
            root_cause=root_cause,
            fix_action=fix_action,
            method=method,
            confidence=float(confidence),
            hexagram_index=idx,
            hexagram_name=HexagramIndex.to_name(trits),
            trits=trits,
            verified=verified,
            device_type=device_type,
        )
        self.chains.append(chain)
        self._try_immunize(chain)
        self.save()
        return chain

    def _try_immunize(self, chain: CausalChain) -> None:
        """
        免疫门槛：同根因链 >=3 条，或已被人工/工单核验。
        只有核验过的才能把置信度抬到 0.9 以上 — 防止把错误经验固化成"本能"。
        """
        hits = [c for c in self.chains if c.root_cause == chain.root_cause]
        if len(hits) < 3 and not chain.verified:
            return
        key = f"im_{abs(hash((chain.root_cause, chain.fix_action))) % 10**10}"
        if key in self.immunity:
            p = self.immunity[key]
            p.hit_count += 1
            p.last_seen = time.time()
            p.verified = p.verified or chain.verified
            p.confidence = min(0.97, p.confidence + (0.12 if chain.verified else 0.03))
        else:
            self.immunity[key] = ImmunityPattern(
                pattern_id=key,
                root_cause=chain.root_cause,
                fix_action=chain.fix_action,
                symptom=chain.symptom,
                trits=chain.trits,
                hit_count=1,
                confidence=0.92 if chain.verified else 0.65,
                verified=chain.verified,
            )
        self.save()

    def seed_from_knowledge(self, cases: List[Dict[str, Any]], verified: bool = True) -> int:
        """知识库 / FMEA / 历史工单 → 免疫初始资产（与灵擎知识库的唯一必要耦合点）"""
        n = 0
        for c in cases:
            self.add_chain(
                symptom=c.get("symptom", ""),
                immediate_cause=c.get("immediate_cause", ""),
                root_cause=c.get("root_cause", ""),
                fix_action=c.get("fix_action", ""),
                trits=c.get("trits", [0] * N_DIMS),
                method=c.get("method", "human"),
                confidence=float(c.get("confidence", 0.8)),
                verified=verified if c.get("verified") is None else bool(c.get("verified")),
                device_type=c.get("device_type", ""),
            )
            n += 1
        return n

    # ---------- 查询 ----------
    def query(
        self,
        symptom: str,
        trits: Optional[Iterable[int]] = None,
        immunity_th: float = 0.62,
        similar_th: float = 0.52,
    ) -> Dict[str, Any]:
        trits = list(trits) if trits is not None else []

        # 第一层：免疫记忆
        best_p, best_s = None, 0.0
        ranked_p = sorted(
            (
                (match_score(symptom, trits, p.symptom or p.root_cause, p.trits), p)
                for p in self.immunity.values()
            ),
            key=lambda x: x[0],
            reverse=True,
        )
        if ranked_p:
            best_s, best_p = ranked_p[0]
        if best_p is not None and best_s >= immunity_th:
            return {
                "method": "immunity",
                "confidence": round(best_p.confidence, 3),
                "root_cause": best_p.root_cause,
                "fix_action": best_p.fix_action,
                "immediate_cause": "命中已核验免疫模式",
                "pattern_id": best_p.pattern_id,
                "verified": best_p.verified,
                "hit_count": best_p.hit_count,
                "match": best_s,
                "candidates": [
                    {"root_cause": p.root_cause, "fix_action": p.fix_action, "score": round(s, 3)}
                    for s, p in ranked_p[:3]
                    if s >= 0.5
                ],
            }

        # 第二层：相似匹配（含拓扑距离）
        ranked_c = sorted(
            ((match_score(symptom, trits, c.symptom, c.trits), c) for c in self.chains),
            key=lambda x: x[0],
            reverse=True,
        )
        best_cs, best_c = ranked_c[0] if ranked_c else (0.0, None)
        if best_c is not None and best_cs >= similar_th:
            return {
                "method": "similarity",
                "confidence": round(min(0.86, 0.5 + best_cs * 0.4), 3),
                "root_cause": best_c.root_cause,
                "fix_action": best_c.fix_action,
                "immediate_cause": best_c.immediate_cause,
                "chain_id": best_c.chain_id,
                "verified": best_c.verified,
                "match": best_cs,
                "candidates": [
                    {"root_cause": c.root_cause, "fix_action": c.fix_action, "score": round(s, 3)}
                    for s, c in ranked_c[:3]
                    if s >= 0.5
                ],
            }

        # 第三层：明确交还给上层推理，绝不编造根因
        return {
            "method": "inference",
            "confidence": 0.0,
            "root_cause": None,
            "fix_action": None,
            "match": round(best_cs, 4),
            "candidates": [],
            "note": "未命中免疫与相似层，交多 Agent 推理；结论必须标注不确定性并人工复核",
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "chains": len(self.chains),
            "immunity_patterns": len(self.immunity),
            "verified_patterns": sum(1 for p in self.immunity.values() if p.verified),
        }


# ============================================================
# L6  七层记忆
# ============================================================

LEVELS = ["环境", "行为", "能力", "信念", "身份", "使命", "灵性"]


class SevenLayerMemory:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.layers: Dict[str, List[Dict[str, Any]]] = {k: [] for k in LEVELS}
        self.current = "环境"
        self.evolve_log: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for k in LEVELS:
                self.layers[k] = data.get("layers", {}).get(k, [])
            self.current = data.get("current", "环境")
            self.evolve_log = data.get("evolve_log", [])
        except Exception:
            pass

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"layers": self.layers, "current": self.current, "evolve_log": self.evolve_log}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write(self, level: str, payload: Dict[str, Any]) -> None:
        if level not in self.layers:
            return
        self.layers[level].append({"ts": time.time(), "data": payload})
        self._try_evolve()
        self.save()

    def _try_evolve(self) -> None:
        idx = LEVELS.index(self.current)
        if idx >= len(LEVELS) - 1:
            return
        cur = self.layers[self.current]
        if len(cur) >= 5 and cur[-1]["data"].get("confidence", 0) >= 0.8:
            nxt = LEVELS[idx + 1]
            self.evolve_log.append({"from": self.current, "to": nxt, "ts": time.time(), "trigger": "confidence>=0.8"})
            self.current = nxt

    def summary(self) -> Dict[str, Any]:
        return {
            "current_level": self.current,
            "counts": {k: len(v) for k, v in self.layers.items()},
            "evolutions": len(self.evolve_log),
        }


# ============================================================
# L7  幻觉中和
# ============================================================

class HallucinationGuard:
    """
    生产判断的三重闸门。关键设计：区分"证据不足"与"认知弥散"。
      - 命中免疫/相似层且已核验：信任证据（匹配度是硬的），不因场熵高而拦单；
        但若两个根因分数接近（复合故障），必须降级为待复核，不允许单点结论。
      - 推理型结论（无落地根因）：对未经核验的推断从严，场熵高即中和。
    这样既不会把已知故障拦成"无法判断"，也不会让推测借高置信蒙混派单。
    """

    def __init__(self, base_threshold: float = 0.6, entropy_limit: float = 2.6):
        self.base_threshold = base_threshold
        self.entropy_limit = entropy_limit

    def check(
        self,
        output: Dict[str, Any],
        memory_level: str = "环境",
        entropy: float = 0.0,
        dispersed: bool = False,
        conflict_gap: float = 0.25,
    ) -> Dict[str, Any]:
        risks: List[str] = []
        soft: List[str] = []
        advice: List[str] = []

        method = output.get("method")
        conf = float(output.get("confidence", 0.0) or 0.0)
        verified = bool(output.get("causal", {}).get("verified"))
        cands = output.get("causal", {}).get("candidates") or []
        high_levels = {"信念", "身份", "使命", "灵性"}
        threshold = self.base_threshold - 0.2 if memory_level in high_levels else self.base_threshold

        hard_gap = None
        if len(cands) >= 2:
            hard_gap = float(cands[0]["score"]) - float(cands[1]["score"])
            if hard_gap < conflict_gap:
                soft.append(f"composite_ambiguity(gap={hard_gap:.2f})")
                advice.append("疑似多根因并置，需并联处置并由工程师确认优先级")

        if method in (None, "inference") or not output.get("root_cause"):
            risks.append("no_grounded_cause")
            advice.append("无落地根因，转人工复核或补采测点/历史工单")
            if entropy > self.entropy_limit:
                risks.append(f"field_dispersion(entropy={entropy:.2f})")
                advice.append("认知场弥散，说明该构型无历史相似情形，结论不得直接派单")
            if dispersed:
                risks.append("no_stable_attractor")
        else:
            if conf < 0.55:
                risks.append("low_confidence")
                advice.append("补采数据或延长观测窗口")
            if not verified and int(output.get("causal", {}).get("hit_count", 0) or 0) < 3:
                soft.append("unverified_experience")
                advice.append("该经验未核验，处置动作建议随单附复核要求")

        risk_score = min(1.0, len(risks) * 0.28)
        out = dict(output)
        out["risk_factors"] = risks + soft
        out["risk"] = round(risk_score, 3)
        if risks:
            out["status"] = "被中和"
            out["judgement"] = "无法给出可靠判断，转人工"
        elif soft:
            out["status"] = "待复核"
            out["judgement"] = "已给出候选结论，需人工确认后执行"
        else:
            out["status"] = "通过"
            out["judgement"] = "判断可靠，可直接派单"
        out["suggestions"] = advice
        return out


# ============================================================
# 顶层引擎
# ============================================================

class RippleEngine:
    def __init__(self, data_dir: Optional[str] = None, lam: float = 1.4):
        base = Path(data_dir) if data_dir else Path(os.environ.get("RIPPLE_DATA_DIR", "./data"))
        base.mkdir(parents=True, exist_ok=True)
        self.field = RippleField(lam=lam)
        self.causal = CausalEngine(base / "causal_state.json")
        self.memory = SevenLayerMemory(base / "memory_state.json")
        self.guard = HallucinationGuard()
        self.breath_count = 0
        self.started_at = time.time()

    # ---------- 感知 ----------
    def perceive(self, trits: Iterable[int], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        trits = list(trits)
        self.field.inject_source(trits)
        snap = self.field.snapshot()
        self.memory.write(
            "环境",
            {
                "trits": trits,
                "hexagram": snap["main"]["name"],
                "entropy": snap["entropy"],
                "meta": meta or {},
                "confidence": 0.85,
            },
        )
        return snap

    # ---------- 诊断 ----------
    def diagnose(
        self,
        symptom: str,
        trits: Iterable[int],
        device: Optional[Dict[str, Any]] = None,
        evolve_steps: int = 1,
    ) -> Dict[str, Any]:
        trits = list(trits)
        self.field.inject_source(trits)
        self.field.evolve(evolve_steps)
        snap = self.field.snapshot()

        self.memory.write(
            "环境",
            {"trits": trits, "hexagram": snap["main"]["name"], "entropy": snap["entropy"], "confidence": 0.85},
        )

        # 三态在此编码"该维度的损益倾向"：阳=充足/健康，和=临界/居中，阴=不足或异常
        abnormal = any(t == -1 for t in trits)

        if not abnormal:
            # 正常态：不进因果层，避免"无异常也判高风险"的告警疲劳
            out = {
                "device": device or {},
                "symptom": symptom,
                "hexagram": snap["main"],
                "field": self._field_block(snap),
                "top_candidates": snap["top"][:3],
                "causal": {"method": "normal", "root_cause": None, "fix_action": None, "confidence": 0.0},
                "method": "normal",
                "confidence": 0.0,
                "root_cause": None,
                "fix_action": None,
                "readable": snap["main"]["readable"],
                "status": "通过",
                "judgement": "运行状态正常，无需处置",
                "risk": 0.0,
                "risk_factors": [],
                "suggestions": ["按周期点检即可"],
            }
            self.memory.write("行为", {"action": "diagnose_normal", "hexagram": snap["main"]["name"], "confidence": 0.9})
            return out

        causal = self.causal.query(symptom, trits)
        merged = {
            "device": device or {},
            "symptom": symptom,
            "hexagram": snap["main"],
            "field": self._field_block(snap),
            "top_candidates": snap["top"][:3],
            "causal": causal,
            "method": causal["method"],
            "confidence": causal["confidence"],
            "root_cause": causal.get("root_cause"),
            "fix_action": causal.get("fix_action"),
            "readable": snap["main"]["readable"],
        }
        checked = self.guard.check(
            merged,
            memory_level=self.memory.current,
            entropy=snap["entropy"],
            dispersed=snap["dispersion"] > 0.95,
        )
        self.memory.write(
            "行为",
            {
                "action": "diagnose",
                "symptom": symptom,
                "hexagram": snap["main"]["name"],
                "status": checked["status"],
                "confidence": 0.9 if checked["status"] == "通过" else 0.5,
            },
        )
        if checked["status"] == "通过" and causal["method"] in ("immunity", "similarity"):
            self.memory.write(
                "能力",
                {"skill": "ripple_diagnose", "method": causal["method"], "confidence": 0.85},
            )
        return checked

    @staticmethod
    def _field_block(snap: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entropy": snap["entropy"],
            "dispersion": snap["dispersion"],
            "stationary_ratio": snap["stationary_ratio"],
            "lam": snap["lam"],
        }

    # ---------- 反事实 ----------
    def counterfactual(self, intervention: Dict[int, int], steps: int = 8) -> Dict[str, Any]:
        """
        预案推演：把若干维锁定为"修复后的状态"，看整体最可能构型变成什么。
          - 主场完全不受影响（条件化场是独立对象，可逆由结构保证）
          - 比较口径用"主构型概率"而非熵：条件化必然压缩支撑集，熵的单向下降没有信息量；
            真正有意义的是"锁定后最可能的构型还站不站得住"（峰值升降）。
        """
        before = self.field.main()
        before_prob = float(before["prob"])
        before_ent = float(self.field.entropy())

        locks = {int(k): int(v) for k, v in (intervention or {}).items() if 0 <= int(k) < N_DIMS}
        if not locks:
            return {"error": "未给出有效干预维度"}

        # 干预可行度 = "修复后状态"在当前认知里本来占多少先验质量。
        # 这一项才是沙盘真正有用的读数：它回答"这一维现在偏得多不多"。
        # 若该维已明显偏离（质量很小），说明它是主要抓手、必须先修；
        # 若该维本来就接近目标状态（质量很大），动它没有收益。
        T = trit_matrix()
        mask = np.ones(N_NODES, dtype=bool)
        for i, v in locks.items():
            mask &= T[:, int(i)] == v
        compat = float(self.field.prior[mask].sum()) if self.field.source_count else 0.0

        shadow = self.field.conditional(locks)
        after = shadow.main()
        after_prob = float(after["prob"])
        after_ent = float(shadow.entropy())
        changed = sum(1 for a, b in zip(before["trits"], after["trits"]) if a != b)

        if compat < 0.05:
            verdict = "该维是当前主要偏离项，干预价值最高；修完必须复测确认（复查结果会决定是否再动别处）"
            lever = "高"
        elif compat < 0.30:
            verdict = "该维有偏离但不主导，可作为次要抓手"
            lever = "中"
        else:
            verdict = "该维当前已接近此状态，干预收益很小，优先级下调"
            lever = "低"

        return {
            "intervention": {DIMS[k]: TRIT_CHAR[v] for k, v in locks.items()},
            "before": {
                "hexagram": before["name"],
                "prob": round(before_prob, 4),
                "entropy": round(before_ent, 4),
            },
            "after": {
                "hexagram": after["name"],
                "prob": round(after_prob, 4),
                "entropy": round(after_ent, 4),
            },
            "changed_dims": changed,
            "compat": round(compat, 4),
            "lever": lever, 
            "entropy_shift": round(after_ent - before_ent, 4),
            "verdict": verdict,
            "reversible": True,
        }

    # ---------- 生命体征 ----------
    def life_status(self, tick: bool = False) -> Dict[str, Any]:
        if tick:
            self.field.evolve_step()
            self.breath_count += 1
        snap = self.field.snapshot()
        return {
            "alive": True,
            "breath_count": self.breath_count,
            "uptime_sec": round(time.time() - self.started_at, 1),
            "main": snap["main"],
            "entropy": snap["entropy"],
            "stationary_ratio": snap["stationary_ratio"],
            "dispersion": snap["dispersion"],
            "lam": snap["lam"],
            "sources": snap["sources"],
            "memory": self.memory.summary(),
            "causal": self.causal.stats(),
        }

    # ---------- 学习回灌 ----------
    def learn(
        self,
        symptom: str,
        immediate_cause: str,
        root_cause: str,
        fix_action: str,
        trits: Iterable[int],
        verified: bool = False,
        device_type: str = "",
    ) -> Dict[str, Any]:
        chain = self.causal.add_chain(
            symptom=symptom,
            immediate_cause=immediate_cause,
            root_cause=root_cause,
            fix_action=fix_action,
            trits=list(trits),
            method="human" if verified else "inference",
            confidence=0.85 if verified else 0.55,
            verified=verified,
            device_type=device_type,
        )
        if verified:
            self.memory.write(
                "信念",
                {"principle": f"{root_cause} → {fix_action}", "confidence": 0.85, "hexagram": chain.hexagram_name},
            )
        return {
            "chain_id": chain.chain_id,
            "hexagram": chain.hexagram_name,
            "immunity": self.causal.stats(),
            "next": "已入因果库；同类症状下次查询将命中免疫或相似层",
        }

    def reset(self) -> Dict[str, Any]:
        self.field.reset()
        return {"status": "场已清零", "breath_count": self.breath_count}
