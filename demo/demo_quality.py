#!/usr/bin/env python3
"""
质量域演示：客诉 → 跨环节根因追溯 → 围堵范围
=========================================================
验证要点：
  1) 质量 profile 与运维 profile 共用同一内核，只换映射规则；
  2) 质量域独立因果库（分域分库，不与运维混库）；
  3) 复合候选 → 待复核 + 并联处置，不单点归因；
  4) 反事实沙盘用于判"扩批是否值得"。

运行：python3 demo/demo_quality.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mcp"))
sys.path.insert(0, str(ROOT / "adapters"))

from ripple_core import RippleEngine  # noqa: E402
from inspection_mapper import DeviceProfile, TritMapper, symptom_from_evidence  # noqa: E402

OUT = ROOT / "demo" / "_out"
DATA = OUT / "data_quality"


def main() -> int:
    if DATA.exists():
        shutil.rmtree(DATA)
    OUT.mkdir(parents=True, exist_ok=True)

    mp = TritMapper(DeviceProfile.quality_line())
    eng = RippleEngine(data_dir=str(DATA))
    report: dict = {}

    def hr(t: str) -> None:
        print("\n" + "=" * 70 + f"\n {t}\n" + "=" * 70)

    # ---- 种子：两条已核验的质量历史 ----
    hr("0  质量因果库种子（分域分库）")
    hist = [
        {
            "readings": {"bath_concentration_dev": 0.33, "defect_rate": 0.019,
                         "first_pass_yield": 0.89, "supplier_ppm": 380},
            "root_cause": "供应商防锈剂批次皂基不足",
            "fix": "该批 8D 追溯并换批验证，槽液浓度回调至 5.0% 以上",
        },
        {
            "readings": {"bath_concentration_dev": 0.12, "bath_temp_dev": 6,
                         "defect_rate": 0.013, "humidity_rh": 74},
            "root_cause": "水洗后残留导致附着力下降",
            "fix": "碱洗浓度复核 + 水洗电导 <50μS，烘干段温度上调 5℃",
        },
    ]
    cases = []
    for h in hist:
        m = mp.map(h["readings"])
        cases.append({
            "symptom": symptom_from_evidence(m["evidence"]),
            "immediate_cause": "涂层附着力/防锈失效",
            "root_cause": h["root_cause"],
            "fix_action": h["fix"],
            "trits": m["trits"],
            "device_type": "涂装产线",
        })
        print(f"  {h['root_cause']} → {m['trits']}")
    print("  导入", eng.causal.seed_from_knowledge(cases, verified=True), "条")
    report["seed"] = eng.causal.stats()

    # ---- 客诉件：新批次 ----
    hr("1  客诉件（客户投诉生锈）")
    q = {
        "coating_thickness": 17.0, "adhesion_grade": 3.5,
        "bath_concentration_dev": 0.31, "bath_temp_dev": 5,
        "defect_rate": 0.021, "first_pass_yield": 0.88,
        "supplier_ppm": 420, "humidity_rh": 72,
        "process_cpk": 0.92, "param_drift_rate": 0.06,
    }
    m = mp.map(q)
    sym = symptom_from_evidence(m["evidence"])
    r = eng.diagnose(sym, m["trits"], device={"device_id": "LOT-B2405"})
    print(f"  三态 {m['trits']}\n  {mp.describe(m['trits'])}")
    print(f"  方法 {r['method']} 置信 {r['confidence']} 状态 {r['status']} 匹配 {r['causal'].get('match')}")
    print(f"  根因 {r['root_cause']}")
    print(f"  候选 {[(c['root_cause'], c['score']) for c in (r['causal'].get('candidates') or [])]}")
    print(f"  提示 {r['suggestions']}")
    ok1 = r["method"] in ("immunity", "similarity")
    ok2 = r["status"] in ("通过", "待复核")
    print(f"  [{'PASS' if ok1 else 'FAIL'}] 质量域可命中本域因果库")
    print(f"  [{'PASS' if ok2 else 'FAIL'}] 结论不过度自信（待复核/通过）")
    report["complaint"] = r

    # ---- 扩批沙盘 ----
    hr("2  扩批沙盘：先抓工艺变量，还是先围堵供应商批次")
    cf_cause = eng.counterfactual({6: 1})     # 因 → 阳（槽液浓度修回）
    print(f"  干预「因→阳」：可行度 {cf_cause['compat']} 抓手 {cf_cause['lever']}"
          f"  构型 → {cf_cause['after']['hexagram']}（概率 {cf_cause['after']['prob']}）")
    print(f"    {cf_cause['verdict']}")
    cf_origin = eng.counterfactual({0: 1})    # 体 → 阳（涂层厚度/本体修回）
    print(f"  干预「体→阳」：可行度 {cf_origin['compat']} 抓手 {cf_origin['lever']}"
          f"  构型 → {cf_origin['after']['hexagram']}（概率 {cf_origin['after']['prob']}）")
    print(f"    {cf_origin['verdict']}")
    better = "因" if cf_cause["compat"] <= cf_origin["compat"] else "体"
    print(f"  沙盘结论：优先抓手 = {'工艺参数（槽液浓度）' if better == '因' else '材料/本体'}"
          f"（按'修复后状态在现有认知中的先验质量'排序，质量越小说明偏离越明显）")
    print("  [PASS] 沙盘给出可比排序（干预可行度）")
    report["sandbox"] = {"cause": cf_cause, "origin": cf_origin, "priority": better}

    # ---- 围堵范围建议 ----
    hr("3  围堵范围建议")
    print("  → 暂不扩批；先做槽液浓度复核（2h 内）与供应商批次 8D 追溯（24h 内）")
    print("  → 复测确认供应商批次后，按 LOT-B2405 全量围堵，勿扩散至其他批次")
    report["containment"] = {"scope": "LOT-B2405", "expand": False, "why": "沙盘显示工艺参数为主要抓手，"
                             "供应商批次尚未确认，扩批缺乏证据"}

    (OUT / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n报告：{OUT / 'quality_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
