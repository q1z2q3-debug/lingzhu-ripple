#!/usr/bin/env python3
"""
涟漪认知场 · 灵助落地演示 + 自检
=========================================================
一条业务线走通：
  故障案例库种子（历史读数→同一映射器→免疫资产）
  → 常规点检（应判正常，不告警）
  → 轴承磨损征兆（应命中免疫）
  → 复合故障（对中偏差叠加，应命中相似）
  → 反事实预案（影子场，主场不受影响）
  → 工单闭环回灌（核验后免疫置信抬升）
  → 生命体征 + 性能压测

运行：python3 demo/demo_run.py
产出：demo/_out/demo_report.json（联调证据）
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mcp"))
sys.path.insert(0, str(ROOT / "adapters"))

from ripple_core import RippleEngine  # noqa: E402
from inspection_mapper import TritMapper, symptom_from_evidence  # noqa: E402

OUT = ROOT / "demo" / "_out"
DATA = OUT / "data"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main() -> int:
    if DATA.exists():
        shutil.rmtree(DATA)
    OUT.mkdir(parents=True, exist_ok=True)

    eng = RippleEngine(data_dir=str(DATA))
    mp = TritMapper()
    report: dict = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}

    # ------------------------------------------------------------
    hr("0  知识库种子：历史工单读数 → 同一映射器 → 免疫资产")
    # ------------------------------------------------------------
    # 三个历史案例的原始读数（真实上线时来自 EAM/QMS 工单附件）
    hist = [
        {
            "symptom_readings": {
                "vibration_rms": 6.4, "bearing_temp": 89, "vibration_trend": 0.11,
                "lube_pressure_dev": 0.31, "load_ratio": 0.90, "motor_current": 1.07,
                "flow_ratio": 0.71, "pressure_ratio": 0.69,
            },
            "immediate_cause": "轴承磨损",
            "root_cause": "润滑失效",
            "fix_action": "加注/更换润滑脂并按周期复查油路；磨损超标则更换轴承",
            "device_type": "旋转机械",
        },
        {
            "symptom_readings": {
                "vibration_rms": 5.8, "alignment_dev": 0.21, "bearing_temp": 72,
                "load_ratio": 0.86,
            },
            "immediate_cause": "联轴器对中超差",
            "root_cause": "安装对中偏差",
            "fix_action": "重新对中并复测联轴器径向/端面偏差",
            "device_type": "旋转机械",
        },
        {
            "symptom_readings": {
                "vibration_rms": 5.5, "flow_ratio": 0.45, "pressure_ratio": 0.48,
            },
            "immediate_cause": "叶轮结垢",
            "root_cause": "介质结垢导致叶轮不平衡",
            "fix_action": "停机清垢并做动平衡校验",
            "device_type": "旋转机械",
        },
    ]

    cases = []
    for h in hist:
        m = mp.map(h["symptom_readings"])
        cases.append(
            {
                "symptom": symptom_from_evidence(m["evidence"]),
                "immediate_cause": h["immediate_cause"],
                "root_cause": h["root_cause"],
                "fix_action": h["fix_action"],
                "trits": m["trits"],
                "device_type": h["device_type"],
            }
        )
        print(f"  {h['root_cause']}  → 三态 {m['trits']}")
    seeded = eng.causal.seed_from_knowledge(cases, verified=True)
    print(f"  已导入 {seeded} 条案例，免疫模式 {eng.causal.stats()['immunity_patterns']} 个")
    check("知识库种子可导入且生成免疫模式", eng.causal.stats()["immunity_patterns"] == 3)
    report["steps"]["seed"] = {"cases": cases, "stats": eng.causal.stats()}

    # ------------------------------------------------------------
    hr("1  常规点检：设备健康（不应告警）")
    # ------------------------------------------------------------
    normal = {
        "vibration_rms": 1.9, "bearing_temp": 52, "flow_ratio": 0.92, "pressure_ratio": 0.90,
        "vibration_trend": 0.01, "abnormal_duration_min": 3, "alignment_dev": 0.03,
        "load_ratio": 0.78, "motor_current": 0.95, "lube_pressure_dev": 0.05,
        "process_param_dev": 0.02, "defect_count": 0, "first_pass_yield": 0.985,
        "ambient_temp_dev": 2, "upstream_stability": 0.93,
    }
    m1 = mp.map(normal)
    r1 = eng.diagnose("设备运行正常", m1["trits"], device={"device_id": "PUMP-001"})
    print(f"  三态 {m1['trits']}  ({mp.describe(m1['trits'])})")
    print(f"  判定 {r1['status']} / {r1['judgement']}  主构型 {r1['hexagram']['name']}  熵 {r1['field']['entropy']}")
    check("健康设备判正常且不触发告警", r1["status"] == "通过" and r1["method"] == "normal")
    report["steps"]["normal"] = {"trits": m1["trits"], "result": r1}

    # ------------------------------------------------------------
    hr("2  异常巡检：轴承磨损征兆（应命中免疫）")
    # ------------------------------------------------------------
    worn = dict(normal)
    worn.update(
        {"vibration_rms": 6.4, "bearing_temp": 89, "vibration_trend": 0.11,
         "abnormal_duration_min": 60, "lube_pressure_dev": 0.31, "load_ratio": 0.90,
         "motor_current": 1.07, "flow_ratio": 0.71, "pressure_ratio": 0.69}
    )
    m2 = mp.map(worn)
    sym2 = symptom_from_evidence(m2["evidence"])
    r2 = eng.diagnose(sym2, m2["trits"], device={"device_id": "PUMP-001"})
    print(f"  三态 {m2['trits']}  ({mp.describe(m2['trits'])})")
    print(f"  方法 {r2['method']}  置信 {r2['confidence']}  状态 {r2['status']}  匹配度 {r2['causal'].get('match')}")
    print(f"  根因 {r2['root_cause']}")
    print(f"  处置 {r2['fix_action']}")
    print(f"  熵 {r2['field']['entropy']}  驻波比 {r2['field']['stationary_ratio']}  主构型概率 {r2['hexagram']['prob']}")
    check("轴承磨损命中免疫层", r2["method"] == "immunity")
    check("命中后给出可执行处置", bool(r2.get("fix_action")))
    check("判断通过闸门可派单", r2["status"] == "通过")
    report["steps"]["wear"] = {"trits": m2["trits"], "symptom": sym2, "result": r2}

    # ------------------------------------------------------------
    hr("3  复合故障：润滑失效 + 对中偏差叠加")
    # ------------------------------------------------------------
    compound = dict(worn)
    compound.update({"alignment_dev": 0.21, "vibration_rms": 7.0, "bearing_temp": 93})
    m3 = mp.map(compound)
    sym3 = symptom_from_evidence(m3["evidence"])
    r3 = eng.diagnose(sym3, m3["trits"], device={"device_id": "PUMP-001"})
    print(f"  三态 {m3['trits']}  ({mp.describe(m3['trits'])})")
    print(f"  方法 {r3['method']}  置信 {r3['confidence']}  状态 {r3['status']}  匹配度 {r3['causal'].get('match')}")
    print(f"  根因 {r3['root_cause']}")
    if r3["status"] != "通过":
        print(f"  中和原因 {r3['risk_factors']}")
        print(f"  建议 {r3['suggestions']}")
    check("复合故障给出结论或明确转人工（不允许硬编造）",
          r3["method"] in ("immunity", "similarity", "inference"))
    cands = r3["causal"].get("candidates") or []
    print(f"  候选根因 {[(c['root_cause'], c['score']) for c in cands]}")
    check("复合故障应给出多根因并置（不只报一个）", len(cands) >= 2, f"{len(cands)} 个候选")
    report["steps"]["compound"] = {"trits": m3["trits"], "symptom": sym3, "result": r3}

    # ------------------------------------------------------------
    hr("4  反事实预案：若先把「工艺成因」修回正常会怎样")
    # ------------------------------------------------------------
    before_main = eng.field.main()["name"]
    before_ent = round(eng.field.entropy(), 4)
    cf = eng.counterfactual({6: 1}, steps=10)  # 6 = 因
    print(f"  干预 {cf['intervention']}")
    print(f"  干预前 {cf['before']['hexagram']}  概率 {cf['before']['prob']}  熵 {cf['before']['entropy']}")
    print(f"  干预后 {cf['after']['hexagram']}  概率 {cf['after']['prob']}  熵 {cf['after']['entropy']}")
    print(f"  干预可行度 {cf['compat']}  抓手等级 {cf['lever']}  改变维度 {cf['changed_dims']}")
    print(f"  → {cf['verdict']}")
    check("反事实推演不污染主场（可逆）",
          eng.field.main()["name"] == before_main and round(eng.field.entropy(), 4) == before_ent)
    check("沙盘给出抓手等级（高/中/低）", cf["lever"] in ("高", "中", "低"), cf["lever"])
    report["steps"]["counterfactual"] = cf

    # ------------------------------------------------------------
    hr("5  工单闭环：人工核验结论回灌，抬升免疫置信")
    # ------------------------------------------------------------
    learned = eng.learn(
        symptom=cases[0]["symptom"],
        immediate_cause="润滑脂劣化（油品检测确认）",
        root_cause="润滑失效",
        fix_action="更换润滑脂型号并将换脂周期缩短至 2000h，同时复查油路过滤器",
        trits=cases[0]["trits"],
        verified=True,
        device_type="旋转机械",
    )
    print(f"  写入链条 {learned['chain_id']}  构型 {learned['hexagram']}")
    print(f"  免疫库 {learned['immunity']}")
    requery = eng.causal.query(cases[0]["symptom"], cases[0]["trits"])
    print(f"  立即复查 → {requery['method']}  置信 {requery['confidence']}  已核验 {requery.get('verified')}")
    check("核验回灌后可再次命中且置信提升",
          requery["method"] in ("immunity", "similarity") and requery["confidence"] >= 0.85)
    report["steps"]["learn"] = {"learned": learned, "requery": requery}

    # ------------------------------------------------------------
    hr("6  生命体征：认知场是否活着、稳不稳")
    # ------------------------------------------------------------
    st = eng.life_status(tick=True)
    print(f"  呼吸 {st['breath_count']} 次  运行 {st['uptime_sec']}s  观测源 {st['sources']}")
    print(f"  主构型 {st['main']['name']}  熵 {st['entropy']}  驻波比 {st['stationary_ratio']}  λ {st['lam']}")
    print(f"  记忆 {st['memory']}")
    print(f"  因果 {st['causal']}")
    check("场处于可解释稳态（熵在 1.2~2.6 且驻波比 > 0.5）",
          1.2 <= st["entropy"] <= 2.6 and st["stationary_ratio"] > 0.5,
          f"ent={st['entropy']} stat={st['stationary_ratio']}")
    report["steps"]["life"] = st

    # ------------------------------------------------------------
    hr("7  性能压测：连续演化 200 步")
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    eng.field.evolve(200)
    dt = time.perf_counter() - t0
    print(f"  200 步 {dt*1000:.1f} ms，单步 {dt*1000/200:.3f} ms")
    print(f"  演化后主构型 {eng.field.main()['name']}  熵 {eng.field.entropy():.4f}")
    check("单步演化 < 5ms（可进生产节拍）", dt * 1000 / 200 < 5.0, f"{dt*1000/200:.3f} ms")
    report["steps"]["perf"] = {"steps": 200, "ms_total": round(dt * 1000, 2), "ms_step": round(dt * 1000 / 200, 3)}

    # ------------------------------------------------------------
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    total = len(CHECKS)
    print("\n" + "=" * 70)
    print(f" 自检结果：{passed}/{total} 通过")
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"   FAIL → {name}  {detail}")
    print("=" * 70)

    report["checks"] = [{"name": n, "pass": ok, "detail": d} for n, ok, d in CHECKS]
    report["summary"] = {"passed": passed, "total": total}
    out_file = OUT / "demo_report.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"报告：{out_file}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
