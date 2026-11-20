#!/usr/bin/env python3
"""
涟漪认知场 · MCP Server
=========================================================
用途：在灵擎「MCP 工具管理」里注册这一个 Server，全部灵助助手即可调用涟漪判断内核。
传输：stdio（JSON-RPC 2.0，一行一条消息），零第三方依赖，便于私有化部署。
协议：MCP 2024-11-05（initialize / tools/list / tools/call）

注册示例（灵擎 MCP 工具管理）：
    名称：涟漪认知场
    传输：stdio
    命令：python3 mcp/ripple_mcp_server.py
    环境变量：RIPPLE_DATA_DIR=/opt/ripple/data

自检：python3 mcp/ripple_mcp_server.py --selftest
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mcp"))
sys.path.insert(0, str(ROOT / "adapters"))

from ripple_core import (  # noqa: E402
    CausalEngine,
    DIMS,
    HexagramIndex,
    RippleEngine,
    TRIT_CHAR,
)
from inspection_mapper import TritMapper, symptom_from_evidence  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "ripple-cognitive-field"
SERVER_VERSION = "1.0.0"

_engine: RippleEngine | None = None
_mapper: TritMapper | None = None


def engine() -> RippleEngine:
    global _engine
    if _engine is None:
        _engine = RippleEngine()
    return _engine


def mapper() -> TritMapper:
    global _mapper
    if _mapper is None:
        _mapper = TritMapper()
    return _mapper


# ============================================================
# 工具实现
# ============================================================

def tool_read_condition(readings: Dict[str, float], device_id: str = "") -> Dict[str, Any]:
    """测点 → 九维三态，并给出每维判定依据（不诊断，只看态势）"""
    mapped = mapper().map(readings)
    snap = engine().perceive(mapped["trits"], {"device_id": device_id})
    return {
        "device_id": device_id,
        "profile": mapped["profile"],
        "trits": mapped["trits"],
        "dim_scores": mapped["dim_scores"],
        "evidence": mapped["evidence"],
        "missing_signals": mapped["missing_signals"],
        "field": {
            "main_hexagram": snap["main"]["name"],
            "readable": snap["main"]["readable"],
            "entropy": round(snap["entropy"], 4),
            "stationary_ratio": round(snap["stationary_ratio"], 4),
        },
        "symptom": symptom_from_evidence(mapped["evidence"]),
    }


def tool_diagnose(
    readings: Dict[str, float],
    device_id: str = "",
    symptom: str = "",
    evolve_steps: int = 1,
) -> Dict[str, Any]:
    """异常诊断：映射 → 涟漪演化 → 因果三层查询 → 幻觉中和 → 工单建议"""
    mapped = mapper().map(readings)
    auto_symptom = symptom or symptom_from_evidence(mapped["evidence"])
    result = engine().diagnose(
        symptom=auto_symptom,
        trits=mapped["trits"],
        device={"device_id": device_id, "profile": mapped["profile"]},
        evolve_steps=evolve_steps,
    )
    result["evidence"] = mapped["evidence"]
    result["auto_symptom"] = auto_symptom
    result["work_order"] = _build_work_order(result, mapped)
    return result


def _build_work_order(result: Dict[str, Any], mapped: Dict[str, Any]) -> Dict[str, Any]:
    """把认知结论落成可派单的结构化工单（对应灵助"进入具体岗位流程"）"""
    status = result.get("status")
    method = result.get("method")
    root = result.get("root_cause")
    fix = result.get("fix_action")

    if status != "通过":
        level, action, owner = "待复核", "人工确认后再派单", "值班工程师"
    elif method == "immunity":
        level, action, owner = "一级(紧急)", fix or "按免疫处置卡执行", "设备工程师"
    elif method == "similarity":
        level, action, owner = "二级(重要)", fix or "参照历史工单处置", "设备工程师"
    else:
        level, action, owner = "三级(观察)", "补充测点后复判", "点检员"

    cands = (result.get("causal") or {}).get("candidates") or []
    if status == "待复核" and cands:
        action = "并联处置（按优先级）：" + " → ".join(c["fix_action"] for c in cands[:2])

    return {
        "title": f"设备异常处置 - {result.get('device', {}).get('device_id', '')}",
        "candidate_causes": [{"root_cause": c["root_cause"], "score": c["score"]} for c in cands],
        "level": level,
        "symptom": result.get("symptom"),
        "hexagram": result.get("hexagram", {}).get("name"),
        "readable": result.get("readable"),
        "root_cause": root,
        "fix_action": action,
        "assignee": owner,
        "evidence": [
            {"dim": e["dim_full"], "score": e["score"], "trit": e["trit_char"]}
            for e in mapped["evidence"]
            if e["trit"] != 0
        ],
        "trace": {
            "method": method,
            "confidence": result.get("confidence"),
            "risk": result.get("risk"),
            "status": status,
            "suggestions": result.get("suggestions", []),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def tool_immunity_query(symptom: str, trits: List[int] | None = None) -> Dict[str, Any]:
    return engine().causal.query(symptom, trits)


def tool_counterfactual(intervention: Dict[str, int], steps: int = 3) -> Dict[str, Any]:
    """intervention 允许用中文维名或序号，例 {"因": 1} 或 {"6": 1}"""
    key_map = {d: i for i, d in enumerate(DIMS)}
    parsed: Dict[int, int] = {}
    for k, v in (intervention or {}).items():
        ks = str(k)
        if ks in key_map:
            parsed[key_map[ks]] = int(v)
        elif ks.isdigit() and 0 <= int(ks) < 9:
            parsed[int(ks)] = int(v)
    return engine().counterfactual(parsed, steps)


def tool_field_status(tick: bool = False) -> Dict[str, Any]:
    return engine().life_status(tick=tick)


def tool_learn_case(
    symptom: str,
    root_cause: str,
    fix_action: str,
    trits: List[int],
    immediate_cause: str = "",
    verified: bool = True,
    device_type: str = "",
) -> Dict[str, Any]:
    return engine().learn(
        symptom=symptom,
        immediate_cause=immediate_cause,
        root_cause=root_cause,
        fix_action=fix_action,
        trits=trits,
        verified=verified,
        device_type=device_type,
    )


def tool_seed_knowledge(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = engine().causal.seed_from_knowledge(cases, verified=True)
    return {"seeded": n, "stats": engine().causal.stats()}


def tool_evolve(steps: int = 1) -> Dict[str, Any]:
    engine().field.evolve(steps)
    return engine().field.snapshot()


# ============================================================
# 工具注册表
# ============================================================

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "ripple_read_condition",
        "description": "把设备测点读数映射为九维三态认知构型，输出每维判定依据与场态摘要。用于点检/巡检的态势判读，不做根因结论。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "readings": {"type": "object", "description": "测点字典，如 {vibration_rms, bearing_temp, lube_pressure_dev, ...}"},
                "device_id": {"type": "string", "description": "设备标识"},
            },
            "required": ["readings"],
        },
    },
    {
        "name": "ripple_diagnose",
        "description": "设备异常诊断主入口：涟漪演化 + 因果三层查询（免疫/相似/待推理）+ 幻觉中和 + 结构化工单建议。生产常态应优先调用本工具。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "readings": {"type": "object", "description": "测点字典"},
                "device_id": {"type": "string"},
                "symptom": {"type": "string", "description": "可选，人工症状描述；留空则由测点自动生成"},
                "evolve_steps": {"type": "integer", "default": 1, "description": "演化步数，越大推演越深、延迟越高"},
            },
            "required": ["readings"],
        },
    },
    {
        "name": "ripple_immunity_query",
        "description": "只查因果免疫库：命中返回已验证根因与处置动作，未命中明确返回需推理，不编造结论。传入 trits 时按『文本+拓扑』双通道匹配，更准。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symptom": {"type": "string"},
                "trits": {"type": "array", "items": {"type": "integer"}, "description": "可选，9 个三态值"},
            },
            "required": ["symptom"],
        },
    },
    {
        "name": "ripple_counterfactual",
        "description": "反事实干预推演：锁定某几个维度后看全局演化，推完立即恢复原场（可逆）。用于'若改这一项会怎样'的预案演练。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "intervention": {"type": "object", "description": "维度→三态，如 {\"因\": 1}、{\"6\": 1}、{\"体\": -1}"},
                "steps": {"type": "integer", "default": 3},
            },
            "required": ["intervention"],
        },
    },
    {
        "name": "ripple_field_status",
        "description": "生命体征与认知场状态：主构型、场熵、驻波比、呼吸计数、记忆层级、因果库规模。用于看板与健康巡检。",
        "inputSchema": {
            "type": "object",
            "properties": {"tick": {"type": "boolean", "default": False, "description": "是否推进一次呼吸"}},
        },
    },
    {
        "name": "ripple_learn_case",
        "description": "经验回灌：把人工复核/工单闭环的结论写入因果库；verified=true 表示已核验，可抬升免疫置信度。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symptom": {"type": "string"},
                "root_cause": {"type": "string"},
                "fix_action": {"type": "string"},
                "trits": {"type": "array", "items": {"type": "integer"}, "description": "9 个三态值"},
                "immediate_cause": {"type": "string"},
                "verified": {"type": "boolean", "default": True},
                "device_type": {"type": "string"},
            },
            "required": ["symptom", "root_cause", "fix_action", "trits"],
        },
    },
    {
        "name": "ripple_seed_knowledge",
        "description": "把知识库/FMEA/历史工单批量导入为免疫种子，让助手第一天就有经验可用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cases": {"type": "array", "items": {"type": "object"}, "description": "[{symptom, immediate_cause, root_cause, fix_action, trits, device_type}]"}
            },
            "required": ["cases"],
        },
    },
    {
        "name": "ripple_evolve",
        "description": "推进认知场演化并返回快照，用于长时态势推演。",
        "inputSchema": {
            "type": "object",
            "properties": {"steps": {"type": "integer", "default": 1}},
        },
    },
]

DISPATCH = {
    "ripple_read_condition": tool_read_condition,
    "ripple_diagnose": tool_diagnose,
    "ripple_immunity_query": tool_immunity_query,
    "ripple_counterfactual": tool_counterfactual,
    "ripple_field_status": tool_field_status,
    "ripple_learn_case": tool_learn_case,
    "ripple_seed_knowledge": tool_seed_knowledge,
    "ripple_evolve": tool_evolve,
}


# ============================================================
# JSON-RPC / MCP
# ============================================================

def _ok(msg_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle(msg: Dict[str, Any]) -> Dict[str, Any] | None:
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method is None:
        return None

    if method == "initialize":
        return _ok(
            msg_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )

    if method in ("notifications/initialized", "initialized"):
        return None

    if method == "ping":
        return _ok(msg_id, {})

    if method == "tools/list":
        return _ok(msg_id, {"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        fn = DISPATCH.get(name)
        if fn is None:
            return _err(msg_id, -32601, f"未知工具: {name}")
        try:
            out = fn(**args) if isinstance(args, dict) else fn()
        except TypeError as e:
            return _err(msg_id, -32602, f"参数不匹配: {e}")
        except Exception as e:  # 生产环境要保证工具异常不打断会话
            return _err(msg_id, -32000, f"工具执行失败: {e}")
        return _ok(
            msg_id,
            {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, indent=2)}], "isError": False},
        )

    return _err(msg_id, -32601, f"未实现的方法: {method}")


def serve() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(_err(None, -32700, "JSON 解析失败"), ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def _call(msg_id: int, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    resp = handle(
        {"jsonrpc": "2.0", "id": msg_id, "method": "tools/call", "params": {"name": name, "arguments": args}}
    )
    if not resp or "result" not in resp:
        raise RuntimeError(resp.get("error", {}).get("message", "调用失败") if resp else "无响应")
    return json.loads(resp["result"]["content"][0]["text"])


def selftest() -> int:
    import os
    import tempfile

    # 自检必须用临时目录，不能污染生产数据目录
    os.environ["RIPPLE_DATA_DIR"] = tempfile.mkdtemp(prefix="ripple_selftest_")
    global _engine
    _engine = None

    fails: List[str] = []

    def expect(name: str, ok: bool, detail: str = "") -> None:
        print(f"[selftest] {name} -> {'ok' if ok else 'FAIL'}" + (f"  ({detail})" if detail else ""))
        if not ok:
            fails.append(name)

    print(f"[selftest] {SERVER_NAME} v{SERVER_VERSION} / protocol {PROTOCOL_VERSION} / data={os.environ['RIPPLE_DATA_DIR']}")
    init = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    expect("initialize", bool(init and init["result"]["protocolVersion"] == PROTOCOL_VERSION))

    listed = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = [t["name"] for t in listed["result"]["tools"]]
    expect("tools/list", len(names) == 8, f"{len(names)} tools")

    readings = {
        "vibration_rms": 6.4, "bearing_temp": 89, "flow_ratio": 0.71, "pressure_ratio": 0.69,
        "vibration_trend": 0.11, "abnormal_duration_min": 60, "alignment_dev": 0.04,
        "load_ratio": 0.90, "motor_current": 1.07, "lube_pressure_dev": 0.31,
        "process_param_dev": 0.03, "defect_count": 0, "first_pass_yield": 0.96,
    }
    healthy = {
        "vibration_rms": 1.9, "bearing_temp": 52, "flow_ratio": 0.92, "pressure_ratio": 0.90,
        "vibration_trend": 0.01, "abnormal_duration_min": 3, "alignment_dev": 0.03,
        "load_ratio": 0.78, "motor_current": 0.95, "lube_pressure_dev": 0.05,
        "process_param_dev": 0.02, "defect_count": 0, "first_pass_yield": 0.985,
        "ambient_temp_dev": 2, "upstream_stability": 0.93,
    }

    # 1) 空知识库：不得编造，必须中和
    empty = _call(3, "ripple_diagnose", {"readings": {**readings, "alignment_dev": 0.22}, "device_id": "PUMP-001"})
    expect("空知识库不编造（被中和转人工）", empty["status"] == "被中和", f"{empty['method']}/{empty['status']}")

    # 2) 健康设备：直接放行
    ok_health = _call(4, "ripple_diagnose", {"readings": healthy, "device_id": "PUMP-001"})
    expect("健康设备判正常不告警", ok_health["status"] == "通过" and ok_health["method"] == "normal")

    # 3) 灌入知识后再诊断：必须命中免疫并给出工单
    rc = _call(5, "ripple_read_condition", {"readings": readings, "device_id": "PUMP-001"})
    trits = rc["trits"]
    seed = _call(6, "ripple_seed_knowledge", {"cases": [{
        "symptom": rc["symptom"], "immediate_cause": "轴承磨损", "root_cause": "润滑失效",
        "fix_action": "更换润滑脂并复查油路", "trits": trits, "device_type": "旋转机械",
    }]})
    expect("知识导入", seed["seeded"] == 1, f"patterns={seed['stats']['immunity_patterns']}")

    dg = _call(7, "ripple_diagnose", {"readings": readings, "device_id": "PUMP-001"})
    expect("命中免疫层", dg["method"] == "immunity", f"{dg['method']} match={dg['causal'].get('match')}")
    expect("已核验经验可直接派单", dg["status"] == "通过", dg["judgement"])
    expect("工单等级与处置动作齐备", dg["work_order"]["level"] == "一级(紧急)" and bool(dg["work_order"]["fix_action"]),
           f"{dg['work_order']['level']}")

    # 4) 反事实可逆
    before = _call(8, "ripple_field_status", {"tick": False})["main"]["name"]
    cf = _call(9, "ripple_counterfactual", {"intervention": {"因": 1}, "steps": 10})
    after = _call(10, "ripple_field_status", {"tick": False})["main"]["name"]
    expect("反事实不污染主场", before == after and cf["reversible"], f"{cf['entropy_shift']}")

    # 5) 生命体征在稳态带
    st = _call(11, "ripple_field_status", {"tick": True})
    expect("认知场处于稳态带", 1.0 <= st["entropy"] <= 2.6 and st["stationary_ratio"] > 0.5,
           f"ent={st['entropy']} stat={st['stationary_ratio']}")

    print(f"[selftest] {'全部通过' if not fails else '失败项: ' + ', '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    serve()
