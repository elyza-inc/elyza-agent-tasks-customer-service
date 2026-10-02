"""Direct, deterministic matrix checks for the v2 audio scorers."""

from __future__ import annotations


import pytest

from elyza_agent_tasks_customer_service.evaluation.audio.audio_value_metrics import (
    score_value_metrics,
)


SLOT = {"slot_id": "phone", "value": "09000000035", "value_type": "phone"}


def _message(seq: int, actor: str, content: str, *profiles: str) -> dict:
    """Build one v2 event-log message with optional ASR profile evidence."""

    event = {"seq": seq, "event_type": "message", "actor": actor, "content": content}
    if profiles:
        event["metadata"] = {
            "diagnostic_user_audio_asr": {
                "asr": {"profiles": [{"transcript": value} for value in profiles]}
            }
        }
    return event


def _tool(seq: int, value: str, source_kind: str = "customer_utterance") -> dict:
    """Build one v2 tool use for the declared important value."""

    return {
        "seq": seq,
        "event_id": f"call:{seq}",
        "event_type": "tool_call",
        "arguments": {"phone": value},
        "argument_provenance": {"phone": {"source_kind": source_kind}},
    }


def _score(events: list[dict], scenario: dict | None = None) -> dict:
    """Score one minimal v2 record and return its metric rows."""

    scenario = scenario or {"contracts": {"m20": {"important_values": [SLOT]}}}
    return score_value_metrics(record={"event_log": events}, scenario=scenario)["metrics"]


@pytest.mark.parametrize(
    ("ledger", "events", "metric_id", "status", "unit_id"),
    [
        (
            "B/M20 F1: same-shape ASR evidence uses a wrong value",
            [_message(1, "user", "09000000035"), _message(2, "operator", "確認: 09000000034", "09000000034", "09000000034")],
            "M20", "failed", "phone",
        ),
        (
            "B/M20 F2: spoken important value has no use evidence",
            [_message(1, "user", "09000000035")],
            "M20", "failed", "phone",
        ),
        (
            "B/M20 NA1: no customer-spoken important value",
            [_message(1, "user", "別件です")],
            "M20", "N/A", None,
        ),
        (
            "B/M25 F1: agreed repeat is the wrong value",
            [_message(1, "user", "09000000035"), _message(2, "operator", "09000000034", "09000000034", "09000000034")],
            "M25", "failed", "phone@1",
        ),
        (
            "B/M25 NA1: no customer-spoken important value",
            [_message(1, "user", "別件です")],
            "M25", "N/A", None,
        ),
        (
            "B/M25 NA2: no agreed repeat before next user turn",
            [_message(1, "user", "09000000035"), _message(2, "operator", "確認します")],
            "M25", "N/A", None,
        ),
        (
            "B/M26 F1: matching tool use is explicitly unknown",
            [_message(1, "user", "09000000035"), _tool(2, "09000000035", "unknown")],
            "M26", "failed", "call:2:phone",
        ),
        (
            "B/M26 NA1: no matching important-value tool use",
            [_message(1, "user", "09000000035"), _tool(2, "not-a-phone")],
            "M26", "N/A", None,
        ),
    ],
)
def test_audio_failure_and_na_matrix(
    ledger: str,
    events: list[dict],
    metric_id: str,
    status: str,
    unit_id: str | None,
) -> None:
    """台帳Bの指定音声失敗/N/A行を直接v2 scorerで固定する。"""

    del ledger
    row = _score(events)[metric_id]

    assert row["status"] == status
    if unit_id is None:
        assert row["units"] == []
    else:
        assert row["units"][0]["unit_id"] == unit_id


def test_m20_invalid_slot_is_not_measurable() -> None:
    """台帳B/M20 NM2: malformed important-value slot is N/M."""

    rows = _score([_message(1, "user", "09000000035")], {"contracts": {"m20": {"important_values": [{}]}}})

    assert rows["M20"]["status"] == "N/M"
    assert rows["M20"]["reason"] == "important value slot requires name and value"
    assert rows["M20"]["units"] == []


def test_m25_zero_repeat_is_na_with_zero_observation_rate() -> None:
    """台帳B/M25 NA2: no repeat remains the current N/A behavior."""

    row = _score([_message(1, "user", "09000000035"), _message(2, "operator", "承知しました")])["M25"]

    assert row["status"] == "N/A"
    assert row["observation"] == {"opportunity_count": 0, "target_value_count": 1, "rate": 0.0}


def test_m25_partial_observation_rate_is_fixed() -> None:
    """台帳B/M25 O1: partial repeat observation reports the exact rate."""

    scenario = {"contracts": {"m20": {"important_values": [SLOT, {"slot_id": "code", "value": "12", "value_type": "integer"}]}}}
    row = _score([_message(1, "user", "09000000035 と 12"), _message(2, "operator", "09000000035", "09000000035", "09000000035")], scenario)["M25"]

    assert row["status"] == "passed"
    assert row["observation"] == {"opportunity_count": 1, "target_value_count": 2, "rate": 0.5}
    assert row["units"][0]["unit_id"] == "phone@1"


@pytest.mark.parametrize(
    ("ledger", "event", "reason"),
    [
        ("B/M26 NM1: every call requires provenance", {"seq": 2, "event_type": "tool_call", "arguments": {}}, "argument_provenance_missing"),
        ("B/M26 NM2: matching argument requires a provenance entry", {"seq": 2, "event_type": "tool_call", "arguments": {"phone": "09000000035"}, "argument_provenance": {}}, "argument_provenance_missing"),
    ],
)
def test_m26_missing_provenance_is_not_measurable(ledger: str, event: dict, reason: str) -> None:
    """台帳B/M26のprovenance欠落・shape不正をN/Mとして固定する。"""

    del ledger
    row = _score([_message(1, "user", "09000000035"), event])["M26"]

    assert row["status"] == "N/M"
    assert row["reason"] == reason
    assert row["units"] == []


def test_m22_invalid_disclosure_contract_is_not_measurable() -> None:
    """台帳B/M22 NM3: invalid m05 disclosure contract is N/M."""

    row = _score([], {"contracts": {"m05": {"disclosure": None}}})["M22"]

    assert row["status"] == "N/M"
    assert row["reason"] == "contracts.m05.disclosure_invalid"
    assert row["units"] == []




def test_m20_ignores_corrected_names_other_numbers_and_ungoverned_slots() -> None:
    from elyza_agent_tasks_customer_service.evaluation.audio import audio_value_metrics as A

    kana = A._with_gold_argument_names([A._slot({"slot_id": "name_kana", "value": "ハラダユウ"})], {"gold_tool_calls": [{"arguments": {"name_kana": "ハラダユウ"}}]})[0]
    corrected = [{"event_type": "tool_call", "arguments": {"name_kana": "ササキレン"}, "argument_provenance": {"name_kana": {"source_kind": "customer_utterance"}}}]
    assert A._tool_evidence(kana, corrected, ["ササキレンデス"]) == []
    assert A._tool_evidence(kana, corrected, ["ハラダユウデス"]) == [{"kind": "tool_argument", "correct": False}]
    phone = {"slot_id": "phone", "normalized": "09000000045", "kind": "digits"}
    assert A._typed_profile_evidence(phone, ["58120739656です", "58120739656です"]) == []
    assert A._typed_profile_evidence(phone, ["09000000046です", "09000000046です"]) == [{"kind": "audio_asr", "correct": False}]
    country = A._with_gold_argument_names([A._slot({"slot_id": "country", "value": "セレノア"})], {"gold_tool_calls": []})[0]
    assert not A._argument_names_slot("contractor_name_kana", country)
    assert A._score_m20([country], [], [])["status"] == "N/A"


def test_missing_filter_argument_is_a_tool_error_not_a_stop() -> None:
    import json
    from pathlib import Path
    from elyza_agent_tasks_customer_service.evaluation.engine.package_runtime import PackageRuntime

    root = Path(__file__).resolve().parents[1] / "data"
    task = json.loads((root / "tasks/ec_flea/ecf-017.json").read_text(encoding="utf-8"))
    sol = json.loads((root / "solutions/ec_flea/ecf-017.json").read_text(encoding="utf-8"))
    pkg = {**task, **sol}
    pkg.pop("metadata", None)
    runtime = PackageRuntime(pkg, scenario_id=task["scenario_id"])
    tool_id = next(t for t, spec in runtime.tools.items() if any(isinstance(r, dict) and r.get("argument") == "name_kana" for r in spec["filters"]))
    for spec in runtime.tools[tool_id]["arguments"]:
        spec.pop("required", None)  # schema allows the omission; the filter still needs the value
    result = runtime.call(tool_id, {})
    assert result == {"ok": False, "error": "invalid_arguments", "message": "missing filter argument: name_kana"}


def test_unplanned_create_before_identity_is_a_tool_error_not_a_stop() -> None:
    import json
    from pathlib import Path
    from elyza_agent_tasks_customer_service.evaluation.engine.package_runtime import PackageRuntime

    root = Path(__file__).resolve().parents[1] / "data"
    task = json.loads((root / "tasks/parcel/par-001.json").read_text(encoding="utf-8"))
    sol = json.loads((root / "solutions/parcel/par-001.json").read_text(encoding="utf-8"))
    pkg = {**task, **sol}
    pkg.pop("metadata", None)
    runtime = PackageRuntime(pkg, scenario_id=task["scenario_id"])
    args = {a["name"]: "x" for a in runtime.tools["register_department_handoff"]["arguments"] if a.get("required")}
    result = runtime.call("register_department_handoff", args)
    assert result["error"] == "identity_not_verified"


def test_mutation_argument_problems_are_tool_errors() -> None:
    import json
    from pathlib import Path
    from elyza_agent_tasks_customer_service.evaluation.engine.package_runtime import PackageRuntime

    root = Path(__file__).resolve().parents[1] / "data"
    task = json.loads((root / "tasks/hotel/htl-001.json").read_text(encoding="utf-8"))
    sol = json.loads((root / "solutions/hotel/htl-001.json").read_text(encoding="utf-8"))
    pkg = {**task, **sol}
    pkg.pop("metadata", None)
    runtime = PackageRuntime(pkg, scenario_id=task["scenario_id"])
    tool_id, tool = next(
        (k, v) for k, v in runtime.tools.items()
        if v["operation"] == "update" and any(isinstance(s, dict) and "from_argument" in s for s in v["mutation"]["set"].values())
    )
    name = next(s["from_argument"] for s in tool["mutation"]["set"].values() if isinstance(s, dict) and "from_argument" in s)
    assert runtime._mutation_argument_problems(tool, {}) == [f"{name}: required for this operation"]


def test_asr_partial_name_and_shared_digit_mishearing_are_not_errors() -> None:
    from elyza_agent_tasks_customer_service.evaluation.audio import audio_value_metrics as A

    kana = {"slot_id": "name", "normalized": "モリカナエ", "kind": "kana"}
    assert A._typed_profile_evidence(kana, ["森カナエ様", "森 カナエ様"]) == []
    phone = {"slot_id": "phone", "normalized": "09000000048", "kind": "digits"}
    heard = ["090-0000-0018です", "090-0000-0018です"]
    assert A._typed_profile_evidence(phone, heard, "電話番号は090-0000-0048です") == []
    assert A._typed_profile_evidence(phone, heard, "電話番号は090-0000-0018です") == [{"kind": "audio_asr", "correct": False}]


def test_m22_is_not_applicable_before_the_deadline_tool() -> None:
    from elyza_agent_tasks_customer_service.evaluation.audio import audio_value_metrics as A

    scenario = {"contracts": {"m05": {"disclosure": "当日は100%の取消料", "deadline_tool_id": "cancel"}}}
    assert A._score_m22(scenario, [], [])["status"] == "N/A"
