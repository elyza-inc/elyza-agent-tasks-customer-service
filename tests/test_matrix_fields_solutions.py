"""台帳 A の data/solutions 未カバー欄を、現行 scorer の境界で固定する。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from elyza_agent_tasks_customer_service.evaluation.engine.package_adapter import (
    PackageAdapterError,
    convert_package,
)
from elyza_agent_tasks_customer_service.evaluation.engine.package_runtime import (
    _run_outcome_from_completion,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.package_scoring import (
    score_package_record,
)
from tests.helpers.gold_record_builder import (
    JUDGE_CONFIG,
    load_source_package,
    perfect_judge_transport,
    perfect_record,
)
from tests.test_single_defect_sweep import inject_defect


SCENARIO = ("hotel", "htl-001")
REASON_REQUIRED_MISSING = "required_mutation_missing"
REASON_FORBIDDEN_OBSERVED = "forbidden_mutation_observed"
UNREAD_FIELDS = (
    ("S02", ("contracts", "completion", "handling_type")),
    *(
        ("S06", ("contracts", "completion", name))
        for name in (
            "required_callback", "required_handoff", "required_next_step",
            "required_outcomes", "required_record", "required_refusal_reason",
            "required_route", "required_state",
        )
    ),
    ("S11", ("execution_trace", 0, "returned_rows", 0, "row_identity", "customer")),
    ("S11", ("execution_trace", 0, "returned_rows", 0, "row_identity", "role")),
    ("S11", ("execution_trace", 0, "returned_rows", 0, "row_identity", "role_name")),
    ("S11", ("execution_trace", 0, "returned_rows", 0, "row_identity", "sequence")),
    ("S11", ("execution_trace", 0, "returned_rows", 0, "values", "guest_id")),
    ("S12", ("execution_trace", 4, "world_changes", 0, "operation")),
    ("S12", ("execution_trace", 4, "world_changes", 0, "entity")),
    ("S12", ("execution_trace", 4, "world_changes", 0, "row_identity", "customer")),
    ("S12", ("execution_trace", 4, "world_changes", 0, "row_identity", "role")),
    ("S12", ("execution_trace", 4, "world_changes", 0, "row_identity", "role_name")),
    ("S12", ("execution_trace", 4, "world_changes", 0, "row_identity", "sequence")),
    ("S12", ("execution_trace", 4, "world_changes", 0, "before", "status")),
    ("S12", ("execution_trace", 4, "world_changes", 0, "after", "status")),
    ("S14", ("gold_dialogue", "scenario_id")),
    ("S14", ("gold_dialogue", "final_world_hash")),
    ("S15", ("gold_dialogue", "turns", 0, "turn_id")),
    ("S15", ("gold_dialogue", "turns", 0, "organization")),
    ("S16", ("gold_dialogue", "turns", 0, "source")),
    ("S18", ("gold_dialogue", "turns", 7, "returned_rows", 0, "row_identity", "customer")),
    ("S18", ("gold_dialogue", "turns", 7, "returned_rows", 0, "row_identity", "role")),
    ("S18", ("gold_dialogue", "turns", 7, "returned_rows", 0, "row_identity", "role_name")),
    ("S18", ("gold_dialogue", "turns", 7, "returned_rows", 0, "row_identity", "sequence")),
    ("S18", ("gold_dialogue", "turns", 7, "returned_rows", 0, "values", "guest_id")),
    ("S18", ("gold_dialogue", "turns", 19, "world_changes", 0, "operation")),
    ("S18", ("gold_dialogue", "turns", 19, "world_changes", 0, "entity")),
    ("S18", ("gold_dialogue", "turns", 19, "world_changes", 0, "row_identity", "customer")),
    ("S18", ("gold_dialogue", "turns", 19, "world_changes", 0, "row_identity", "role")),
    ("S18", ("gold_dialogue", "turns", 19, "world_changes", 0, "row_identity", "role_name")),
    ("S18", ("gold_dialogue", "turns", 19, "world_changes", 0, "row_identity", "sequence")),
    ("S18", ("gold_dialogue", "turns", 19, "world_changes", 0, "before", "status")),
    ("S18", ("gold_dialogue", "turns", 19, "world_changes", 0, "after", "status")),
    ("S18", ("gold_dialogue", "turns", 14, "error")),
    ("S19", ("gold_dialogue", "tool_trace", 0, "tool_id")),
    ("S19", ("gold_dialogue", "tool_trace", 0, "arguments", "guest_name_kana")),
    ("S19", ("gold_dialogue", "tool_trace", 0, "returned_rows", 0, "row_identity", "customer")),
    ("S19", ("gold_dialogue", "tool_trace", 0, "returned_rows", 0, "row_identity", "role")),
    ("S19", ("gold_dialogue", "tool_trace", 0, "returned_rows", 0, "row_identity", "role_name")),
    ("S19", ("gold_dialogue", "tool_trace", 0, "returned_rows", 0, "row_identity", "sequence")),
    ("S19", ("gold_dialogue", "tool_trace", 0, "returned_rows", 0, "values", "guest_id")),
    ("S19", ("gold_dialogue", "tool_trace", 4, "world_changes", 0, "operation")),
    ("S19", ("gold_dialogue", "tool_trace", 4, "world_changes", 0, "entity")),
    ("S19", ("gold_dialogue", "tool_trace", 4, "world_changes", 0, "row_identity", "customer")),
    ("S19", ("gold_dialogue", "tool_trace", 4, "world_changes", 0, "row_identity", "role")),
    ("S19", ("gold_dialogue", "tool_trace", 4, "world_changes", 0, "row_identity", "role_name")),
    ("S19", ("gold_dialogue", "tool_trace", 4, "world_changes", 0, "row_identity", "sequence")),
    ("S19", ("gold_dialogue", "tool_trace", 4, "world_changes", 0, "before", "status")),
    ("S19", ("gold_dialogue", "tool_trace", 4, "world_changes", 0, "after", "status")),
    ("S19", ("gold_dialogue", "tool_trace", 2, "error")),
)


def _package() -> dict[str, Any]:
    """Return the representative object-root source package for this ledger."""

    return load_source_package(*SCENARIO)


def _delete(value: Any, path: tuple[str | int, ...]) -> None:
    """Remove an existing leaf, or add an absent schema leaf as a canary."""

    current = value
    for part in path[:-1]:
        current = current[part]
    if path[-1] in current:
        del current[path[-1]]
    else:
        current[path[-1]] = {"unread_canary": True}


def _metric_rows(package: dict[str, Any], record: dict[str, Any], cache_dir: Path) -> list[dict[str, Any]]:
    """Return stable text metric rows for one decoded package and fixed record."""

    return score_package_record(
        package=package,
        record=record,
        judge_config=JUDGE_CONFIG,
        cache_dir=cache_dir,
        transport=perfect_judge_transport,
    )["metric_results"]


@pytest.mark.parametrize(
    "ledger,path",
    UNREAD_FIELDS,
    ids=[f"{ledger}-{'-'.join(map(str, path))}" for ledger, path in UNREAD_FIELDS],
)
def test_unread_solution_fields_do_not_change_metric_rows(
    ledger: str, path: tuple[str | int, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S02/S06/S11/S12/S14/S15/S16/S18/S19 の台帳行の未読 leaf を固定する。"""

    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")
    baseline = _package()
    record = perfect_record(baseline)
    candidate = deepcopy(baseline)
    _delete(candidate, path)

    assert _metric_rows(candidate, record, tmp_path / "candidate") == _metric_rows(
        baseline, record, tmp_path / "baseline"
    ), f"{ledger}:{'.'.join(map(str, path))}: unread_projection_changed"


@pytest.mark.parametrize("field", ("table", "customer", "role", "role_name", "sequence", "set", "kind"))
def test_s03_required_mutation_leaf_has_fixed_missing_reason(field: str) -> None:
    """| S03 | `contracts.completion.required_mutations[]` のS01と同じ7 leaf | 7 | R1/R5 (`package_runtime.py:1353-1424,1557-1568`) | run outcome、M01/M14 |  |"""

    package = _package()
    record = perfect_record(package)
    requirement = package["contracts"]["completion"]["required_mutations"][1]
    if field == "table":
        requirement[field] = "missing_table"
    elif field == "set":
        requirement[field] = {"outcome": "missing_outcome"}
    elif field == "kind":
        requirement[field] = "update"
    else:
        requirement["record"][field] = "missing" if field != "sequence" else 999

    outcome = _run_outcome_from_completion(package["contracts"]["completion"], record["event_log"])
    assert outcome["reason"] == REASON_REQUIRED_MISSING, f"S03:{field}:case_records"


def test_s03_existing_defect_injection_reaches_completion_reason() -> None:
    """| S03 | `contracts.completion.required_mutations[]` のS01と同じ7 leaf | 7 | R1/R5 (`package_runtime.py:1353-1424,1557-1568`) | run outcome、M01/M14 |  |"""

    package = _package()
    record = perfect_record(package)
    inject_defect(record, "m11_required_tool_missing")

    outcome = _run_outcome_from_completion(package["contracts"]["completion"], record["event_log"])
    assert outcome["reason"] == REASON_REQUIRED_MISSING, "S03:m11_required_tool_missing:case_records"


@pytest.mark.parametrize("field", ("table", "customer", "role", "role_name", "sequence", "set", "kind"))
def test_s04_forbidden_mutation_leaf_has_fixed_observed_reason(field: str) -> None:
    """| S04 | `contracts.completion.forbidden_mutations[]` のS01と同じ7 leaf | 7 | R5 (`:1557-1574`) | run outcome、M01/M14 |  |"""

    package = _package()
    record = perfect_record(package)
    forbidden = package["contracts"]["completion"]["forbidden_mutations"][0]
    observed = package["contracts"]["completion"]["required_mutations"][0]
    forbidden.update(deepcopy(observed))
    forbidden["record"]["role_name"] = "answer"
    if field == "kind":
        forbidden[field] = "update"

    outcome = _run_outcome_from_completion(package["contracts"]["completion"], record["event_log"])
    assert outcome["reason"] == REASON_FORBIDDEN_OBSERVED, f"S04:{field}:reservations"


@pytest.mark.parametrize(
    "field",
    ("slot_id", "value", "value_type", "question_step_id", "readback_role"),
)
def test_s09_important_value_leaf_changes_audio_contract(field: str) -> None:
    """| S09 | `contracts.m20.important_values[].{slot_id,value,value_type,question_step_id,readback_role}` | 5 | R3 (`:201-227`), R9 (`audio_value_metrics.py:70-104`) | M20/M25/M26、M16-M4/M16-M5 |  |"""

    package = _package()
    candidate = deepcopy(package)
    candidate["contracts"]["m20"]["important_values"][0][field] = "changed"
    baseline = convert_package(package, mode="audio-text")
    actual = convert_package(candidate, mode="audio-text")

    assert actual["contracts"]["m20"]["important_values"][0][field] == "changed", f"S09:{field}:phone_number"
    assert actual["contracts"]["m20"] != baseline["contracts"]["m20"], f"S09:{field}:phone_number:projection"


@pytest.mark.parametrize("field", ("kind", "speaker", "text", "slot_id", "tool_id", "arguments"))
def test_s15_s16_s17_read_dialogue_leaf_changes_or_rejects_projection(field: str) -> None:
    """| S15 | `gold_dialogue.turns[].{kind,turn_id,speaker,organization}` | 4 | `kind,speaker`: R3 (`:299-327`); `turn_id,organization`: **未読** | ideal conversation、M05/M11 |  |\n
    | S16 | utterance turnの `text,source,slot_id` | 3 | `text,slot_id`: R3 (`:311-316,473-490`); `source`: **未読** | M05 disclosure/consent、judge gold | `text,slot_id` ✓E |\n
    | S17 | tool_call turnの `tool_id,arguments.*` | 2 | R3 (`:317-324`) | M04/M05/M11 |  |"""

    package = _package()
    candidate = deepcopy(package)
    turns = candidate["gold_dialogue"]["turns"]
    first_tool_call = next(i for i, turn in enumerate(turns) if turn["kind"] == "tool_call")
    first_explain = next(i for i, turn in enumerate(turns) if "explain" in str(turn.get("slot_id") or ""))
    index = 1 if field in {"kind", "speaker", "text", "slot_id"} else first_tool_call
    if field == "kind":
        turns[index][field] = "tool_result"
    elif field == "speaker":
        turns[index][field] = "tool"
    elif field == "text":
        turns[index][field] = "changed utterance"
    elif field == "slot_id":
        turns[first_explain][field] = "changed_slot"
    elif field == "tool_id":
        turns[index][field] = "changed_tool"
    else:
        turns[index][field] = {"changed": True}

    if field in {"speaker", "slot_id", "tool_id"}:
        with pytest.raises(PackageAdapterError, match="(invalid utterance|must bind one explain turn|Tool is absent)"):
            convert_package(candidate, mode="text")
    else:
        assert convert_package(candidate, mode="text")["_ideal_conversation"] != convert_package(
            package, mode="text"
        )["_ideal_conversation"], f"S15/S16/S17:{field}:htl-001"


@pytest.mark.parametrize("field", ("tool_id", "arguments"))
def test_s20_gold_tool_call_leaf_changes_expected_trace(field: str) -> None:
    """| S20 | `gold_tool_calls[].{tool_id,arguments.*}` | 2 | R3 (`:150-155,620-635`) | M11 |  |"""

    package = _package()
    candidate = deepcopy(package)
    candidate["gold_tool_calls"][0][field] = "changed_tool" if field == "tool_id" else {"changed": True}

    assert convert_package(candidate, mode="text")["expected_procedure"] != convert_package(
        package, mode="text"
    )["expected_procedure"], f"S20:{field}:verify_identity"


@pytest.mark.parametrize("field", ("action_codes", "evidence_kinds", "handling_types", "id_types"))
def test_s22_ticket_enum_leaf_is_contract_invalid(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """| S22 | `post_call_ticket_contract.enum_catalog.{action_codes,evidence_kinds,handling_types,id_types}` | 4 | R1/R7（同上） | M14/M15 ticket validation | `handling_types` ✓E / ✓B |"""

    package = _package()
    record = perfect_record(package)
    del package["post_call_ticket_contract"]["enum_catalog"][field]
    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")
    result = score_package_record(
        package=package,
        record=record,
        judge_config=JUDGE_CONFIG,
        cache_dir=tmp_path / field,
        transport=perfect_judge_transport,
    )
    row = next(item for item in result["metric_results"] if item["metric_id"] == "M14")

    assert row["status"] == "N/M", f"S22:{field}:M14"
    assert row["reason"] == "deterministic ticket enum_catalog fields are invalid", f"S22:{field}:M14"


def test_s25_metadata_canary_is_not_projected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """| S25 | `metadata.canary` | 1 | **未読** | なし |  |"""

    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")
    package = _package()
    record = perfect_record(package)
    candidate = deepcopy(package)
    candidate["metadata"] = {"canary": "changed"}

    assert _metric_rows(candidate, record, tmp_path / "candidate") == _metric_rows(
        package, record, tmp_path / "baseline"
    ), "S25:metadata.canary:unread_projection_changed"
