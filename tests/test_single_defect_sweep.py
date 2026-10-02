"""Single-defect checks for separation between the integrated text metrics."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from elyza_agent_tasks_customer_service.evaluation.scoring.package_scoring import (
    score_package_record,
)
from tests.helpers.gold_record_builder import (
    JUDGE_CONFIG,
    load_source_package,
    perfect_judge_transport,
    perfect_record,
    project_record,
)


TEXT_METRIC_IDS = (
    "M01",
    "M04",
    "M05",
    "M09",
    "M11",
    "M14",
    "M15",
    "M16",
    "M17",
    "M19",
    "M24",
)
REPRESENTATIVE_SCENARIO = ("hotel", "htl-001")
ERROR_SCENARIO = ("hotel", "htl-001")
UPDATE_TOOL_ID = "update_planned_checkin_time"
TICKET_TOOL_ID = "register_case_record"
IDENTITY_TOOL_ID = "verify_identity"
RESERVATION_LOOKUP_TOOL_ID = "lookup_reservation_by_guest"
SOP_SEARCH_TOOL_ID = "search_sops_in_category"
PHONE_NUMBER = "090-0000-0035"
DISCLOSURE_MARKER = "無連絡キャンセル"
CONFIRMATION_MARKER = "登録してよろしいでしょうか"
NO_ANSWER_QUESTION_BY_DEFECT = {
    "m05_disclosure_after_deadline": ":M05:",
    "m15_unspoken_ticket_claim": ":M15:",
    "m16_summary_omitted": ":M16:M2",
    "m19_difficult_request_ignored": ":M19:CQ1",
}
CASES = (
    (*REPRESENTATIVE_SCENARIO, "m01_final_world_mismatch", "M01", None),
    (*REPRESENTATIVE_SCENARIO, "m04_action_before_identity", "M04", None),
    (*REPRESENTATIVE_SCENARIO, "m05_disclosure_after_deadline", "M05", None),
    (*REPRESENTATIVE_SCENARIO, "m09_empty_sop_search", "M09", None),
    (*REPRESENTATIVE_SCENARIO, "m11_required_tool_missing", "M11", "required_success"),
    (*REPRESENTATIVE_SCENARIO, "m11_argument_value_wrong", "M11", "argument_value"),
    (*REPRESENTATIVE_SCENARIO, "m11_argument_source_unknown", "M11", "argument_provenance"),
    (*REPRESENTATIVE_SCENARIO, "m11_dependency_reversed", "M11", "dependency_order"),
    (*ERROR_SCENARIO, "m11_injected_error_mismatch", "M11", "error_recovery"),
    (*REPRESENTATIVE_SCENARIO, "m14_ticket_field_wrong", "M14", None),
    (*REPRESENTATIVE_SCENARIO, "m15_unspoken_ticket_claim", "M15", None),
    (*REPRESENTATIVE_SCENARIO, "m16_summary_omitted", "M16", None),
    (*REPRESENTATIVE_SCENARIO, "m17_malformed_polite_sequence", "M17", None),
    (*REPRESENTATIVE_SCENARIO, "m19_difficult_request_ignored", "M19", None),
    (*REPRESENTATIVE_SCENARIO, "m24_written_markup", "M24", None),
)


def _resequence(events: list[dict[str, Any]]) -> None:
    for seq, event in enumerate(events, 1):
        event["seq"] = seq


def _call_result_block(
    events: list[dict[str, Any]], tool_id: str, *, ok: bool
) -> list[dict[str, Any]]:
    for index, event in enumerate(events[:-1]):
        result = events[index + 1]
        if (
            event["event_type"] == "tool_call"
            and event.get("tool") == tool_id
            and result["event_type"] == "tool_result"
            and result.get("tool") == tool_id
            and bool(result["result"].get("ok")) is ok
        ):
            return events[index : index + 2]
    raise AssertionError(f"missing {tool_id} call/result block")


def inject_defect(record: dict[str, Any], defect_kind: str) -> None:
    """Inject one named defect into a complete ``perfect_record`` object.

    ``record`` must contain the object/list fields emitted by ``perfect_record``;
    unsupported names raise ``ValueError``. Event-derived projections are updated
    by the caller after this function returns.
    """

    events = record["event_log"]
    if defect_kind == "m01_final_world_mismatch":
        record["final_world"][next(iter(record["final_world"]))] = []
    elif defect_kind == "m04_action_before_identity":
        failed = deepcopy(
            _call_result_block(events, UPDATE_TOOL_ID, ok=False)
        )
        trigger_index = next(
            index for index, event in enumerate(events) if event.get("persona_fired")
        )
        for suffix, event in zip(("call", "result"), failed, strict=True):
            ref = f"event:{record['run_id']}:defect:m04:{suffix}"
            event.update(event_id=ref, event_ref=ref, event_aliases=[])
        events[trigger_index + 1 : trigger_index + 1] = failed
        _resequence(events)
    elif defect_kind == "m05_disclosure_after_deadline":
        disclosure_index = next(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "message"
            and DISCLOSURE_MARKER in event["content"]
        )
        disclosure = events.pop(disclosure_index)
        deadline_index = max(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "tool_result"
            and event.get("tool") == UPDATE_TOOL_ID
            and event["result"].get("ok")
        )
        events.insert(deadline_index + 1, disclosure)
        _resequence(events)
    elif defect_kind == "m09_empty_sop_search":
        result = next(
            event["result"]
            for event in events
            if event["event_type"] == "tool_result"
            and event.get("tool") == SOP_SEARCH_TOOL_ID
        )
        result["candidates"] = []
    elif defect_kind == "m11_required_tool_missing":
        record["event_log"] = [
            event for event in events if event.get("tool") != TICKET_TOOL_ID
        ]
        _resequence(record["event_log"])
    elif defect_kind == "m11_argument_value_wrong":
        call = _call_result_block(events, UPDATE_TOOL_ID, ok=True)[0]
        call["arguments"]["late_arrival_notified"] = False
    elif defect_kind == "m11_argument_source_unknown":
        message = next(
            event
            for event in events
            if event["event_type"] == "message" and PHONE_NUMBER in event["content"]
        )
        message["content"] = message["content"].replace(f"、{PHONE_NUMBER}", "")
    elif defect_kind == "m11_dependency_reversed":
        first = _call_result_block(events, IDENTITY_TOOL_ID, ok=True)
        second = _call_result_block(events, RESERVATION_LOOKUP_TOOL_ID, ok=True)
        start = events.index(first[0])
        end = events.index(second[-1]) + 1
        events[start:end] = second + first
        _resequence(events)
    elif defect_kind == "m11_injected_error_mismatch":
        result = next(
            event["result"]
            for event in events
            if event["event_type"] == "tool_result" and event["result"].get("error")
        )
        result["error"] = "テスト用の未知のエラー"
    elif defect_kind == "m14_ticket_field_wrong":
        record["operator_ticket_artifact"]["ticket"]["handling_type"] = "refusal"
    elif defect_kind == "m15_unspoken_ticket_claim":
        record["operator_ticket_artifact"]["ticket"]["answer_summary"] = (
            "実際には案内していない内容です。"
        )
    elif defect_kind == "m16_summary_omitted":
        message = next(
            event
            for event in events
            if event["event_type"] == "message"
            and CONFIRMATION_MARKER in event["content"]
        )
        message["content"] = "登録してよろしいでしょうか。"
    elif defect_kind == "m17_malformed_polite_sequence":
        message = next(
            event
            for event in reversed(events)
            if event["event_type"] == "message" and event["actor"] == "operator"
        )
        message["content"] = "内容を確認しますです。"
    elif defect_kind == "m19_difficult_request_ignored":
        message = next(
            event
            for event in reversed(events)
            if event["event_type"] == "message" and event["actor"] == "operator"
        )
        message["content"] = "通常の手順で進めます。"
    elif defect_kind == "m24_written_markup":
        message = next(
            event
            for event in reversed(events)
            if event["event_type"] == "message" and event["actor"] == "operator"
        )
        message["content"] = "# 対応内容\n処理を続けます。"
    else:
        raise ValueError(f"unsupported defect kind: {defect_kind}")


def _judge_transport(defect_kind: str) -> Callable[..., dict[str, Any]]:
    marker = NO_ANSWER_QUESTION_BY_DEFECT.get(defect_kind)

    def transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        response = perfect_judge_transport(url, payload, headers, timeout)
        schema_name = payload["response_format"]["json_schema"]["name"]
        if marker and schema_name != "m05_consent_equivalence":
            content = json.loads(response["choices"][0]["message"]["content"])
            for judgement in content["judgements"]:
                if marker in judgement["question_id"]:
                    judgement.update(answer="no", evidence=[])
            response["choices"][0]["message"]["content"] = json.dumps(content)
        return response

    return transport


def _score_value(row: dict[str, Any]) -> float | None:
    if row["status"] == "N/A":
        return None
    if row["metric_id"] in {"M09", "M11"}:
        return row["value"]["applicable_pass_rate"]
    if row["metric_id"] == "M16":
        return row["value"]["fired_item_pass_rate"]
    return 1.0 if row["status"] == "pass" else 0.0


@pytest.mark.parametrize(
    ("domain", "scenario_id", "defect_kind", "target_metric", "target_facet"),
    CASES,
    ids=[case[2] for case in CASES],
)
def test_single_defect_only_reduces_its_target_metric(
    domain: str,
    scenario_id: str,
    defect_kind: str,
    target_metric: str,
    target_facet: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = load_source_package(domain, scenario_id)
    baseline_record = perfect_record(package)
    record = deepcopy(baseline_record)
    inject_defect(record, defect_kind)
    if not defect_kind.startswith(("m01_", "m14_", "m15_")):
        project_record(record, package)
    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")

    def score(
        candidate: dict[str, Any],
        transport: Callable[..., dict[str, Any]],
        name: str,
    ) -> dict[str, dict[str, Any]]:
        report = score_package_record(
            package=package,
            record=candidate,
            judge_config=JUDGE_CONFIG,
            cache_dir=tmp_path / name,
            transport=transport,
        )
        return {row["metric_id"]: row for row in report["metric_results"]}

    baseline = score(baseline_record, perfect_judge_transport, "baseline")
    actual = score(record, _judge_transport(defect_kind), "defect")

    assert _score_value(baseline[target_metric]) == 1.0
    if defect_kind == "m09_empty_sop_search":
        assert actual["M09"]["value"]["search_hit_correct"] is False
        assert _score_value(actual[target_metric]) == 1.0
    else:
        assert _score_value(actual[target_metric]) < 1.0
    for metric_id in TEXT_METRIC_IDS:
        if metric_id != target_metric:
            assert _score_value(actual[metric_id]) == _score_value(
                baseline[metric_id]
            ), metric_id
    if target_facet:
        facets = actual["M11"]["diagnostics"]["source_result"]["details"]["sub_facets"]
        assert facets[target_facet]["passed"] is False
