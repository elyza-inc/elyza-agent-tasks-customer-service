"""Uncovered text-metric ledger rows, using the smallest direct scorer inputs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from elyza_agent_tasks_customer_service.evaluation.engine.package_adapter import (
    _japanese_register_context,
    convert_package,
)
from elyza_agent_tasks_customer_service.evaluation.observers.japanese_parse_adapter import (
    JapaneseParserDependencyError,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.conversation_log_metrics import (
    score_conversation_log_metrics,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.conversation_log_scoring import (
    score_conversation,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.core_metric_scoring import (
    score_m05_obligations,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.m09 import calculate_m09
from elyza_agent_tasks_customer_service.evaluation.scoring.metric_bridge import (
    _m24_result_from_m17,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.package_scoring import (
    score_package_record,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.post_call_ticket_scoring import (
    score_post_call_ticket,
)
from tests.helpers.gold_record_builder import (
    JUDGE_CONFIG,
    load_source_package,
    perfect_judge_transport,
    perfect_record,
)
from tests.test_single_defect_sweep import inject_defect


HOTEL = ("hotel", "htl-001")
NO_ERROR_HOTEL = ("hotel", "htl-011")
M05_VERSION = "m05_obligation_contract"
M17_INSTANCE_ID = "m17:matrix"


def _rows(
    package: dict[str, Any],
    record: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Callable[..., dict[str, Any]] = perfect_judge_transport,
) -> dict[str, dict[str, Any]]:
    """Score one complete package record with the fixture judge transport."""

    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")
    report = score_package_record(
        package=package,
        record=record,
        judge_config=JUDGE_CONFIG,
        cache_dir=tmp_path,
        transport=transport,
    )
    return {row["metric_id"]: row for row in report["metric_results"]}


def _no_answers(*markers: str, consent: bool = False) -> Callable[..., dict[str, Any]]:
    """Return a valid fixture judge that answers no only for selected question IDs."""

    def transport(
        url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> dict[str, Any]:
        response = perfect_judge_transport(url, payload, headers, timeout)
        schema = payload["response_format"]["json_schema"]["name"]
        if schema == "m05_consent_equivalence":
            if consent:
                content = json.loads(response["choices"][0]["message"]["content"])
                for judgement in content["judgements"]:
                    judgement["answer"] = "no"
                response["choices"][0]["message"]["content"] = json.dumps(content)
            return response
        content = json.loads(response["choices"][0]["message"]["content"])
        for judgement in content["judgements"]:
            if any(marker in judgement["question_id"] for marker in markers):
                judgement.update(answer="no", evidence=[])
        response["choices"][0]["message"]["content"] = json.dumps(content)
        return response

    return transport


def _m05_contract(*, consent: bool = False) -> dict[str, Any]:
    """Build one exact M05 contract for deterministic predicate tests."""

    obligation: dict[str, Any] = {
        "obligation_id": "obligation:matrix",
        "deadline_tool_id": "act",
        "deadline_event_id": "deadline",
        "required_event_ids": ["required"],
        "consent_event_id": "consent" if consent else None,
        "protected_action_event_id": "protected" if consent else None,
    }
    return {"schema_version": M05_VERSION, "obligations": [obligation]}


def _event(event_id: str, **values: Any) -> dict[str, Any]:
    """Build a call event with its stable ID required by M05 predicates."""

    return {"event_id": event_id, "event_ref": event_id, "event_aliases": [], **values}


def test_m05_consent_missing_before_protected_action_fails() -> None:
    """台帳B/M05 F4: consentがprotected action前にない単一欠陥は対象obligationをfailにする。"""

    events = [_event("trigger", event_type="tool_call", tool="act"), _event("required"), _event("deadline"), _event("protected")]
    row = score_m05_obligations(contract=_m05_contract(consent=True), call_events=events)
    obligation = row["value"]["obligations"][0]

    assert (row["status"], row["reason"]) == ("fail", "one_or_more_m05_obligations_failed")
    assert obligation["obligation_id"] == "obligation:matrix"
    assert obligation["deterministic_checks"]["consent_before_protected_action"] is False


def test_m05_consent_equivalence_judge_no_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳B/M05 F6 consent equivalence judge=no を固定する。"""

    package = load_source_package("hotel", "htl-006")
    rows = _rows(package, perfect_record(package), tmp_path, monkeypatch, _no_answers(consent=True))

    assert (rows["M05"]["status"], rows["M05"]["reason"]) == (
        "fail",
        "semantically_matching_consent_not_observed",
    )
    assert rows["M05"]["value"]["obligations"][0]["obligation_id"] == "htl-006:m05"
    assert rows["M05"]["diagnostics"]["semantic_consent"]["judgements"][0]["question_id"] == "htl-006:M05:recalc_agreement:0000"
    assert rows["M05"]["diagnostics"]["semantic_consent"]["judgements"][0]["answer"] == "no"


@pytest.mark.parametrize(
    ("ledger", "scenario", "match"),
    [
        ("M09 NA1 expected SOPなし", {"scenario_id": "m09:na"}, "scenario.expected_sop_path must be a non-empty string array"),
        ("M09 X1 trace/expected SOP shape不正", {"expected_sop_path": "bad"}, "scenario.expected_sop_path must be a non-empty string array"),
    ],
)
def test_m09_unavailable_contracts_raise_the_fixed_exception(
    ledger: str, scenario: dict[str, Any], match: str
) -> None:
    """台帳B/M09 NA1, X1: adapter外のM09 contract欠落/不正は例外で固定する。"""

    with pytest.raises(ValueError, match=match):
        calculate_m09(scenario=scenario, events=[])


def test_m11_natural_error_without_retry_fails_only_recovery_facet() -> None:
    """台帳B/M11 F-ER2: 自然エラーを放置するとerror_recoveryがfailになる。"""

    package = load_source_package(*NO_ERROR_HOTEL)
    record = perfect_record(package)
    expected_tool = package["execution_trace"][0]["tool_id"]
    failed_call = next(
        event for event in record["event_log"] if event.get("event_type") == "tool_call" and event.get("tool") == expected_tool
    )
    failed_result = {
        "event_id": "event:natural:error",
        "event_ref": "event:natural:error",
        "event_aliases": [],
        "episode_id": record["run_id"],
        "seq": failed_call["seq"] + 1,
        "epoch": "call",
        "actor": "tool",
        "event_type": "tool_result",
        "tool": failed_call["tool"],
        "result": {"ok": False, "error": "natural timeout"},
    }
    failed_call = deepcopy(failed_call)
    failed_call.update(event_id="event:natural:call", event_ref="event:natural:call", event_aliases=[])
    bad_retry = deepcopy(failed_call)
    bad_retry.update(event_id="event:natural:bad-retry", event_ref="event:natural:bad-retry")
    bad_retry["arguments"][next(iter(bad_retry["arguments"]))] = "wrong-value"
    bad_retry_result = deepcopy(failed_result)
    bad_retry_result.update(
        event_id="event:natural:bad-retry-result",
        event_ref="event:natural:bad-retry-result",
        result={"ok": True},
    )
    record["event_log"].extend((failed_call, failed_result, bad_retry, bad_retry_result))
    for sequence, event in enumerate(record["event_log"], 1):
        event["seq"] = sequence
    result = score_conversation_log_metrics(
        scenario=convert_package(package, mode="text"),
        event_log=record["event_log"],
        run_id=record["run_id"],
    )
    facet = result["M11"]["details"]["sub_facets"]["error_recovery"]

    assert (facet["status"], facet["reason"]) == (
        "fail",
        "one_or_more_naturally_observed_retry_series_not_recovered_correctly",
    )
    assert facet["details"]["series"][0]["error_attempt"]["tool"] == expected_tool


def test_m11_error_recovery_is_na_when_no_error_is_observed() -> None:
    """台帳B/M11 NA-ER: エラーなしではerror_recoveryだけN/Aになる。"""

    package = load_source_package(*NO_ERROR_HOTEL)
    record = perfect_record(package)
    result = score_conversation_log_metrics(
        scenario=convert_package(package, mode="text"),
        event_log=record["event_log"],
        run_id=record["run_id"],
    )
    facet = result["M11"]["details"]["sub_facets"]["error_recovery"]

    assert (facet["status"], facet["reason"], facet["passed"]) == (
        "not_applicable",
        "no_naturally_observed_error_to_retry_series",
        None,
    )


def test_m14_invalid_ticket_contract_is_contract_invalid() -> None:
    """台帳B/M14 CI1: ticket contract不正はM14 contract_invalidを返す。"""

    row = score_post_call_ticket(
        contract={}, artifact=None, initial_world={}, final_world={}, call_events=[]
    )

    assert (row["schema_version"], row["status"]) == (
        "post_call_ticket_scoring",
        "contract_invalid",
    )
    assert row["reason"] == "deterministic ticket contract fields are invalid"


@pytest.mark.parametrize(
    ("ledger", "field", "value", "marker"),
    [
        ("M15 F1 refusal_reasonが会話にない", "refusal_reason", "decline", "ticket:refusal_reason_code"),
        ("M15 F2 promisesが会話にない", "promises", [{"owner_id": "agent", "channel": "phone", "deadline": "tomorrow"}], "ticket:promises[0]:callback"),
    ],
)
def test_m15_unspoken_ticket_claims_fail_with_their_question_id(
    ledger: str, field: str, value: Any, marker: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳B/M15 F1, F2: 未発話ticket claimは対応question IDでfailになる。"""

    package = load_source_package(*HOTEL)
    record = perfect_record(package)
    record["operator_ticket_artifact"]["ticket"][field] = value
    rows = _rows(package, record, tmp_path, monkeypatch, _no_answers(marker))
    judgement = next(item for item in rows["M15"]["value"]["judgements"] if marker in item["question_id"])

    assert (rows["M15"]["status"], rows["M15"]["reason"]) == ("fail", "one_or_more_judgements_mismatch"), ledger
    assert (judgement["question_id"], judgement["correct"]) == (f"htl-001:M15:{marker}", False), ledger


def test_m15_without_conversational_claims_uses_contract_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳B/M15: conversational claim不足は正解契約で補完する。"""

    package = load_source_package(*HOTEL)
    record = perfect_record(package)
    ticket = record["operator_ticket_artifact"]["ticket"]
    ticket.update(refusal_reason=None, promises=[], answer_summary=None)
    rows = _rows(package, record, tmp_path, monkeypatch)

    assert (rows["M15"]["status"], rows["M15"]["reason"]) == ("pass", "all_judgements_match")
    assert [row["expected_answer"] for row in rows["M15"]["value"]["judgements"]] == ["yes", "no", "no"]


@pytest.mark.parametrize(
    ("ledger", "marker", "criterion_id"),
    [
        ("M16 F-M1 待ち機会にwait案内なし", ":M16:M1", "M16-M1"),
        ("M16 F-M4 重要値の復唱確認なし", ":M16:M4", "M16-M4"),
    ],
)
def test_m16_missing_judged_interaction_is_fixed_to_criterion(
    ledger: str, marker: str, criterion_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳B/M16 F-M1, F-M4: no judge answerは対象criterionだけをfailにする。"""

    package = load_source_package(*HOTEL)
    rows = _rows(package, perfect_record(package), tmp_path, monkeypatch, _no_answers(marker))
    observation = next(
        item for item in rows["M16"]["value"]["machine_observations"] if item["criterion_id"] == criterion_id
    )

    assert (rows["M16"]["status"], rows["M16"]["reason"]) == ("measured", "independent_judge_rows_with_fired_item_pass_rate"), ledger
    assert (observation["criterion_id"], observation["status"], observation["passed"]) == (criterion_id, "fail", False), ledger


def test_m16_without_important_values_marks_only_m4_not_measurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳B/M16 NA-M4: important valuesなしではM4だけ非適用状態を固定する。"""

    package = load_source_package(*HOTEL)
    package["contracts"]["m20"]["important_values"] = []
    observations = _rows(package, perfect_record(package), tmp_path, monkeypatch)["M16"]["value"]["machine_observations"]
    m4 = next(item for item in observations if item["criterion_id"] == "M16-M4")

    assert (m4["criterion_id"], m4["status"], m4["passed"]) == ("M16-M4", "N/M", None)
    assert m4["values"]["reason"] == "explicitly_not_applicable"


def test_m16_invalid_integration_contract_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """台帳B/M16 X1: interaction contract不正は統合経路で例外になる。"""

    package = load_source_package(*HOTEL)
    package["contracts"]["m20"] = {"important_values": "bad"}

    with pytest.raises(ValueError, match="htl-001: package.contracts.m20.important_values: list is required"):
        _rows(package, perfect_record(package), tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("ledger", "marker", "question_id"),
    [
        ("M19 F-CQ2 escalation要求への応答なし", ":M19:CQ2", "htl-001:M19:CQ2"),
        ("M19 F-CQ3 correction/emotionへの応答なし", ":M19:CQ3", "htl-001:M19:CQ3"),
    ],
)
def test_m19_unanswered_persona_followups_fail_with_question_id(
    ledger: str, marker: str, question_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳B/M19 F-CQ2, F-CQ3: persona follow-upのjudge noを対象IDで固定する。"""

    package = load_source_package(*HOTEL)
    rows = _rows(package, perfect_record(package), tmp_path, monkeypatch, _no_answers(marker))
    judgement = next(item for item in rows["M19"]["value"]["judgements"] if item["question_id"] == question_id)

    assert (rows["M19"]["status"], rows["M19"]["reason"]) == ("fail", "one_or_more_judgements_mismatch"), ledger
    assert (judgement["question_id"], judgement["correct"]) == (question_id, False), ledger


def test_m19_without_persona_is_not_applicable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """台帳B/M19 NA1: personaなしではM19はN/Aになる。"""

    package = load_source_package(*HOTEL)
    package["persona"] = {key: None for key in package["persona"]}
    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")
    report = score_conversation(
        package=package,
        record=perfect_record(package),
        config=JUDGE_CONFIG,
        cache_dir=tmp_path,
        transport=perfect_judge_transport,
    )
    rows = {row["metric_id"]: row for row in report["metric_results"]}

    assert (rows["M19"]["status"], rows["M19"]["reason"]) == ("N/A", "explicitly_not_applicable")
    assert rows["M19"]["judgements"] == []


def _m17_context() -> dict[str, Any]:
    """Return the representative package's valid explicit M17 relation context."""

    package = load_source_package(*HOTEL)
    return _japanese_register_context(package, package["scenario_id"])


def test_m17_parser_startup_failure_is_not_measurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """台帳B/M17 NM1: parser依存の起動不可はN/Mとして収束する。"""

    import elyza_agent_tasks_customer_service.evaluation.scoring.extended_metric_scoring as extended

    package = load_source_package(*HOTEL)
    scenario = {
        "scenario_id": package["scenario_id"],
        "japanese_register_context": _m17_context(),
        "measurement_contract": {"metric_instances": [{"metric_instance_id": M17_INSTANCE_ID, "metric_id": "M17", "contract": {}}]},
    }
    monkeypatch.setattr(extended, "score_japanese_register_from_parse", lambda **_: (_ for _ in ()).throw(JapaneseParserDependencyError("missing parser")))
    row = extended.score_extended_metrics(
        scenario=scenario, artifacts={"m17_parse_bundle": {}}
    )["metric_results"][0]

    assert (row["metric_instance_id"], row["status"], row["reason"]) == (
        M17_INSTANCE_ID,
        "N/M",
        "pinned_japanese_parser_dependency_unavailable",
    )


def test_m17_invalid_context_is_contract_invalid() -> None:
    """台帳B/M17 CI1: context不正はM17 contract_invalid rowになる。"""

    import elyza_agent_tasks_customer_service.evaluation.scoring.extended_metric_scoring as extended

    row = extended.score_extended_metrics(
        scenario={
            "scenario_id": "m17:invalid",
            "japanese_register_context": {},
            "measurement_contract": {"metric_instances": [{"metric_instance_id": M17_INSTANCE_ID, "metric_id": "M17", "contract": {}}]},
        },
        artifacts={},
    )["metric_results"][0]

    assert (row["metric_instance_id"], row["status"]) == (M17_INSTANCE_ID, "contract_invalid")
    assert row["reason"] == "Japanese register context fields are invalid"


def test_m24_missing_m17_assessment_is_not_measurable() -> None:
    """台帳B/M24 NM2: M17 assessmentなしはM24 N/Mになる。"""

    row = _m24_result_from_m17({"metric_instance_id": M17_INSTANCE_ID, "status": "N/M", "value": None, "diagnostics": {}})

    assert (row["metric_instance_id"], row["status"], row["reason"]) == (
        f"{M17_INSTANCE_ID}:derived:M24",
        "N/M",
        "m17_utterance_assessments_unavailable",
    )


def test_m24_invalid_spoken_style_shape_raises() -> None:
    """台帳B/M24 CI1: M17 spoken_style shape不正は例外として固定する。"""

    with pytest.raises(ValueError, match="M17 spoken_style assessment is missing"):
        _m24_result_from_m17(
            {
                "metric_instance_id": M17_INSTANCE_ID,
                "status": "pass",
                "value": {"utterance_assessments": [{"turn_ref": "operator:bad"}]},
                "diagnostics": {},
            }
        )


def test_imported_single_defect_helper_remains_compatible() -> None:
    """台帳Bの共通欠陥注入helperを本matrixでも再利用できることを固定する。"""

    package = load_source_package(*HOTEL)
    record = perfect_record(package)
    inject_defect(record, "m14_ticket_field_wrong")

    assert record["operator_ticket_artifact"]["ticket"]["handling_type"] == "refusal"
    assert record["scenario_id"] == "htl-001"
