"""Boundary checks for offline conversation-log scorer helpers."""

from __future__ import annotations


import pytest

from elyza_agent_tasks_customer_service.evaluation.scoring import conversation_log_metrics as metrics
from elyza_agent_tasks_customer_service.evaluation.scoring import conversation_log_scoring as judge


def _event(seq, event_type, tool=None, **extra):
    row = {"seq": seq, "event_type": event_type}
    if tool is not None:
        row["tool"] = tool
    row.update(extra)
    return row


def _metric_record(run_id="r"):
    facet = {"status": "pass"}
    return {
        "schema_version": metrics.SCHEMA_VERSION,
        "scenario_id": "s",
        "run_id": run_id,
        "M09": {
            "search_hit_correct": facet,
            "final_correct": facet,
            "one_shot_correct": facet,
        },
        "M11": {
            "status": "pass",
            "details": {"sub_facets": {name: facet for name in metrics.M11_FACETS}},
            "error_recovery": facet,
            "value": {"applicable_pass_rate": 1.0},
        },
        "M08": {"value": {"premature_dependency_operation_observed": False, "premature_attempt_count": 0, "evaluated_attempt_count": 1}},
        "resolution_and_major_violations": {
            "raw_M01": facet,
            "major_violation_fail_information": {
                metric_id: {"applicable": True, "measurable": True, "failed": False}
                for metric_id in ("M04", "M05")
            },
        },
    }


def test_judge_response_evidence_scoring_and_metadata_helpers(monkeypatch):
    source = {"ref": "conversation[0000]", "event": {"actor": "operator", "content": "x", "seq": 1}}
    task = judge._task("q", "M15", "k", "?", "yes", {"deadline_seq": 2}, ["operator_utterance_before_deadline"])
    schema = judge.judge_response_schema(["q"], [source["ref"]])
    decisions = judge._validate_response({"judgements": [{"question_id": "q", "answer": "yes", "evidence": [{"ref": source["ref"], "description": "x"}]}]}, [task], {"conversation": [source], "tool_calls": []}, schema)
    assert decisions["q"]["status"] == "valid"
    assert judge._validate_evidence_rules({**task, "evidence_rules": ["operator_utterance_after_trigger"], "relevant_context": {"trigger_seq": 2}}, "yes", [source["ref"]], {source["ref"]: source})
    with pytest.raises(ValueError, match="schema"):
        judge._validate_response({"judgements": [{"question_id": "other", "answer": "yes", "evidence": []}]}, [task], {"conversation": [source], "tool_calls": []}, schema)
    decision = {"status": "valid", "answer": "yes", "evidence": [], "contract_violations": []}
    rows = judge._score_tasks([judge._task("m16", "M16", "M16-M1", "?", "yes", {}, []), task], {"by_id": {"m16": decision, "q": decision}}, {"M05": "N/A", "M15": True, "M16": True, "M19": "N/A"})
    assert next(row for row in rows if row["metric_id"] == "M16")["status"] == "measured"
    evidence = judge._evidence({"conversation": [{"actor": "operator", "content": "x", "seq": 1}], "tool_calls": [{"tool_id": "t"}], "event_log": []})
    assert judge._source_kind(evidence["conversation"][0]) == "operator_utterance"
    assert judge._source_kind(evidence["tool_calls"][0]) == "tool"
    assert judge._important_values({"m20": None}) is None
    assert judge._important_values({"m20": {"important_values": []}}) is None
    assert judge._edge({"persona": {}}) is None
    edge = judge._edge({"persona": {"customer_pressure": {"utterance": "no"}}})
    assert judge._trigger_source(edge, {"event_log": [{"event": {"actor": "user", "persona_fired": "customer_pressure", "seq": 3}}]})["event"]["seq"] == 3
    with pytest.raises(ValueError, match="integer"):
        judge._event_seq({"ref": "x", "event": {"seq": True}})
    assert judge._judge_contract({"scenario_id": "s", "contracts": {}})["scenario_id"] == "s"
    monkeypatch.setattr(judge, "validate_judge_member", lambda *args, **kwargs: {"sampling": {"seed": 0}})
    with pytest.raises(ValueError, match="positive"):
        judge._validate_config_value({"schema_version": judge.CONFIG_VERSION, "judge": {}, "max_input_bytes": 0, "minimum_accuracy": 1})


def test_judge_tasks_and_all_not_applicable_score(monkeypatch, tmp_path):
    record = {"conversation": [], "tool_calls": [], "event_log": [], "operator_ticket_artifact": None}
    evidence = judge._evidence(record)
    package = {"scenario_id": "s", "contracts": {"m05": None, "m20": None}, "persona": {}}
    tasks, applicability = judge._tasks(package, record, evidence)
    assert [task["kind"] for task in tasks] == ["M16-M1", "M16-M2"]
    assert applicability["M05"] == "N/A" and applicability["M19"] == "N/A"
    monkeypatch.setattr(judge, "_validate_config_value", lambda value: {"judge": {"resolved_model_revision": "m", "sampling": {}}, "max_input_bytes": 1})
    score = judge.score_conversation(package=package, record=record, config={}, cache_dir=tmp_path)
    assert score["judge"]["api_call_count"] == 1
    deadline_package = {"scenario_id": "s", "contracts": {"m05": {"disclosure": "d", "deadline_tool_id": "t"}, "m20": {"important_values": [{"slot_id": "x", "value": 1, "value_type": "string", "question_step_id": "q", "readback_role": "readback"}]}}, "persona": {"customer_pressure": {"utterance": "x"}}}
    # The synthetic package has no gold dialogue; its declared deadline Tool is the bound.
    monkeypatch.setattr(judge, "m05_judge_deadline_tool", lambda package, scenario_id: "t")
    deadline_record = {"conversation": [], "tool_calls": [{"tool_id": "t", "ok": False, "seq": 3}], "event_log": [{"actor": "user", "persona_fired": ["customer_pressure"], "seq": 1}], "operator_ticket_artifact": None}
    tasks, applicability = judge._tasks(deadline_package, deadline_record, judge._evidence(deadline_record))
    assert {task["metric_id"] for task in tasks} == {"M05", "M16", "M19"}
    assert applicability["M15"] == "N/M"


def test_m05_judge_deadline_moves_to_whole_call_for_record_only_deadline(monkeypatch):
    """A case-record deadline is not a change operation: the disclosure bound becomes the whole call."""

    record = {"conversation": [{"actor": "operator", "content": "a", "seq": 9}], "tool_calls": [{"tool_id": "register_case_record", "ok": True, "seq": 4}], "event_log": [], "operator_ticket_artifact": None}
    package = {"scenario_id": "s", "contracts": {"m05": {"disclosure": "d", "deadline_tool_id": "register_case_record"}, "m20": None}, "persona": {}}
    monkeypatch.setattr(judge, "m05_judge_deadline_tool", lambda package, scenario_id: None)
    tasks, _ = judge._tasks(package, record, judge._evidence(record))
    m05 = [task for task in tasks if task["metric_id"] == "M05"][0]
    assert m05["relevant_context"]["deadline_seq"] > 9


def test_polite_end_accepts_yone_and_completion_match_normalizes_like_tools():
    """「ですよね」 is polite; completion matching ignores phone separators and kana spaces like tool matching."""

    from elyza_agent_tasks_customer_service.evaluation.engine.package_runtime import _values_match_for_completion
    from elyza_agent_tasks_customer_service.evaluation.observers.japanese_rule_observer import POLITE_END_PATTERN

    assert POLITE_END_PATTERN.search("ご不安ですよね") and not POLITE_END_PATTERN.search("ご不安だよね")
    assert _values_match_for_completion({"phone_number": "09000000050"}, {"phone_number": "090-0000-0050"})
    assert not _values_match_for_completion({"handling_type": "change"}, {"handling_type": "guidance"})
