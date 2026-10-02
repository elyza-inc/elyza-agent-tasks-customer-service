"""Direct M04/M05 contract and predicate branch tests."""

from __future__ import annotations

import pytest

from elyza_agent_tasks_customer_service.evaluation.scoring import core_metric_scoring as core


def _m04():
    return {"schema_version": core.M04_CONTRACT_VERSION, "edges": [{"edge_id": "e", "customer_trigger_type": "t", "critical_action_tool_id": "act", "required_precondition_tool_ids": ["pre"]}]}


def _m05():
    return {"schema_version": core.M05_CONTRACT_VERSION, "disclosure_questions": [{"question_id": "q", "question_text": "x", "candidate_span_refs": ["a"], "source_refs": ["b"]}],
            "obligations": [{"obligation_id": "o", "trigger_event_id": "trigger", "deadline_event_id": "deadline", "required_event_ids": ["required"], "stop_event_id": None, "prohibited_event_ids": [], "consent_event_id": None, "protected_action_event_id": None, "required_disclosure_question_ids": ["q"]}]}


def test_m04_tool_statuses():
    assert core.score_m04_customer_trigger_control(contract=None, call_events=[])["status"] == "N/A"
    assert core.score_m04_customer_trigger_control(contract={}, call_events=[])["status"] == "contract_invalid"
    assert core.score_m04_customer_trigger_control(contract=_m04(), call_events=[])["status"] == "N/A"
    tool = _m04()
    assert core.score_m04_customer_trigger_control(contract=tool, call_events=[{"persona_fired": ["t"]}, {"event_type": "tool_call", "tool": "pre"}, {"event_type": "tool_call", "tool": "act"}])["status"] == "pass"
    assert core.score_m04_customer_trigger_control(contract=tool, call_events=[{"persona_fired": "t"}, {"event_type": "tool_call", "tool": "act"}])["status"] == "fail"
    with pytest.raises(ValueError, match="persona_fired"):
        core._persona_fired_positions([{"persona_fired": [1]}], "t")


