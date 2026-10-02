"""Checks for the scorer fixes found by the 2026-09-30 all-models-fail audit."""

from __future__ import annotations

from elyza_agent_tasks_customer_service.evaluation.contracts.type_contracts import _normalized_match_value
from elyza_agent_tasks_customer_service.evaluation.scoring.post_call_ticket_scoring import compare_ticket_facts
from elyza_agent_tasks_customer_service.evaluation.scoring.ticket_fact_builder import build_ticket_fact


def test_email_match_ignores_letter_case() -> None:
    assert _normalized_match_value("CUSTOMER01@example.jp", "registered_email", None) == _normalized_match_value(
        "customer01@example.jp", "registered_email", None
    )


def _events() -> list[dict]:
    return [
        {"event_type": "tool_call", "tool": "get_sop", "event_ref": "e1"},
        {"event_type": "tool_result", "tool": "get_sop", "event_ref": "e2", "result": {"ok": False, "error": "x"}},
        {"event_type": "tool_call", "tool": "lookup", "event_ref": "e3"},
        {
            "event_type": "tool_result",
            "tool": "lookup",
            "event_ref": "e4",
            "result": {"ok": True, "rows": [{"guest_id": "G1", "area_id": "A1"}]},
        },
    ]


def test_failed_sop_call_is_not_a_performed_action_and_any_result_id_is_a_target() -> None:
    candidates: list[list[str]] = []
    fact = build_ticket_fact(_events(), target_candidates_out=candidates)
    assert [action["action_code"] for action in fact["performed_actions"]] == ["lookup"]
    observed = {**fact, "performed_actions": [{**fact["performed_actions"][0], "target_ref": "A1"}]}
    results = compare_ticket_facts(expected=fact, observed=observed, target_candidates=candidates)
    assert results["field_results"]["performed_actions"] is True
    wrong = {**fact, "performed_actions": [{**fact["performed_actions"][0], "target_ref": "OTHER"}]}
    assert compare_ticket_facts(expected=fact, observed=wrong, target_candidates=candidates)["field_results"]["performed_actions"] is False
