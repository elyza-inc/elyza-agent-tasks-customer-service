"""Mutation boundaries for the deterministic post-call fact builder."""

from __future__ import annotations


from elyza_agent_tasks_customer_service.evaluation.scoring import ticket_fact_builder as facts


def _result(tool="work", result=None, **extra):
    return {"event_type": "tool_result", "tool": tool, "result": result or {}, **extra}


def test_fact_builder_success_and_reference_boundaries():
    no_receipt = facts.build_ticket_fact([{"event_type": "tool_call", "tool": "work"}, _result()])
    assert no_receipt["performed_actions"] == [{"action_code": "work", "target_ref": None, "outcome": "succeeded", "receipt_refs": []}]
    assert facts._result_targets(_result(result={"rows": [{"id": "x", "bad": "ignored"}]})) == [{"id_type": "id", "value": "x"}]
    assert facts._sop_evidence_ref(_result(result={"sop_id": "x", "sop_ref": ""})) == "SOP:x"
    assert not facts._event_succeeded(_result(result={"ok": False}))


def test_recorded_handling_type_reads_rows_the_run_created_or_changed() -> None:
    """Incomplete runs take handling from changed world rows, mapped to the ticket enum."""

    from elyza_agent_tasks_customer_service.evaluation.scoring.ticket_fact_builder import recorded_handling_type

    row = lambda seq, value: {"row_identity": {"customer": "c", "sequence": seq}, "values": {"handling_type": value}}
    initial = {"case_records": [row(1, "change")]}
    assert recorded_handling_type(initial, {"case_records": [row(1, "change")]}) is None
    assert recorded_handling_type(initial, {"case_records": [row(1, "change"), row(2, "escalation")]}) == "escalation"
    assert recorded_handling_type({}, {"case_records": [row(1, "guidance")]}) == "guidance_inquiry_answer"
    assert recorded_handling_type({}, {"case_records": [row(1, "guidance"), row(2, "change")]}) == "change_procedure"
    assert recorded_handling_type({}, {"case_records": [row(1, "unknown")]}) is None
