from __future__ import annotations



from elyza_agent_tasks_customer_service.evaluation.scoring import (
    ticket_fact_builder,
)


def test_ticket_fact_builder_tracks_handling_targets_and_evidence() -> None:
    events = [
        {"event_type": "tool_call", "tool": "change", "event_ref": "call-1"},
        {
            "event_type": "tool_result",
            "tool": "change",
            "event_ref": "result-1",
            "result": {"ok": True, "rows": [{"customer_id": "customer-1"}]},
        },
        {"event_type": "tool_result", "tool": "sop", "event_ref": "sop-1", "result": {"ok": True, "sop_id": "billing"}},
    ]
    fact = ticket_fact_builder.build_ticket_fact(events, run_outcome={"handling_type": "change_procedure"}, claimed_evidence_refs=[{"kind": "event", "ref": "call-1"}, {"kind": "source", "ref": "missing"}])
    assert fact["handling_type"] == "change_procedure"
    assert fact["target_ids"] == [{"id_type": "customer_id", "value": "customer-1"}]
    assert {row["kind"] for row in fact["evidence_refs"]} == {"event", "tool_receipt", "sop"}


