"""Mutation boundaries for deterministic conversation-log metrics."""

from __future__ import annotations


from elyza_agent_tasks_customer_service.evaluation.scoring import conversation_log_metrics as metrics


def _attempt(tool, seq, outcome="success", *, args=None, result=None, metadata=None):
    call = {"event_type": "tool_call", "tool": tool, "seq": seq, "arguments": args or {}}
    if metadata is not None:
        call["metadata"] = metadata
    return {"attempt_index": seq, "tool": tool, "call_seq": seq, "result_seq": seq + 1, "args": args or {}, "result": result or {}, "outcome": outcome, "error": "bad" if outcome == "error" else None, "call_event": call, "result_event": {"event_type": "tool_result", "tool": tool, "seq": seq + 1, "result": result or {}}}


def test_argument_provenance_reattributes_an_unaligned_exact_success():
    scenario = {"tools": {"work": {"arguments": [{"name": "name", "required": True}]}}}
    expected = [{"trace_index": 0, "tool": "work", "args": {"name": "山田"}}]
    attempts = [_attempt("work", 2, args={"name": "山田"})]
    event_log = [{"seq": 1, "actor": "user", "event_type": "message", "content": "山田です"}]

    findings = metrics._call_findings(
        scenario, event_log, expected, attempts, {}, aligned_attempts=[]
    )

    field = findings[0]["fields"][0]
    assert (field["observed"], field["call_seq"], field["derived_source_kind"]) == (
        "山田",
        2,
        "customer_utterance",
    )
    assert metrics._score_argument_provenance(findings)["status"] == "pass"
    missing = metrics._call_findings(scenario, event_log, expected, [], {}, aligned_attempts=[])
    assert metrics._score_argument_provenance(missing)["status"] == "fail"
