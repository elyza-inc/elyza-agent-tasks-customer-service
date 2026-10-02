"""Boundary examples that distinguish bridge and canonical M09 mutations."""

from __future__ import annotations

from elyza_agent_tasks_customer_service.evaluation.scoring import metric_bridge as bridge


def _m17(utterances, status="pass"):
    return {"metric_id": "M17", "metric_instance_id": "m17", "status": status, "value": {"utterance_assessments": utterances}, "diagnostics": {}}


def test_bridge_m24_counts_and_availability_edges():
    missing = bridge._m24_result_from_m17({"metric_instance_id": "m17", "status": "pass"})
    assert missing["value"]["utterance_count"] == missing["value"]["utterance_pass_count"] == missing["value"]["violation_count"] == 0
    report = {"metric_results": [_m17([])], "violations": [{"metric_id": "M01"}, {"metric_id": "M24"}], "not_measured": [{"metric_id": "M01"}, {"metric_id": "M24"}], "diagnostics": {}}
    bridge._attach_m24_result(report)
    assert report["violations"] == [{"metric_id": "M01"}]
    assert report["not_measured"] == [{"metric_id": "M01"}, {"metric_instance_id": "m17:derived:M24", "metric_id": "M24", "reason": "operator_utterances_unavailable"}]
    assert "blocking_rule_families" not in bridge._availability_row("M17", {"status": "pass", "value": {"rule_families": [{"passed": False}]}})
    assert bridge._availability_row("M01", {"status": "N/A"})["availability"] == "not_applicable"
    assert bridge._availability_row("M01", {"status": "pass", "reason": ""})["reason"] == "pass"


def _call(tool, seq, arguments=None):
    return {"event_type": "tool_call", "tool": tool, "seq": seq, "arguments": arguments or {}}


def _result(tool, seq, result):
    return {"event_type": "tool_result", "tool": tool, "seq": seq, "result": result}


