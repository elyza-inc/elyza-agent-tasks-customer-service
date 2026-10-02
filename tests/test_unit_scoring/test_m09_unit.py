"""Direct M09 calculation tests with paired synthetic tool events."""

from __future__ import annotations

import pytest

from elyza_agent_tasks_customer_service.evaluation.scoring import m09


def test_required_path_variants_and_errors():
    assert m09.required_sop_path({"expected_sop_path": ["a"]}) == ["a"]
    for value, message in [([], "scenario must"), ({}, "non-empty string array"), ({"expected_sop_path": "a"}, "non-empty string array"),
                           ({"expected_sop_path": ["a", ""]}, "non-empty string array")]:
        with pytest.raises(ValueError, match=message):
            m09.required_sop_path(value)


def test_calculate_m09_success_and_failure():
    scenario = {"expected_sop_path": ["s1"]}
    events = [
        {"seq": 1, "event_type": "tool_call", "tool": m09.SOP_SEARCH_TOOL, "arguments": {}},
        {"seq": 2, "event_type": "tool_result", "tool": m09.SOP_SEARCH_TOOL, "result": {"items": [{"sop_id": "s1"}]}},
        {"seq": 3, "event_type": "tool_call", "tool": m09.SOP_DETAIL_TOOL, "arguments": {"sop_id": "s1"}},
        {"seq": 4, "event_type": "tool_result", "tool": m09.SOP_DETAIL_TOOL, "result": {"sop": {"id": "s1", "steps": [{"kind": "tool", "tool_id": "apply"}]}}},
        {"seq": 5, "event_type": "tool_call", "tool": "apply", "arguments": {}},
        {"seq": 6, "event_type": "tool_result", "tool": "apply", "result": {"success": True}},
    ]
    result = m09.calculate_m09(scenario=scenario, events=events)
    assert result["final_correct"] and result["one_shot_correct"] and result["applications"][0]["applied"]
    assert (result["k"], result["score_inverse_k"]) == (1, 1.0)
    assert result["search_hit_correct"] and result["applications"][0]["executable_tool"] == "apply"
    failed = m09.calculate_m09(scenario={"expected_sop_path": ["s1"]}, events=[{"event_type": "tool_call", "tool": m09.SOP_DETAIL_TOOL, "arguments": {"sop_id": "s1"}}])
    assert not failed["final_correct"] and not failed["one_shot_correct"]


def test_m09_scores_inverse_detail_open_count_until_required_sop_is_reached():
    scenario = {"expected_sop_path": ["correct"]}
    events = [
        {"seq": 1, "event_type": "tool_call", "tool": m09.SOP_DETAIL_TOOL, "arguments": {"sop_id": "wrong"}},
        {"seq": 2, "event_type": "tool_result", "tool": m09.SOP_DETAIL_TOOL, "result": {"sop_id": "wrong"}},
        {"seq": 3, "event_type": "tool_call", "tool": m09.SOP_DETAIL_TOOL, "arguments": {"sop_id": "correct"}},
        {"seq": 4, "event_type": "tool_result", "tool": m09.SOP_DETAIL_TOOL, "result": {"sop_id": "correct"}},
    ]

    result = m09.calculate_m09(scenario=scenario, events=events)

    assert result["final_correct"] is True
    assert (result["k"], result["score_inverse_k"]) == (2, 0.5)
    assert m09.calculate_m09(scenario=scenario, events=events[:2])["score_inverse_k"] == 0.0


