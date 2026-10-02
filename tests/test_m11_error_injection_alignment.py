"""M11 のエラー注入時トレース整列に関する回帰テスト。"""

from __future__ import annotations

import hashlib
import json
import unittest

from elyza_agent_tasks_customer_service.evaluation.scoring.conversation_log_metrics import (
    _attempts,
    _score_m11,
)


ERROR_MESSAGE = "更新に失敗しました"
LEGACY_NO_INJECTION_M11_SHA256 = (
    "ad1d2f0e698ea3800306ff302af51075e485f553baea94a3272c770df47b4749"
)


def _scenario(*, injected: bool) -> dict:
    lookup = {"tool": "lookup", "args": {"customer_id": "customer-1"}}
    update = {"tool": "update", "args": {"entity_id": "entity-1"}}
    expected_trace = [lookup, update, lookup]
    execution_trace = [
        {"tool_id": "lookup", "arguments": lookup["args"]},
        {"tool_id": "update", "arguments": update["args"]},
        {"tool_id": "lookup", "arguments": lookup["args"]},
    ]
    if injected:
        unlock = {"tool": "unlock", "args": {"entity_id": "entity-1"}}
        expected_trace = [lookup, update, unlock, update]
        execution_trace = [
            {"tool_id": "lookup", "arguments": lookup["args"]},
            {
                "tool_id": "update",
                "arguments": update["args"],
                "error": ERROR_MESSAGE,
            },
            {"tool_id": "unlock", "arguments": unlock["args"]},
            {"tool_id": "update", "arguments": update["args"]},
        ]
    scenario = {
        "scenario_id": "m11-alignment",
        "expected_procedure": {"expected_tool_trace": expected_trace},
        "execution_trace": execution_trace,
        "tools": {
            "lookup": {"arguments": [{"name": "customer_id", "required": True}]},
            "update": {"arguments": [{"name": "entity_id", "required": True}]},
            "unlock": {"arguments": [{"name": "entity_id", "required": True}]},
        },
    }
    if injected:
        scenario["error_injection"] = {
            "tool_id": "update",
            "occurrence": 1,
            "error_message": ERROR_MESSAGE,
            "recovery_tool_id": "unlock",
            "recovery_arguments": {
                "entity_id": {"source": "prior_result", "name": "entity_id"}
            },
        }
    return scenario


def _event_log(*, injected: bool) -> list[dict]:
    if not injected:
        return [
            {"seq": 1, "actor": "user", "event_type": "message", "content": "customer-1"},
            {
                "seq": 2,
                "actor": "operator",
                "event_type": "tool_call",
                "tool": "lookup",
                "arguments": {"customer_id": "customer-1"},
            },
            {
                "seq": 3,
                "actor": "tool",
                "event_type": "tool_result",
                "tool": "lookup",
                "result": {"entity_id": "entity-1"},
            },
            {
                "seq": 4,
                "actor": "operator",
                "event_type": "tool_call",
                "tool": "lookup",
                "arguments": {"customer_id": "customer-1"},
            },
            {
                "seq": 5,
                "actor": "tool",
                "event_type": "tool_result",
                "tool": "lookup",
                "result": {"entity_id": "entity-1"},
            },
            {
                "seq": 6,
                "actor": "operator",
                "event_type": "tool_call",
                "tool": "update",
                "arguments": {"entity_id": "entity-1"},
            },
            {
                "seq": 7,
                "actor": "tool",
                "event_type": "tool_result",
                "tool": "update",
                "result": {"entity_id": "entity-1"},
            },
            {
                "seq": 8,
                "actor": "operator",
                "event_type": "tool_call",
                "tool": "lookup",
                "arguments": {"customer_id": "customer-1"},
            },
            {
                "seq": 9,
                "actor": "tool",
                "event_type": "tool_result",
                "tool": "lookup",
                "result": {"entity_id": "entity-1"},
            },
        ]
    rows = [
        {"seq": 1, "actor": "user", "event_type": "message", "content": "customer-1"},
        {
            "seq": 2,
            "actor": "operator",
            "event_type": "tool_call",
            "tool": "lookup",
            "arguments": {"customer_id": "customer-1"},
        },
        {
            "seq": 3,
            "actor": "tool",
            "event_type": "tool_result",
            "tool": "lookup",
            "result": {"entity_id": "entity-1"},
        },
        {
            "seq": 4,
            "actor": "operator",
            "event_type": "tool_call",
            "tool": "update",
            "arguments": {"entity_id": "entity-1"},
        },
        {
            "seq": 5,
            "actor": "tool",
            "event_type": "tool_result",
            "tool": "update",
            "result": {"error": ERROR_MESSAGE},
        },
    ]
    rows.extend(
        [
            {
                "seq": 6,
                "actor": "operator",
                "event_type": "tool_call",
                "tool": "unlock",
                "arguments": {"entity_id": "entity-1"},
            },
            {
                "seq": 7,
                "actor": "tool",
                "event_type": "tool_result",
                "tool": "unlock",
                "result": {"entity_id": "entity-1"},
            },
        ]
    )
    rows.extend(
        [
            {
                "seq": 8,
                "actor": "operator",
                "event_type": "tool_call",
                "tool": "update",
                "arguments": {"entity_id": "entity-1"},
            },
            {
                "seq": 9,
                "actor": "tool",
                "event_type": "tool_result",
                "tool": "update",
                "result": {"entity_id": "entity-1"},
            },
        ]
    )
    return rows


class M11ErrorInjectionAlignmentTest(unittest.TestCase):
    def test_expected_failure_aligns_for_order_and_argument_provenance(self) -> None:
        scenario = _scenario(injected=True)
        event_log = _event_log(injected=True)

        score = _score_m11(scenario, event_log, _attempts(event_log))
        facets = score["details"]["sub_facets"]

        self.assertTrue(facets["dependency_order"]["passed"])
        self.assertEqual(
            [2, 4, 6, 8],
            facets["dependency_order"]["details"]["matched_call_seqs"],
        )
        self.assertTrue(facets["argument_provenance"]["passed"])
        self.assertEqual(
            [2, 4, 6, 8],
            [field["call_seq"] for field in facets["argument_provenance"]["details"]["fields"]],
        )

    def test_without_injection_keeps_legacy_occurrence_assignment(self) -> None:
        scenario = _scenario(injected=False)
        event_log = _event_log(injected=False)

        score = _score_m11(scenario, event_log, _attempts(event_log))
        facets = score["details"]["sub_facets"]
        serialized = json.dumps(
            score,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        self.assertEqual(
            LEGACY_NO_INJECTION_M11_SHA256,
            hashlib.sha256(serialized.encode()).hexdigest(),
        )
        self.assertEqual(
            [2, 6, 8],
            facets["dependency_order"]["details"]["matched_call_seqs"],
        )
        self.assertEqual(
            [2, 6, 4],
            [field["call_seq"] for field in facets["argument_provenance"]["details"]["fields"]],
        )

    def test_injected_execution_trace_must_align_with_gold_trace(self) -> None:
        scenario = _scenario(injected=True)
        scenario["execution_trace"][1]["tool_id"] = "other"
        event_log = _event_log(injected=True)

        with self.assertRaisesRegex(ValueError, "must align"):
            _score_m11(scenario, event_log, _attempts(event_log))


if __name__ == "__main__":
    unittest.main()
