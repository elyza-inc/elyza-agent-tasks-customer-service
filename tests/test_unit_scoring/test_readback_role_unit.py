"""Unit tests for explicit M20 readback-role routing into M16 tasks."""

from __future__ import annotations

from copy import deepcopy

import pytest

from elyza_agent_tasks_customer_service.evaluation.scoring import (
    conversation_log_scoring as scoring,
)


def _value(slot_id: str, role: str) -> dict:
    return {
        "slot_id": slot_id,
        "value": f"{slot_id}-value",
        "value_type": "string",
        "question_step_id": "question",
        "readback_role": role,
    }


def _tasks(values: list[dict]) -> list[dict]:
    package = {
        "scenario_id": "unit-001",
        "contracts": {
            "m05": None,
            "m20": {"important_values": deepcopy(values)},
        },
        "persona": {},
    }
    record = {
        "conversation": [],
        "tool_calls": [],
        "event_log": [],
        "operator_ticket_artifact": None,
    }
    evidence = scoring._evidence(record)
    tasks, _ = scoring._tasks(package, record, evidence)
    return tasks


@pytest.mark.parametrize("role", (None, "unknown"))
def test_important_values_reject_missing_or_invalid_readback_role(
    role: str | None,
) -> None:
    row = _value("value", "readback")
    if role is None:
        del row["readback_role"]
    else:
        row["readback_role"] = role

    with pytest.raises(ValueError):
        scoring._important_values({"m20": {"important_values": [row]}})


def test_consent_is_excluded_from_m4_and_routed_to_m5() -> None:
    readback = _value("readback-value", "readback")
    consent = _value("consent-value", "consent")
    statement = _value("statement-value", "statement")

    tasks = _tasks([readback, consent, statement])
    m4 = next(task for task in tasks if task["kind"] == "M16-M4")
    m5 = next(task for task in tasks if task["kind"] == "M16-M5")

    assert m4["question_id"] == "unit-001:M16:M4"
    assert m4["relevant_context"] == {"important_values": [readback]}
    assert m5["question_id"] == "unit-001:M16:M5"
    assert m5["metric_id"] == "M16"
    assert m5["relevant_context"] == {"consent_values": [consent]}
    assert m5["evidence_rules"] == ["operator_utterance_required"]


def test_m5_is_absent_without_consent_values() -> None:
    tasks = _tasks(
        [
            _value("readback-value", "readback"),
            _value("statement-value", "statement"),
        ]
    )

    assert all(task["kind"] != "M16-M5" for task in tasks)


def test_m4_is_absent_without_readback_values() -> None:
    tasks = _tasks(
        [
            _value("consent-value", "consent"),
            _value("statement-value", "statement"),
        ]
    )

    assert all(task["kind"] != "M16-M4" for task in tasks)
    assert any(task["kind"] == "M16-M5" for task in tasks)