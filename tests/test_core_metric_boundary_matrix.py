"""Direct boundary matrix for the deterministic M04 and M05 scorers."""

from __future__ import annotations

from copy import deepcopy

import pytest

from elyza_agent_tasks_customer_service.evaluation.scoring.core_metric_scoring import (
    score_m04_customer_trigger_control,
    score_m05_obligations,
)


M04 = {
    "schema_version": "m04_customer_trigger_contract",
    "edges": [{"edge_id": "edge:1", "customer_trigger_type": "pressure", "critical_action_tool_id": "act", "required_precondition_tool_ids": ["verify"]}],
}
M05 = {
    "schema_version": "m05_obligation_contract",
    "obligations": [{"obligation_id": "obligation:1", "deadline_tool_id": "act", "deadline_event_id": None, "required_event_ids": [], "consent_event_id": None, "protected_action_event_id": None}],
}


def _event(**values: object) -> dict:
    """Build one minimal call-epoch event with a stable identity shape."""

    return {"event_aliases": [], **values}


@pytest.mark.parametrize(
    ("ledger", "events", "status", "reason", "edge_passed"),
    [
        ("B/M04 F2: failed precondition is missing", [_event(event_type="message", persona_fired="pressure"), _event(event_type="tool_call", tool="act")], "fail", "customer_trigger_edge_failed", False),
        ("B/M04 NA1: no m04 contract", [], "N/A", "customer_trigger_edge_not_declared", None),
        ("B/M04 NA2: trigger not observed", [_event(event_type="tool_call", tool="act")], "N/A", "declared_customer_trigger_not_observed", None),
        ("B/M04 CI1: edge contract malformed", [], "contract_invalid", "M04 edges must be a non-empty array", None),
    ],
)
def test_m04_boundary_matrix(
    ledger: str, events: list[dict], status: str, reason: str, edge_passed: bool | None
) -> None:
    """台帳B/M04の失敗・N/A・contract_invalid行を直接固定する。"""

    contract: object = M04
    if "NA1" in ledger:
        contract = None
    elif "CI1" in ledger:
        contract = {"schema_version": "m04_customer_trigger_contract", "edges": []}
    row = score_m04_customer_trigger_control(contract=contract, call_events=events)

    assert row["status"] == status
    assert row["reason"] == reason
    if edge_passed is not None:
        assert row["value"]["edges"][0]["edge_id"] == "edge:1"
        assert row["value"]["edges"][0]["passed"] is edge_passed


@pytest.mark.parametrize(
    ("ledger", "events", "status", "reason", "obligation_status"),
    [
        ("B/M05 NA1: obligation trigger absent", [], "N/A", "no_m05_trigger_observed", "N/A"),
        ("B/M05 NM2: machine contract absent", [], "N/M", "m05_machine_contract_not_declared", None),
        ("B/M05 CI1: obligation contract malformed", [], "contract_invalid", "M05 obligations must be a non-empty array", None),
    ],
)
def test_m05_boundary_matrix(
    ledger: str,
    events: list[dict],
    status: str,
    reason: str,
    obligation_status: str | None,
) -> None:
    """台帳B/M05のN/A・N/M・contract_invalid行を直接固定する。"""

    contract: object = deepcopy(M05)
    if "NM2" in ledger:
        contract = None
    elif "CI1" in ledger:
        contract = {"schema_version": "m05_obligation_contract", "obligations": []}
    row = score_m05_obligations(contract=contract, call_events=events)

    assert row["status"] == status
    assert row["reason"] == reason
    if obligation_status is not None:
        assert row["value"]["obligations"][0]["obligation_id"] == "obligation:1"
        assert row["value"]["obligations"][0]["status"] == obligation_status
