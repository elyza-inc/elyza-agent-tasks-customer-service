"""Additional synthetic coverage for deterministic engine boundaries."""

from __future__ import annotations


import pytest

from elyza_agent_tasks_customer_service.evaluation.engine import fault_injection as fault


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({}, "required fields"),
        ({"tool_id": "x", "occurrence": True, "error_message": "x", "recovery_tool_id": "r", "recovery_arguments": {}}, "positive integer"),
        ({"tool_id": "x", "occurrence": 1, "error_message": "x", "recovery_tool_id": "r", "recovery_arguments": {"x": {"source": "bad", "name": "x"}}}, "prior_result"),
    ],
)
def test_legacy_fault_injector_validates_declared_shapes(value: object, message: str) -> None:
    """Reject malformed deterministic recovery declarations before execution."""

    with pytest.raises(ValueError, match=message):
        fault.FaultInjector(value, scenario_id="unit")


def test_legacy_fault_injector_resolves_result_rows_and_lifecycle() -> None:
    """Use a prior result when the failed call did not carry the recovery key."""

    injector = fault.FaultInjector(
        {"tool_id": "lookup", "occurrence": 2, "error_message": "locked", "recovery_tool_id": "unlock", "recovery_arguments": {"id": {"source": "prior_result", "name": "id"}}},
        scenario_id="unit",
    )
    assert injector.before_call("other", {}, result_rows=[], world_snapshot="x") is None
    injector.after_success("lookup", {})
    failed = injector.before_call("lookup", {}, result_rows=[], world_snapshot="before")
    assert failed and failed["world_unchanged"]
    # Before the run has read the value, the recovery call is an unvalidated normal call.
    assert injector.before_call("unlock", {"id": "x"}, result_rows=[], world_snapshot="before") is None
    assert injector.recovery_call_validated is False
    assert injector.before_call("unlock", {"id": "x"}, result_rows=[{"id": "x"}], world_snapshot="before") is None
    injector.after_success("unlock", {})
    injector.after_success("lookup", {})
    assert injector.completed


