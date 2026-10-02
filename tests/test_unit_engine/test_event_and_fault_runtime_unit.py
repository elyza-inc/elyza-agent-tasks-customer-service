"""Unit coverage for event projections and deterministic fault recovery."""

from __future__ import annotations

import pytest

from elyza_agent_tasks_customer_service.evaluation.engine import fault_injection


def test_legacy_fault_injector_requires_the_resolved_recovery_arguments() -> None:
    """The package-runtime injector also preserves its one-shot recovery contract."""

    injector = fault_injection.FaultInjector({"tool_id": "write", "occurrence": 1, "error_message": "locked", "recovery_tool_id": "unlock", "recovery_arguments": {"id": {"source": "prior_result", "name": "id"}}}, scenario_id="unit")
    assert injector.before_call("write", {"id": "1"}, result_rows=[], world_snapshot="before")["error"] == "locked"
    # A wrong recovery argument is an operator error: it runs as a normal call and recovery stays pending.
    assert injector.before_call("unlock", {"id": "wrong"}, result_rows=[], world_snapshot="before") is None
    assert injector.recovery_call_validated is False
    assert injector.recovery_argument_mismatch_count == 1
    injector.after_success("unlock", {"id": "wrong"})
    assert injector.recovered is False
    assert injector.before_call("unlock", {"id": "1"}, result_rows=[], world_snapshot="before") is None
    injector.after_success("unlock", {"id": "1"})
    injector.after_success("write", {"id": "1"})
    assert injector.completed is True


def test_fault_recovery_matches_phone_numbers_like_ordinary_tools() -> None:
    """Separators in phone numbers are ignored, as in ordinary tool matching; data defects still raise."""

    value = {"tool_id": "write", "occurrence": 1, "error_message": "locked", "recovery_tool_id": "reload", "recovery_arguments": {"phone_number": {"source": "prior_result", "name": "phone_number"}}}
    injector = fault_injection.FaultInjector(value, scenario_id="unit")
    injector.before_call("write", {}, result_rows=[], world_snapshot="before")
    rows = [{"phone_number": "090-0000-0050"}]
    assert injector.before_call("reload", {"phone_number": "09000000050"}, result_rows=rows, world_snapshot="before") is None
    assert injector.recovery_call_validated is True
    with pytest.raises(ValueError, match="multiple prior-result values"):
        injector._recovery_arguments_match({"phone_number": "1"}, {}, [{"phone_number": "1"}, {"phone_number": "2"}])
    # A value the run has not read yet is the operator's gap, not a data defect.
    assert injector._recovery_arguments_match({"phone_number": "1"}, {}, []) is False
