"""Validate evaluator-side declarations of deterministic transient Tool faults."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.contracts.type_contracts import _normalized_match_value


ERROR_INJECTION_FIELDS = {
    "tool_id",
    "occurrence",
    "error_message",
    "recovery_tool_id",
    "recovery_arguments",
}
RECOVERY_ARGUMENT_FIELDS = {"source", "name"}


class FaultInjector:
    """Apply one package ``error_injection`` without adding recovery tools.

    The accepted value is ``None`` or a mapping containing exactly
    ``tool_id``, ``occurrence``, ``error_message``, ``recovery_tool_id``, and
    ``recovery_arguments``. Invalid mappings raise ``ValueError`` with the
    scenario ID and field path.
    """

    def __init__(self, value: Any, *, scenario_id: str) -> None:
        self.scenario_id = scenario_id
        self.value = self._validate(value)
        self.success_counts: dict[str, int] = {}
        self.awaiting_recovery = False
        self.recovery_call_validated = False
        self.recovery_argument_mismatch_count = 0
        self.recovered = False
        self.completed = False
        self.failed_world_snapshot: str | None = None
        self.failed_arguments: dict[str, Any] | None = None

    def _invalid(self, field: str, message: str) -> None:
        raise ValueError(f"{self.scenario_id}: error_injection{field}: {message}")

    def _validate(self, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != ERROR_INJECTION_FIELDS:
            self._invalid("", "must contain exactly the required fields")
        for field in ("tool_id", "error_message", "recovery_tool_id"):
            if not isinstance(value[field], str) or not value[field].strip():
                self._invalid(f".{field}", "non-empty string is required")
        occurrence = value["occurrence"]
        if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 1:
            self._invalid(".occurrence", "positive integer is required")
        arguments = value["recovery_arguments"]
        if not isinstance(arguments, dict):
            self._invalid(".recovery_arguments", "mapping is required")
        for name, source in arguments.items():
            if not isinstance(name, str) or not name:
                self._invalid(".recovery_arguments", "argument names must be non-empty strings")
            if not isinstance(source, dict) or set(source) != RECOVERY_ARGUMENT_FIELDS:
                self._invalid(f".recovery_arguments.{name}", "source and name are required")
            if source["source"] != "prior_result":
                self._invalid(f".recovery_arguments.{name}.source", "must be prior_result")
            if not isinstance(source["name"], str) or not source["name"]:
                self._invalid(f".recovery_arguments.{name}.name", "non-empty string is required")
        return deepcopy(value)

    def before_call(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        result_rows: list[dict[str, Any]],
        world_snapshot: str,
    ) -> dict[str, Any] | None:
        if self.value is None or self.completed:
            return None
        if tool_id == self.value["recovery_tool_id"] and self.awaiting_recovery:
            # A mismatch is the operator's error: the call runs as a normal tool
            # call and recovery stays pending.
            if self._recovery_arguments_match(
                arguments,
                self.failed_arguments or {},
                result_rows,
            ):
                self.recovery_call_validated = True
            else:
                self.recovery_argument_mismatch_count += 1
            return None
        if tool_id != self.value["tool_id"]:
            return None
        occurrence = self.success_counts.get(tool_id, 0) + 1
        if occurrence != self.value["occurrence"]:
            return None
        if self.awaiting_recovery:
            if self.recovered:
                return None
            return self._failure_result()
        self.awaiting_recovery = True
        self.failed_world_snapshot = world_snapshot
        self.failed_arguments = deepcopy(arguments)
        return self._failure_result()

    def _failure_result(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": self.value["error_message"],
            "retryable": True,
            "world_unchanged": True,
        }

    def _recovery_arguments_match(
        self,
        arguments: dict[str, Any],
        failed_arguments: dict[str, Any],
        result_rows: list[dict[str, Any]],
    ) -> bool:
        """Return whether the operator's recovery arguments match the resolved values.

        Values are compared with the same normalization as ordinary tool
        matching. A value the run has not read yet is a mismatch; ambiguous
        expected values still raise ``ValueError``.
        """
        expected_names = set(self.value["recovery_arguments"])
        if set(arguments) != expected_names:
            return False
        matched = True
        for argument, source in self.value["recovery_arguments"].items():
            if argument in failed_arguments:
                expected = failed_arguments[argument]
            else:
                values = [row[source["name"]] for row in result_rows if source["name"] in row]
                distinct = {repr(value) for value in values}
                if len(distinct) > 1:
                    self._invalid(f".recovery_arguments.{argument}", "has multiple prior-result values")
                if not values:
                    # The operator has not read the value yet in this run; data defects
                    # of this kind surface in the perfect-run sweep instead.
                    return False
                expected = values[0]
            if _normalized_match_value(arguments[argument], argument, None) != _normalized_match_value(
                expected, argument, None
            ):
                matched = False
        return matched

    def after_success(self, tool_id: str, arguments: dict[str, Any]) -> None:
        del arguments
        if self.value is None or self.completed:
            return
        if tool_id == self.value["recovery_tool_id"] and self.awaiting_recovery:
            if not self.recovery_call_validated:
                return
            self.recovery_call_validated = False
            self.recovered = True
            return
        if tool_id != self.value["tool_id"]:
            return
        self.success_counts[tool_id] = self.success_counts.get(tool_id, 0) + 1
        if self.awaiting_recovery and self.recovered:
            self.completed = True




