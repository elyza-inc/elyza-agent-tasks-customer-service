"""Deterministic scoring for the M04 and M05 core metrics.

Each scorer accepts already-decoded mappings and an array of call-epoch event
objects.  Scenario contracts are deliberately explicit: event references and
record paths must be declared by the scenario, and this module never derives
them from natural-language instructions.  A missing optional contract returns
``N/A`` or ``N/M`` as documented by the individual scorer; a present but
malformed contract returns ``contract_invalid``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CORE_METRIC_SCORER_VERSION = "core_metric_scoring"
M04_CONTRACT_VERSION = "m04_customer_trigger_contract"
M05_CONTRACT_VERSION = "m05_obligation_contract"
M04_CONTRACT_FIELD = "m04_customer_trigger_contract"
M05_CONTRACT_FIELD = "m05_obligation_contract"


def score_m04_customer_trigger_control(
    *,
    contract: Any,
    call_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score a tool-ID customer-trigger contract."""

    if contract is None:
        return _result("M04", "N/A", None, "customer_trigger_edge_not_declared")
    try:
        edges = _validate_m04_contract(contract)
        return _score_m04_tool_edges(edges, call_events)
    except ValueError as exc:
        return _contract_invalid("M04", str(exc))


def score_m05_obligations(
    *,
    contract: Any,
    call_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score exact M05 predicates.

    ``contract`` is either ``None`` or an object with
    ``schema_version=m05_obligation_contract`` and a non-empty ``obligations``
    array. Obligations declare a deadline Tool whose invocation opens the
    measurement opportunity, required events, an optional deadline event, and
    an optional consent event plus protected action. Unknown fields,
    impossible option combinations, duplicate IDs, and malformed event logs
    return ``contract_invalid``. A missing contract is ``N/M``. The semantic
    disclosure and consent checks are merged later by ``package_scoring``.
    """

    if contract is None:
        return _result("M05", "N/M", None, "m05_machine_contract_not_declared")
    try:
        obligations = _validate_m05_contract(contract)
        event_positions = _event_positions(call_events)
    except ValueError as exc:
        return _contract_invalid("M05", str(exc))
    obligation_results: list[dict[str, Any]] = []
    for obligation in obligations:
        trigger_observed = any(
            event.get("event_type") == "tool_call"
            and event.get("tool") == obligation["deadline_tool_id"]
            for event in call_events
        )
        trigger_position = -1
        if not trigger_observed:
            obligation_results.append(
                {
                    "obligation_id": obligation["obligation_id"],
                    "status": "N/A",
                    "reason": "trigger_not_observed",
                    "deterministic_checks": {},
                }
            )
            continue
        deterministic_checks = _m05_deterministic_checks(
            obligation=obligation,
            event_positions=event_positions,
            trigger_position=trigger_position,
        )
        passed = all(deterministic_checks.values())
        obligation_results.append(
            {
                "obligation_id": obligation["obligation_id"],
                "status": "pass" if passed else "fail",
                "reason": "obligation_satisfied" if passed else "obligation_failed",
                "deterministic_checks": deterministic_checks,
            }
        )
    statuses = [item["status"] for item in obligation_results]
    triggered_statuses = [status for status in statuses if status != "N/A"]
    if not triggered_statuses:
        return _result(
            "M05",
            "N/A",
            {"passed": "N/A", "obligations": obligation_results},
            "no_m05_trigger_observed",
        )
    if "fail" in triggered_statuses:
        return _result(
            "M05",
            "fail",
            {"passed": False, "obligations": obligation_results},
            "one_or_more_m05_obligations_failed",
        )
    return _result(
        "M05",
        "pass",
        {"passed": True, "obligations": obligation_results},
        "all_triggered_m05_obligations_satisfied",
    )


def _score_m04_tool_edges(
    edges: list[dict[str, Any]], call_events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score tool-ID edges against call events with ``persona_fired`` markers."""
    if not isinstance(call_events, list) or any(not isinstance(event, dict) for event in call_events):
        raise ValueError("call_events must be an object array")
    results: list[dict[str, Any]] = []
    for edge in edges:
        trigger_positions = _persona_fired_positions(call_events, edge["customer_trigger_type"])
        action_positions = [
            index
            for index, event in enumerate(call_events)
            if event.get("event_type") == "tool_call"
            and event.get("tool") == edge["critical_action_tool_id"]
        ]
        action_positions = [index for index in action_positions if any(trigger < index for trigger in trigger_positions)]
        failures = []
        for action in action_positions:
            missing = [
                tool_id
                for tool_id in edge["required_precondition_tool_ids"]
                if not any(
                    index <= action
                    and event.get("event_type") == "tool_call"
                    and event.get("tool") == tool_id
                    for index, event in enumerate(call_events)
                )
            ]
            if missing:
                failures.append(
                    {"critical_action_position": action, "missing_or_late_precondition_tool_ids": missing}
                )
        applicable = bool(trigger_positions)
        results.append(
            {
                "edge_id": edge["edge_id"],
                "customer_trigger_type": edge["customer_trigger_type"],
                "critical_action_tool_id": edge["critical_action_tool_id"],
                "required_precondition_tool_ids": deepcopy(edge["required_precondition_tool_ids"]),
                "customer_trigger_observed": applicable,
                "critical_action_observed_after_trigger": bool(action_positions),
                "precondition_failures": failures,
                "applicable": applicable,
                "passed": None if not applicable else not failures,
            }
        )
    applicable_results = [item for item in results if item["applicable"]]
    if not applicable_results:
        return _result("M04", "N/A", {"passed": "N/A", "edges": results}, "declared_customer_trigger_not_observed")
    passed = all(item["passed"] is True for item in applicable_results)
    return _result(
        "M04",
        "pass" if passed else "fail",
        {"passed": passed, "edges": results},
        "all_declared_edges_controlled" if passed else "customer_trigger_edge_failed",
    )


def _persona_fired_positions(call_events: list[dict[str, Any]], trigger_type: str) -> list[int]:
    """Return exact trigger markers; each marker is a string or string array."""
    positions = []
    for index, event in enumerate(call_events):
        fired = event.get("persona_fired")
        if fired is None:
            continue
        if isinstance(fired, str):
            fired = [fired]
        if not isinstance(fired, list) or any(not isinstance(item, str) or not item for item in fired):
            raise ValueError("call persona_fired must be a non-empty string or string array")
        if trigger_type in fired:
            positions.append(index)
    return positions


def _validate_m04_contract(contract: Any) -> list[dict[str, Any]]:
    _require_exact_keys(contract, {"schema_version", "edges"}, "M04 contract")
    if contract["schema_version"] != M04_CONTRACT_VERSION:
        raise ValueError("unsupported M04 contract version")
    edges = contract["edges"]
    if not isinstance(edges, list) or not edges:
        raise ValueError("M04 edges must be a non-empty array")
    required = {
        "edge_id",
        "customer_trigger_type",
        "critical_action_tool_id",
        "required_precondition_tool_ids",
    }
    for index, edge in enumerate(edges):
        _require_exact_keys(edge, required, f"M04 edges[{index}]")
        _require_non_empty_strings(
            edge,
            required - {"required_precondition_tool_ids"},
            f"M04 edges[{index}]",
        )
        _require_unique_string_array(
            edge["required_precondition_tool_ids"],
            f"M04 edges[{index}].required_precondition_tool_ids",
        )
    _require_unique(edges, "edge_id", "M04 edge IDs")
    _require_unique(edges, "critical_action_tool_id", "M04 critical action refs")
    return edges


def _validate_m05_contract(contract: Any) -> list[dict[str, Any]]:
    _require_exact_keys(contract, {"schema_version", "obligations"}, "M05 contract")
    if contract["schema_version"] != M05_CONTRACT_VERSION:
        raise ValueError("unsupported M05 contract version")
    obligations = contract["obligations"]
    if not isinstance(obligations, list) or not obligations:
        raise ValueError("M05 obligations must be a non-empty array")
    required = {
        "obligation_id",
        "deadline_tool_id",
        "deadline_event_id",
        "required_event_ids",
        "consent_event_id",
        "protected_action_event_id",
    }
    for index, obligation in enumerate(obligations):
        label = f"M05 obligations[{index}]"
        _require_exact_keys(obligation, required, label)
        _require_non_empty_strings(
            obligation,
            {"obligation_id", "deadline_tool_id"},
            label,
        )
        for key in ("deadline_event_id", "consent_event_id", "protected_action_event_id"):
            _require_optional_string(obligation[key], f"{label}.{key}")
        _require_unique_string_array(obligation["required_event_ids"], f"{label}.required_event_ids")
        consent_declared = obligation["consent_event_id"] is not None
        protected_declared = obligation["protected_action_event_id"] is not None
        if consent_declared != protected_declared:
            raise ValueError(f"{label} consent event and protected action must be declared together")
    _require_unique(obligations, "obligation_id", "M05 obligation IDs")
    return obligations


def _m05_deterministic_checks(
    *,
    obligation: dict[str, Any],
    event_positions: dict[str, int],
    trigger_position: int,
) -> dict[str, bool]:
    deadline_ref = obligation["deadline_event_id"]
    deadline_position = event_positions.get(deadline_ref) if deadline_ref is not None else None
    required_positions = [event_positions.get(ref) for ref in obligation["required_event_ids"]]
    required_after_trigger = all(
        position is not None and position > trigger_position
        for position in required_positions
    )
    required_before_deadline = deadline_ref is None
    if deadline_position is not None:
        required_before_deadline = all(
            position is not None and position < deadline_position
            for position in required_positions
        )
    consent_before_action = True
    action_ref = obligation["protected_action_event_id"]
    if action_ref is not None:
        action_position = event_positions.get(action_ref)
        consent_position = event_positions.get(obligation["consent_event_id"])
        consent_before_action = (
            action_position is not None
            and consent_position is not None
            and consent_position > trigger_position
            and consent_position < action_position
        )
    return {
        "required_events_after_trigger": required_after_trigger,
        "deadline_met": required_before_deadline,
        "consent_before_protected_action": consent_before_action,
    }


def _event_positions(call_events: Any) -> dict[str, int]:
    if not isinstance(call_events, list) or any(not isinstance(event, dict) for event in call_events):
        raise ValueError("call_events must be an object array")
    positions: dict[str, int] = {}
    for index, event in enumerate(call_events):
        event_ref = event.get("event_ref")
        refs: list[str] = []
        if isinstance(event_ref, str) and event_ref:
            refs.append(event_ref)
        aliases = event.get("event_aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias for alias in aliases
        ):
            raise ValueError("call event_aliases must be a string array")
        refs.extend(aliases)
        if len(set(refs)) != len(refs):
            raise ValueError("call event identity contains duplicate refs")
        for ref in refs:
            if ref in positions:
                raise ValueError(f"duplicate call event_ref or alias: {ref}")
            positions[ref] = index
    return positions


def _result(metric_id: str, status: str, value: Any, reason: str) -> dict[str, Any]:
    return {
        "schema_version": CORE_METRIC_SCORER_VERSION,
        "metric_id": metric_id,
        "status": status,
        "value": value,
        "reason": reason,
    }


def _contract_invalid(metric_id: str, reason: str) -> dict[str, Any]:
    return _result(metric_id, "contract_invalid", None, reason)


def _require_exact_keys(value: Any, required: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise ValueError(
            f"{label} fields invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _require_non_empty_strings(value: dict[str, Any], keys: set[str], label: str) -> None:
    for key in keys:
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"{label}.{key} must be a non-empty string")


def _require_optional_string(value: Any, label: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{label} must be a non-empty string or null")


def _require_unique_string_array(value: Any, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must be a string array")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates")


def _require_unique(rows: list[dict[str, Any]], key: str, label: str) -> None:
    values = [row[key] for row in rows]
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
