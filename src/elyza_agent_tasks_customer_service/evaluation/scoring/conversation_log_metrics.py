"""Derive M11 (tool-use correctness) from a completed conversation log.

The scorer accepts a decoded object-root scenario and an object-row call-epoch
event log.  The scenario must provide ``expected_procedure.expected_tool_trace``
and may declare ``error_injection`` (with an aligned ``execution_trace``); an
undeclared scenario scores naturally observed error-to-retry series.

Every event row must have a unique integer ``seq``; ``post_session_ticket``
rows are excluded.  Malformed roots, missing required expectations, and
duplicate event sequence numbers raise ``ValueError`` instead of becoming
model failures.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.contracts.type_contracts import (
    _normalized_match_value,
    is_operational_enum_value,
    normalize_typed_value,
    provenance_text_contains_declared_spoken,
    provenance_text_contains_typed_value,
)


SCHEMA_VERSION = "conversation_log_metrics"
POST_CALL_EVENT_TYPES = ("post_session_ticket",)
PACKAGE_SOP_TOOLS = frozenset(
    ("search_sop_categories", "search_sops_in_category", "get_sop")
)
SUCCESS_BOOLEAN_FIELDS = (
    "ok",
    "success",
    "passed",
    "identity_verified",
    "verified",
)
MISSING = object()
M11_FACETS = (
    "required_success",
    "argument_value",
    "argument_provenance",
    "dependency_order",
    "error_recovery",
)
M11_ARGUMENT_PROVENANCE_SOURCES = frozenset(
    ("tool_result", "customer_utterance", "sop_step", "domain_policy", "tool_contract")
)
PROVENANCE_SOP_DETAIL_TOOL = "get_sop"
PROVENANCE_RESULT_METADATA_FIELDS = frozenset(("ok", "error", "message"))
TOOL_CONTRACT_TOKEN_PATTERN = r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])"


def _validate_cell_inputs(
    scenario: dict[str, Any],
    event_log: list[dict[str, Any]],
    run_id: str,
) -> None:
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be an object")
    if not isinstance(event_log, list) or any(
        not isinstance(item, dict) for item in event_log
    ):
        raise ValueError("event_log must be an object array")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(scenario.get("scenario_id"), str) or not scenario["scenario_id"]:
        raise ValueError("scenario.scenario_id must be a non-empty string")
    seen_sequences: set[int] = set()
    for event in _call_epoch_events(event_log):
        seq = event.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise ValueError("every event row must have an integer seq")
        if seq in seen_sequences:
            raise ValueError(f"duplicate event sequence: {seq}")
        seen_sequences.add(seq)


def _expected_tool_trace(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    procedure = scenario.get("expected_procedure")
    if not isinstance(procedure, dict):
        raise ValueError("scenario.expected_procedure must be an object")
    trace = procedure.get("expected_tool_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError(
            "scenario.expected_procedure.expected_tool_trace must be a non-empty object array"
        )
    result: list[dict[str, Any]] = []
    for index, item in enumerate(trace):
        if not isinstance(item, dict):
            raise ValueError(f"expected_tool_trace[{index}] must be an object")
        tool = item.get("tool")
        args = item.get("args")
        if not isinstance(tool, str) or not tool:
            raise ValueError(f"expected_tool_trace[{index}].tool is required")
        if not isinstance(args, dict):
            raise ValueError(f"expected_tool_trace[{index}].args must be an object")
        result.append({"trace_index": index, "tool": tool, "args": deepcopy(args)})
    if scenario.get("error_injection") is None:
        return result
    execution_trace = scenario.get("execution_trace")
    if not isinstance(execution_trace, list) or len(execution_trace) != len(result):
        raise ValueError("scenario.execution_trace must align with expected_tool_trace")
    for index, (expected, executed) in enumerate(zip(result, execution_trace)):
        if (
            not isinstance(executed, dict)
            or executed.get("tool_id") != expected["tool"]
            or executed.get("arguments") != expected["args"]
        ):
            raise ValueError(
                f"scenario.execution_trace[{index}] must align with expected_tool_trace[{index}]"
            )
    for expected, executed in zip(result, execution_trace):
        expected["expected_outcome"] = "error" if "error" in executed else "success"
    return result


def _call_epoch_events(event_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in event_log
        if event.get("event_type") not in POST_CALL_EVENT_TYPES
    ]


def _ordered_events(event_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(_call_epoch_events(event_log), key=lambda item: int(item["seq"]))


def _error_from_result(result: Any) -> str | None:
    if not isinstance(result, dict):
        return "tool_result_missing_or_non_object"
    for field in ("error", "error_type"):
        value = result.get(field)
        if value not in (None, "", False):
            return str(value)
    for field in SUCCESS_BOOLEAN_FIELDS:
        if result.get(field) is False:
            return f"{field}_false"
    return None


def _attempts(event_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending: dict[str, list[dict[str, Any]]] = {}
    attempts: list[dict[str, Any]] = []
    for event in _ordered_events(event_log):
        tool = event.get("tool")
        if not isinstance(tool, str) or not tool:
            continue
        event_type = event.get("event_type")
        if event_type == "tool_call":
            args = event.get("arguments")
            if not isinstance(args, dict):
                args = {}
            attempt = {
                "attempt_index": len(attempts),
                "tool": tool,
                "args": deepcopy(args),
                "call_seq": event["seq"],
                "call_event": event,
                "result_event": None,
                "result_seq": None,
                "result": None,
                "outcome": "missing_result",
                "error": "tool_result_missing",
            }
            attempts.append(attempt)
            pending.setdefault(tool, []).append(attempt)
            continue
        if event_type != "tool_result":
            continue
        tool_pending = pending.get(tool) or []
        if not tool_pending:
            continue
        attempt = tool_pending.pop(0)
        result = event.get("result")
        error = _error_from_result(result)
        attempt["result_seq"] = event["seq"]
        attempt["result_event"] = event
        attempt["result"] = deepcopy(result)
        outcome = "error"
        if error is None:
            outcome = "success"
        attempt["outcome"] = outcome
        attempt["error"] = error
    return attempts


def _domain_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in attempts if item["tool"] not in PACKAGE_SOP_TOOLS]


def _binary_result(passed: bool, reason: str, details: dict[str, Any]) -> dict[str, Any]:
    status = "fail"
    if passed:
        status = "pass"
    return {
        "status": status,
        "applicable": True,
        "measurable": True,
        "passed": passed,
        "value": passed,
        "reason": reason,
        "details": details,
    }


def _not_applicable(reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "not_applicable",
        "applicable": False,
        "measurable": False,
        "passed": None,
        "value": None,
        "reason": reason,
    }
    if details is not None:
        result["details"] = details
    return result


def _typed_values_match(
    observed: Any,
    expected: Any,
    contracts: list[tuple[str, str]],
) -> bool:
    if not contracts:
        return observed == expected
    for column, value_type in contracts:
        try:
            normalized_observed = normalize_typed_value(observed, value_type)
            normalized_expected = normalize_typed_value(expected, value_type)
        except ValueError:
            return False
        if _normalized_match_value(
            normalized_observed, column, value_type
        ) != _normalized_match_value(normalized_expected, column, value_type):
            return False
    return True


def _matching_value_path(
    candidate: Any,
    expected: Any,
    contracts: list[tuple[str, str]],
    *,
    contains_text: bool,
    path: str,
) -> str | None:
    if isinstance(candidate, dict):
        for key, value in candidate.items():
            matched = _matching_value_path(
                value, expected, contracts, contains_text=contains_text, path=f"{path}.{key}"
            )
            if matched is not None:
                return matched
        return None
    if isinstance(candidate, list):
        for index, value in enumerate(candidate):
            matched = _matching_value_path(
                value, expected, contracts, contains_text=contains_text, path=f"{path}[{index}]"
            )
            if matched is not None:
                return matched
        return None
    if contains_text:
        return path if provenance_text_contains_typed_value(candidate, expected, contracts) else None
    return path if _typed_values_match(candidate, expected, contracts) else None


def _declared_spoken_forms(
    scenario: dict[str, Any],
    value: Any,
    contracts: list[tuple[str, str]],
) -> list[str]:
    inputs = scenario.get("inputs")
    if not isinstance(inputs, dict):
        return []
    return [
        row["spoken"]
        for group in ("identity", "declared")
        for row in inputs.get(group, [])
        if isinstance(row, dict)
        and isinstance(row.get("spoken"), str)
        and _typed_values_match(row.get("value"), value, contracts)
    ]


def _recalculate_argument_provenance(
    scenario: dict[str, Any],
    event_log: list[dict[str, Any]],
    *,
    tool_id: str,
    field: str,
    value: Any,
    call_seq: int | None,
    contracts: list[tuple[str, str]],
) -> dict[str, Any]:
    """Reattribute one argument from prior events, domain policy, then its schema.

    ``call_seq`` must identify the current call; only lower integer event
    sequences are eligible.  Returns the standard provenance fields with an
    ``unknown`` source when no deterministic exact/text match is available.
    """

    unknown = {
        "derived_source_kind": "unknown",
        "source_seq": None,
        "source_tool": None,
        "source_ref": None,
        "source_path": None,
    }
    if not isinstance(call_seq, int) or isinstance(call_seq, bool):
        return unknown
    prior_events = [event for event in _ordered_events(event_log) if int(event["seq"]) < call_seq]
    spoken_forms = _declared_spoken_forms(scenario, value, contracts)
    for event in reversed(prior_events):
        if event.get("actor") != "tool" or event.get("event_type") != "tool_result":
            continue
        result = event.get("result")
        if _error_from_result(result) is not None or not isinstance(result, dict):
            continue
        payload = {key: item for key, item in result.items() if key not in PROVENANCE_RESULT_METADATA_FIELDS}
        path = _matching_value_path(payload, value, contracts, contains_text=False, path="result")
        if path is not None:
            return {
                "derived_source_kind": "tool_result",
                "source_seq": event["seq"],
                "source_tool": event.get("tool"),
                "source_ref": event.get("event_ref"),
                "source_path": path,
            }
    for event in reversed(prior_events):
        if event.get("actor") != "user" or event.get("event_type") != "message":
            continue
        content = event.get("content")
        path = _matching_value_path(
            content, value, contracts, contains_text=True, path="content"
        )
        if path is None and any(
            provenance_text_contains_declared_spoken(content, spoken)
            for spoken in spoken_forms
        ):
            path = "content"
        if path is not None:
            return {
                "derived_source_kind": "customer_utterance",
                "source_seq": event["seq"],
                "source_tool": None,
                "source_ref": event.get("event_ref"),
                "source_path": path,
            }
    for event in reversed(prior_events):
        if event.get("tool") != PROVENANCE_SOP_DETAIL_TOOL or event.get("event_type") != "tool_result":
            continue
        sop = event.get("result", {}).get("sop") if isinstance(event.get("result"), dict) else None
        steps = sop.get("steps") if isinstance(sop, dict) else None
        if not isinstance(steps, list):
            continue
        for index, step in enumerate(steps):
            description = step.get("description") if isinstance(step, dict) else None
            path = _matching_value_path(
                description, value, contracts, contains_text=True,
                path=f"result.sop.steps[{index}].description",
            )
            if path is not None:
                return {
                    "derived_source_kind": "sop_step",
                    "source_seq": event["seq"],
                    "source_tool": PROVENANCE_SOP_DETAIL_TOOL,
                    "source_ref": event.get("event_ref"),
                    "source_path": path,
                }
    path = _matching_value_path(scenario.get("domain"), value, contracts, contains_text=True, path="domain")
    if path is not None:
        return {
            "derived_source_kind": "domain_policy",
            "source_seq": 0,
            "source_tool": None,
            "source_ref": "operator_system_prompt",
            "source_path": path,
        }
    tool = scenario.get("tools", {}).get(tool_id) if isinstance(scenario.get("tools"), dict) else None
    specs = tool.get("arguments") if isinstance(tool, dict) else None
    spec = next((item for item in specs or [] if isinstance(item, dict) and item.get("name") == field), None)
    enum = spec.get("enum") if isinstance(spec, dict) else None
    if isinstance(enum, list) and value in enum:
        return {
            "derived_source_kind": "tool_contract",
            "source_seq": 0,
            "source_tool": None,
            "source_ref": "tool_definition",
            "source_path": f"tools.{tool_id}.arguments.{field}.enum",
        }
    if isinstance(spec, dict) and is_operational_enum_value(field, value):
        return {
            "derived_source_kind": "tool_contract",
            "source_seq": 0,
            "source_tool": None,
            "source_ref": "tool_definition",
            "source_path": f"tools.{tool_id}.arguments.{field}.operational_enum",
        }
    description = spec.get("description") if isinstance(spec, dict) else None
    vocabulary_pattern = None
    if isinstance(value, str):
        vocabulary_pattern = TOOL_CONTRACT_TOKEN_PATTERN.format(re.escape(value))
    if (
        vocabulary_pattern is not None
        and isinstance(description, str)
        and re.search(vocabulary_pattern, description)
    ):
        return {
            "derived_source_kind": "tool_contract",
            "source_seq": 0,
            "source_tool": None,
            "source_ref": "tool_definition",
            "source_path": f"tools.{tool_id}.arguments.{field}.description",
        }
    return unknown


def _expected_domain_trace(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in _expected_tool_trace(scenario) if item["tool"] not in PACKAGE_SOP_TOOLS
    ]


def _args_match(
    observed: Any,
    expected: dict[str, Any],
    contracts: dict[str, list[tuple[str, str]]] | None = None,
) -> bool:
    if not isinstance(observed, dict):
        return False
    return all(
        field in observed
        and _typed_values_match(
            observed[field],
            value,
            (contracts or {}).get(field, []),
        )
        for field, value in expected.items()
    )


def _argument_contracts(
    scenario: dict[str, Any],
) -> dict[str, dict[str, list[tuple[str, str]]]]:
    tools = scenario.get("tools")
    world_schema = scenario.get("world_schema")
    if not isinstance(tools, dict) or not isinstance(world_schema, dict):
        return {}
    entities = world_schema.get("entities")
    if not isinstance(entities, dict):
        return {}
    result: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for tool_id, tool in tools.items():
        if not isinstance(tool_id, str) or not isinstance(tool, dict):
            continue
        entity = entities.get(tool.get("entity"))
        columns = entity.get("columns") if isinstance(entity, dict) else None
        if not isinstance(columns, dict):
            columns = {}
        destinations: dict[str, list[str]] = {}
        for rule in tool.get("filters", []):
            if not isinstance(rule, dict):
                continue
            argument = rule.get("argument")
            column = rule.get("column")
            if isinstance(argument, str) and isinstance(column, str):
                destinations.setdefault(argument, []).append(column)
        mutation = tool.get("mutation")
        set_values = mutation.get("set") if isinstance(mutation, dict) else None
        if isinstance(set_values, dict):
            for column, source in set_values.items():
                if isinstance(column, str) and isinstance(source, dict):
                    argument = source.get("from_argument")
                    if isinstance(argument, str):
                        destinations.setdefault(argument, []).append(column)
        fields: dict[str, list[tuple[str, str]]] = {}
        for spec in tool.get("arguments", []):
            if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
                continue
            name = spec["name"]
            target_columns = destinations.get(name, [name])
            contracts = [
                (column, columns.get(column, spec.get("type")))
                for column in target_columns
                if isinstance(columns.get(column, spec.get("type")), str)
            ]
            if contracts:
                fields[name] = list(dict.fromkeys(contracts))
        result[tool_id] = fields
    return result


def _attempt_receipt(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_index": attempt["attempt_index"],
        "tool": attempt["tool"],
        "call_seq": attempt["call_seq"],
        "result_seq": attempt["result_seq"],
        "outcome": attempt["outcome"],
        "error": attempt["error"],
        "args": deepcopy(attempt["args"]),
    }


def _call_findings(
    scenario: dict[str, Any],
    event_log: list[dict[str, Any]],
    expected_trace: list[dict[str, Any]],
    domain_attempts: list[dict[str, Any]],
    argument_contracts: dict[str, dict[str, list[tuple[str, str]]]],
    aligned_attempts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    successful_occurrences: dict[str, int] = {}
    for expected_index, expected in enumerate(expected_trace):
        observed = [
            item for item in domain_attempts if item["tool"] == expected["tool"]
        ]
        successful = [item for item in observed if item["outcome"] == "success"]
        occurrence = successful_occurrences.get(expected["tool"], 0)
        source_attempt = successful[occurrence] if occurrence < len(successful) else None
        successful_occurrences[expected["tool"]] = occurrence + 1
        if aligned_attempts is not None:
            if expected_index < len(aligned_attempts):
                source_attempt = aligned_attempts[expected_index]
            else:
                source_attempt = None
        contracts = argument_contracts.get(expected["tool"], {})
        exact_successful = [
            item
            for item in successful
            if _args_match(item["args"], expected["args"], contracts)
        ]
        selected_attempt = None
        if exact_successful:
            selected_attempt = exact_successful[0]
        if source_attempt is None and selected_attempt is not None:
            source_attempt = selected_attempt
        fields: list[dict[str, Any]] = []
        for field, expected_value in expected["args"].items():
            observed_value = MISSING
            if source_attempt is not None:
                observed_value = source_attempt["args"].get(field, MISSING)
            tools = scenario.get("tools")
            tool = tools.get(expected["tool"]) if isinstance(tools, dict) else None
            specs = tool.get("arguments") if isinstance(tool, dict) else []
            spec = next(
                (item for item in specs if isinstance(item, dict) and item.get("name") == field),
                None,
            )
            required = not isinstance(spec, dict) or spec.get("required") is True
            if observed_value is MISSING:
                source = {
                    "derived_source_kind": "unknown",
                    "source_seq": None,
                    "source_tool": None,
                    "source_ref": None,
                    "source_path": None,
                }
            elif observed_value is None or isinstance(observed_value, bool):
                source = {
                    "derived_source_kind": "not_applicable",
                    "source_seq": None,
                    "source_tool": None,
                    "source_ref": None,
                    "source_path": None,
                }
            else:
                source = _recalculate_argument_provenance(
                    scenario,
                    event_log,
                    tool_id=expected["tool"],
                    field=field,
                    value=observed_value,
                    call_seq=None if source_attempt is None else source_attempt["call_seq"],
                    contracts=contracts.get(field, []),
                )
            fields.append(
                {
                    "field": field,
                    "expected": deepcopy(expected_value),
                    "observed": None if observed_value is MISSING else deepcopy(observed_value),
                    "matched_on_success": source_attempt is not None
                    and observed_value is not MISSING,
                    **source,
                    "required": required,
                    "provenance_recomputed": observed_value is not MISSING
                    and observed_value is not None
                    and not isinstance(observed_value, bool),
                    "call_seq": None if source_attempt is None else source_attempt["call_seq"],
                }
            )
        selected_exact_attempt_index = None
        selected_exact_call_seq = None
        if selected_attempt is not None:
            selected_exact_attempt_index = selected_attempt["attempt_index"]
            selected_exact_call_seq = selected_attempt["call_seq"]
        findings.append(
            {
                "expected_call_index": expected_index,
                "trace_index": expected["trace_index"],
                "tool": expected["tool"],
                "expected_args": deepcopy(expected["args"]),
                "success_observed": bool(successful),
                "exact_success_observed": bool(exact_successful),
                "selected_exact_attempt_index": selected_exact_attempt_index,
                "selected_exact_call_seq": selected_exact_call_seq,
                "observed_attempts": [_attempt_receipt(item) for item in observed],
                "fields": fields,
            }
        )
    return findings


def _score_required_success(findings: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [item for item in findings if not item["success_observed"]]
    passed = not missing
    reason = "one_or_more_required_domain_calls_missing_success"
    if passed:
        reason = "all_required_domain_calls_succeeded"
    return _binary_result(
        passed,
        reason,
        {
            "definition": "required Tool call has a successful matching tool_result",
            "required_calls": deepcopy(findings),
            "missing_success_tools": [item["tool"] for item in missing],
        },
    )


def _score_argument_value(findings: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = [item for item in findings if not item["exact_success_observed"]]
    passed = not mismatches
    reason = "one_or_more_required_argument_values_not_matched_on_success"
    if passed:
        reason = "all_required_argument_values_matched_on_success"
    return _binary_result(
        passed,
        reason,
        {
            "call_findings": deepcopy(findings),
            "mismatch_tools": [item["tool"] for item in mismatches],
        },
    )


def _score_argument_provenance(findings: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [deepcopy(field) for item in findings for field in item["fields"]]
    required_fields = [
        item
        for item in fields
        if item["required"] and item["derived_source_kind"] != "not_applicable"
    ]
    passed = all(item["success_observed"] for item in findings) and all(
        item["matched_on_success"]
        and item["derived_source_kind"] in M11_ARGUMENT_PROVENANCE_SOURCES
        and isinstance(item["source_seq"], int)
        and not isinstance(item["source_seq"], bool)
        and isinstance(item["call_seq"], int)
        and item["source_seq"] < item["call_seq"]
        for item in required_fields
    )
    reason = "one_or_more_required_arguments_have_unknown_or_invalid_provenance"
    if passed:
        reason = "all_required_arguments_have_recomputed_prior_provenance"
    return _binary_result(
        passed,
        reason,
        {
            "fields": fields,
            "derivation_policy": (
                "recomputed prior tool_result, customer_utterance, sop_step, domain_policy, "
                "or matching tool-contract vocabulary; unknown fails required arguments"
            ),
        },
    )


def _match_successful_trace_in_order(
    expected_trace: list[dict[str, Any]],
    domain_attempts: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    cursor = 0
    matched: list[dict[str, Any]] = []
    for expected in expected_trace:
        selected = None
        expected_outcome = expected.get("expected_outcome", "success")
        for index in range(cursor, len(domain_attempts)):
            attempt = domain_attempts[index]
            if attempt["tool"] != expected["tool"]:
                continue
            if attempt["outcome"] != expected_outcome:
                continue
            selected = attempt
            cursor = index + 1
            break
        if selected is None:
            return None
        matched.append(selected)
    return matched


def _match_successful_trace(
    expected_trace: list[dict[str, Any]],
    domain_attempts: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Match expected successful calls, retaining order when it is observed.

    Each expected row needs one distinct same-tool, same-outcome attempt.
    When the full trace is not ordered, callers can score only its actual
    data-dependency edges rather than treating independent calls as missing.
    """

    ordered = _match_successful_trace_in_order(expected_trace, domain_attempts)
    if ordered is not None:
        return ordered
    remaining = list(domain_attempts)
    matched: list[dict[str, Any]] = []
    for expected in expected_trace:
        outcome = expected.get("expected_outcome", "success")
        selected = next(
            (
                attempt
                for attempt in remaining
                if attempt["tool"] == expected["tool"] and attempt["outcome"] == outcome
            ),
            None,
        )
        if selected is None:
            return None
        remaining.remove(selected)
        matched.append(selected)
    return matched


def _scalar_values(value: Any) -> list[str | int | float]:
    if isinstance(value, dict):
        return [scalar for child in value.values() for scalar in _scalar_values(child)]
    if isinstance(value, list):
        return [scalar for child in value for scalar in _scalar_values(child)]
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return [value]
    return []


def _step_sources(steps: list[list[dict[str, Any]]]) -> list[tuple[str, Any]]:
    sources: list[tuple[str, Any]] = []
    for step in steps:
        event = step[-1]
        if event.get("event_type") == "message":
            sources.append(("utterance", event.get("content", "")))
        elif event.get("event_type") == "tool_result":
            sources.extend(("result", value) for value in _scalar_values(event.get("result")))
    return sources


def _matches_step_source(value: str | int | float, sources: list[tuple[str, Any]]) -> bool:
    return any(
        value == source if kind == "result" else str(value) in str(source)
        for kind, source in sources
    )


def _depends_on_preceding_step(steps: list[list[dict[str, Any]]], index: int) -> bool:
    """Whether the next step needs a value first introduced by this step."""

    later_arguments = _scalar_values(steps[index + 1][0].get("arguments", {}))
    former_sources = _step_sources([steps[index]])
    prior_sources = _step_sources(steps[:index])
    return any(
        _matches_step_source(argument, former_sources)
        and not _matches_step_source(argument, prior_sources)
        for argument in later_arguments
    )


def _attempt_step(attempt: dict[str, Any]) -> list[dict[str, Any]]:
    result = attempt.get("result_event")
    return [attempt["call_event"], result] if isinstance(result, dict) else [attempt["call_event"]]


def _score_dependency_order(
    expected_trace: list[dict[str, Any]],
    matched: list[dict[str, Any]] | None,
    event_log: list[dict[str, Any]],
) -> dict[str, Any]:
    passed = matched is not None
    edges: list[dict[str, Any]] = []
    if matched is not None:
        steps = [_attempt_step(attempt) for attempt in matched]
        for index, (before, after) in enumerate(zip(matched, matched[1:])):
            later_sequences = {
                sequence
                for attempt in matched[index + 1 :]
                for sequence in (attempt["call_seq"], attempt["result_seq"])
                if isinstance(sequence, int)
            }
            context = [
                [event]
                for event in _ordered_events(event_log)
                if event["seq"] < before["call_seq"]
                and event["seq"] not in later_sequences
            ]
            dependency_steps = context + steps[: index + 2]
            if not _depends_on_preceding_step(dependency_steps, len(context) + index):
                continue
            edges.append(
                {
                    "before_tool": before["tool"],
                    "after_tool": after["tool"],
                    "before_call_seq": before["call_seq"],
                    "after_call_seq": after["call_seq"],
                    "passed": before["call_seq"] < after["call_seq"],
                }
            )
        passed = all(edge["passed"] for edge in edges)
    reason = "one_or_more_successful_dependency_edges_not_observed_in_order"
    if passed:
        reason = "successful_dependency_edges_observed_in_order"
    matched_call_seqs: list[int] = []
    if matched is not None:
        matched_call_seqs = [int(item["call_seq"]) for item in matched]
    return _binary_result(
        passed,
        reason,
        {
            "expected_tools": [item["tool"] for item in expected_trace],
            "matched_call_seqs": matched_call_seqs,
            "edges": edges,
            "dependency_definition": "later arguments first sourced from the preceding tool result or utterance",
        },
    )


def _score_recovery(
    expected_trace: list[dict[str, Any]],
    domain_attempts: list[dict[str, Any]],
    error_injection: Any = None,
) -> dict[str, Any]:
    if error_injection is not None:
        if not isinstance(error_injection, dict):
            raise ValueError("scenario.error_injection must be an object or null")
        required = {
            "tool_id",
            "occurrence",
            "error_message",
            "recovery_tool_id",
            "recovery_arguments",
        }
        if set(error_injection) != required:
            raise ValueError("scenario.error_injection fields are invalid")
        target_tool = error_injection["tool_id"]
        recovery_tool = error_injection["recovery_tool_id"]
        target_attempts = [
            attempt for attempt in domain_attempts if attempt["tool"] == target_tool
        ]
        first = target_attempts[0] if target_attempts else None
        recovery = None
        retry = None
        if first is not None and first["outcome"] == "error":
            recovery = next(
                (
                    attempt
                    for attempt in domain_attempts
                    if attempt["call_seq"] > first["call_seq"]
                    and attempt["tool"] == recovery_tool
                    and attempt["outcome"] == "success"
                ),
                None,
            )
        if recovery is not None:
            retry = next(
                (
                    attempt
                    for attempt in target_attempts[1:]
                    if attempt["call_seq"] > recovery["call_seq"]
                ),
                None,
            )
        error_observed = (
            first is not None
            and first["outcome"] == "error"
            and first["error"] == error_injection["error_message"]
        )
        recovered = error_observed and recovery is not None and retry is not None and retry["outcome"] == "success"
        return _binary_result(
            recovered,
            "declared_failure_recovery_retry_succeeded"
            if recovered
            else "declared_failure_recovery_retry_incomplete",
            {
                "target_tool": target_tool,
                "recovery_tool": recovery_tool,
                "error_attempt": _attempt_receipt(first) if first is not None else None,
                "recovery_attempt": _attempt_receipt(recovery) if recovery is not None else None,
                "retry_attempt": _attempt_receipt(retry) if retry is not None else None,
                "error_message_observed": error_observed,
                "recovered": recovered,
            },
        )
    allowed = {item["tool"] for item in expected_trace}
    series: list[dict[str, Any]] = []
    for index, attempt in enumerate(domain_attempts):
        if attempt["outcome"] != "error":
            continue
        if attempt["tool"] in allowed:
            retry = next(
                (
                    item
                    for item in domain_attempts[index + 1 :]
                    if item["tool"] == attempt["tool"]
                ),
                None,
            )
        else:
            retry = next(
                (
                    item
                    for item in domain_attempts[index + 1 :]
                    if item["tool"] in allowed
                ),
                None,
            )
        if retry is None:
            continue
        expected_matches = [
            item
            for item in expected_trace
            if item["tool"] == retry["tool"]
            and _args_match(retry["args"], item["args"])
        ]
        recovered = retry["outcome"] == "success" and bool(expected_matches)
        series.append(
            {
                "error_attempt": _attempt_receipt(attempt),
                "retry_attempt": _attempt_receipt(retry),
                "same_tool": retry["tool"] == attempt["tool"],
                "expected_retry_tool": retry["tool"] in allowed,
                "expected_args_on_success": bool(expected_matches),
                "recovered": recovered,
            }
        )
    if not series:
        return _not_applicable("no_naturally_observed_error_to_retry_series")
    passed = all(item["recovered"] for item in series)
    reason = "one_or_more_naturally_observed_retry_series_not_recovered_correctly"
    if passed:
        reason = "all_naturally_observed_retry_series_recovered_correctly"
    return _binary_result(passed, reason, {"series": series})


def _score_m11(
    scenario: dict[str, Any],
    event_log: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = _expected_domain_trace(scenario)
    if not expected:
        raise ValueError("expected_tool_trace must contain at least one domain tool")
    domain = _domain_attempts(attempts)
    matched = _match_successful_trace(expected, domain)
    aligned_attempts = None
    if scenario.get("error_injection") is not None:
        aligned_attempts = matched or []
    findings = _call_findings(
        scenario,
        event_log,
        expected,
        domain,
        _argument_contracts(scenario),
        aligned_attempts,
    )
    facets = {
        "required_success": _score_required_success(findings),
        "argument_value": _score_argument_value(findings),
        "argument_provenance": _score_argument_provenance(findings),
        "dependency_order": _score_dependency_order(expected, matched, event_log),
        "error_recovery": _score_recovery(
            expected,
            domain,
            scenario.get("error_injection"),
        ),
    }
    fired = [facets[name] for name in M11_FACETS if facets[name]["status"] != "not_applicable"]
    passed_count = sum(item["passed"] is True for item in fired)
    pass_rate = passed_count / len(fired) if fired else None
    result = {
        "status": "measured",
        "applicable": True,
        "measurable": True,
        "passed": None,
        "value": {"applicable_pass_rate": pass_rate},
        "reason": "five_independent_tool_execution_results",
        "details": {
            "representative_value": "applicable_pass_rate",
            "sub_facets": facets,
            "expected_domain_trace": expected,
            "domain_attempts": [_attempt_receipt(item) for item in domain],
        },
    }
    result["metric_id"] = "M11"
    return result


def score_conversation_log_metrics(
    scenario: dict[str, Any],
    event_log: list[dict[str, Any]],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Return post-hoc M11 from a decoded scenario object and event-row array.

    Invalid artifact contracts raise ``ValueError``.
    """

    _validate_cell_inputs(scenario, event_log, run_id)
    attempts = _attempts(event_log)
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": scenario["scenario_id"],
        "run_id": run_id,
        "M11": _score_m11(scenario, event_log, attempts),
    }
