"""Derive completion and handling type from observed run evidence.

The reader accepts an object-root record plus a call-epoch event array.  A
stored ``run_outcome`` object is preferred; records without one (packages
without ``contracts.completion``) are derived from terminal state, hard-fail state, successful tool-result metadata,
and final-world records.  Operator-authored ticket fields are never inputs.
"""

from __future__ import annotations

from typing import Any


RUN_OUTCOME_VERSION = "run_outcome"
INCOMPLETE = "incomplete"
HANDLING_TYPES = (
    "change_procedure",
    "guidance_inquiry_answer",
    "refusal",
    "escalation",
    "callback_commitment",
    INCOMPLETE,
)
COMPLETE_HANDLING_TYPES = frozenset(HANDLING_TYPES[:-1])
ALL_HANDLING_TYPES = frozenset(HANDLING_TYPES)
OPERATION_ANNOTATION_KEY = "ticket_operation"
FINAL_HANDLING_FIELDS = (
    "handling_type",
    "record_role",
    "outcome_type",
)
HANDLING_TYPE_ALIASES = {
    "change": "change_procedure",
    "guidance": "guidance_inquiry_answer",
    "refusal": "refusal",
    "escalation": "escalation",
    "callback": "callback_commitment",
}


def derive_run_outcome(
    *,
    record: dict[str, Any],
    call_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a validated ``run_outcome`` object from run evidence."""

    stored = record.get(RUN_OUTCOME_VERSION)
    if stored is not None:
        return validate_run_outcome(stored)
    completed = _legacy_completed(record, call_events)
    if not completed:
        return _outcome(INCOMPLETE, False, "run_not_completed")
    handling_type = _handling_from_events(call_events)
    if handling_type is None:
        handling_type = _handling_from_world(record.get("final_world"))
    if handling_type is None and _has_successful_world_change(call_events):
        handling_type = "change_procedure"
    if handling_type is None:
        return _outcome(INCOMPLETE, False, "completion_record_missing")
    return _outcome(handling_type, True, "observed_run_evidence")


def build_run_outcome_from_facts(
    *,
    completed: bool,
    handling_type: str | None,
    reason: str,
) -> dict[str, Any]:
    """Build an outcome from explicit runtime facts without evidence inference."""

    if not isinstance(completed, bool):
        raise ValueError("run outcome completed must be boolean")
    if not isinstance(reason, str) or not reason:
        raise ValueError("run outcome reason must be a non-empty string")
    if not completed or handling_type is None:
        return _outcome(INCOMPLETE, False, reason)
    if handling_type not in COMPLETE_HANDLING_TYPES:
        raise ValueError("run outcome handling_type is invalid")
    return _outcome(handling_type, True, reason)


def validate_run_outcome(value: Any) -> dict[str, Any]:
    """Validate one exact object-root ``run_outcome`` value."""

    if not isinstance(value, dict):
        raise ValueError("run outcome must be an object")
    required = {"schema_version", "handling_type", "completed", "reason"}
    if set(value) != required:
        raise ValueError("run outcome fields are invalid")
    if value.get("schema_version") != RUN_OUTCOME_VERSION:
        raise ValueError("run outcome version mismatch")
    handling_type = value.get("handling_type")
    if handling_type not in ALL_HANDLING_TYPES:
        raise ValueError("run outcome handling_type is invalid")
    if not isinstance(value.get("completed"), bool):
        raise ValueError("run outcome completed must be boolean")
    if not isinstance(value.get("reason"), str) or not value["reason"]:
        raise ValueError("run outcome reason must be a non-empty string")
    if value["completed"] != (handling_type != INCOMPLETE):
        raise ValueError("run outcome completion/handling_type mismatch")
    return value


def _legacy_completed(
    record: dict[str, Any],
    call_events: list[dict[str, Any]],
) -> bool:
    if record.get("status") in {"success", "ok"}:
        return True
    if record.get("target_reached") is True:
        return True
    if any(
        event.get("event_type") == "conversation_end_candidate"
        for event in call_events
    ):
        return True
    if record.get("status") in {"failed", "benchmark_invalid"}:
        return False
    final_state = record.get("final_state")
    if isinstance(final_state, str) and final_state.startswith("FAIL"):
        return False
    return False


def _handling_from_events(call_events: list[dict[str, Any]]) -> str | None:
    observed: list[str] = []
    for event in call_events:
        if event.get("event_type") != "tool_result" or not event_succeeded(event):
            continue
        metadata = event.get("metadata")
        annotation = metadata.get(OPERATION_ANNOTATION_KEY) if isinstance(metadata, dict) else None
        handling_type = annotation.get("handling_type") if isinstance(annotation, dict) else None
        if handling_type in COMPLETE_HANDLING_TYPES:
            observed.append(str(handling_type))
    for handling_type in ("escalation", "callback_commitment", "refusal"):
        if handling_type in observed:
            return handling_type
    if observed and all(item == "guidance_inquiry_answer" for item in observed):
        return "guidance_inquiry_answer"
    if observed:
        return "change_procedure"
    return None


def _handling_from_world(value: Any) -> str | None:
    world = value.get("data") if isinstance(value, dict) and isinstance(value.get("data"), dict) else value
    if not isinstance(world, dict):
        return None
    found: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for field in FINAL_HANDLING_FIELDS:
                candidate = item.get(field)
                normalized = HANDLING_TYPE_ALIASES.get(candidate, candidate)
                if normalized in COMPLETE_HANDLING_TYPES:
                    found.append(str(normalized))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(world)
    for handling_type in ("escalation", "callback_commitment", "refusal"):
        if handling_type in found:
            return handling_type
    if found and all(item == "guidance_inquiry_answer" for item in found):
        return "guidance_inquiry_answer"
    if found:
        return "change_procedure"
    return None


def _has_successful_world_change(call_events: list[dict[str, Any]]) -> bool:
    return any(
        event.get("event_type") == "tool_result"
        and event_succeeded(event)
        and (
            (
                isinstance((event.get("metadata") or {}).get("world_diff"), dict)
                and bool(event["metadata"]["world_diff"])
            )
            or (
                isinstance(event.get("result"), dict)
                and isinstance(event["result"].get("changes"), list)
                and bool(event["result"]["changes"])
            )
        )
        for event in call_events
    )


def event_succeeded(event: dict[str, Any]) -> bool:
    """Return whether a tool-result event succeeded without a hard fail."""

    result = event.get("result")
    metadata = event.get("metadata")
    return (
        isinstance(result, dict)
        and result.get("ok") is not False
        and result.get("error") is None
        and not (metadata.get("hard_fails") if isinstance(metadata, dict) else None)
    )


# Kept for existing importers; prefer ``event_succeeded``.
_event_succeeded = event_succeeded


def _outcome(handling_type: str, completed: bool, reason: str) -> dict[str, Any]:
    return {
        "schema_version": RUN_OUTCOME_VERSION,
        "handling_type": handling_type,
        "completed": completed,
        "reason": reason,
    }
