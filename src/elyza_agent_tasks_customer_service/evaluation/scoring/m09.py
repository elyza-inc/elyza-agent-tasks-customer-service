"""Canonical M09 SOP-application calculation.

``calculate_m09`` accepts an object-root scenario and an array of runtime event
objects. The scenario must declare a non-empty ``expected_sop_path`` string
array. Event rows are paired as FIFO tool-call/tool-result attempts. A
malformed path raises ``ValueError``; incomplete or failed attempts remain
observations and do not become successful applications.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SOP_SEARCH_TOOL = "search_sops_in_category"
SOP_DETAIL_TOOL = "get_sop"
M09_CALCULATION_VERSION = "m09_sop_application"
SUCCESS_BOOLEAN_FIELDS = (
    "ok",
    "success",
    "passed",
    "identity_verified",
    "verified",
)


def required_sop_path(scenario: Any) -> list[str]:
    """Return the required-SOP path declared by a scenario object."""

    if not isinstance(scenario, dict):
        raise ValueError("scenario must be an object")
    path = scenario.get("expected_sop_path")
    if (
        not isinstance(path, list)
        or not path
        or any(not isinstance(item, str) or not item for item in path)
    ):
        raise ValueError("scenario.expected_sop_path must be a non-empty string array")
    return list(path)


def calculate_m09(
    *,
    scenario: Any,
    events: Any,
) -> dict[str, Any]:
    """Calculate final and first-shot M09 bits from one frozen call event log."""

    expected_path = required_sop_path(scenario)
    attempts = _tool_attempts(events)
    detail_attempts = [
        item for item in attempts if item["tool"] == SOP_DETAIL_TOOL
    ]
    successful_details = [
        item for item in detail_attempts if item["success"] is True
    ]
    observed_path = [
        item["sop_id"]
        for item in successful_details
        if isinstance(item["sop_id"], str) and item["sop_id"]
    ]
    expected_set = set(expected_path)
    search_hits: set[str] = set()
    for attempt in attempts:
        if attempt["tool"] != SOP_SEARCH_TOOL or attempt["success"] is not True:
            continue
        search_hits.update(_result_sop_ids(attempt.get("result")))
    first_detail = detail_attempts[0] if detail_attempts else None
    first_sop_id = first_detail["sop_id"] if first_detail is not None else None
    one_shot_correct = (
        isinstance(first_sop_id, str)
        and first_sop_id in expected_set
        and first_detail["success"] is True
    )
    k = _correct_sop_open_count(successful_details, expected_set)
    score_inverse_k = 1 / k if k is not None else 0.0
    applications: list[dict[str, Any]] = []
    domain_attempts = [item for item in attempts if item["success"] is True]
    for sop_id in expected_path:
        matching_details = [
            item for item in successful_details if item["sop_id"] == sop_id
        ]
        detail_orders = [item["call_order"] for item in matching_details]
        executable_tool = _executable_tool(matching_details)
        application_attempts: list[dict[str, Any]] = []
        if executable_tool is not None:
            application_attempts = [
                item
                for item in domain_attempts
                if item["tool"] == executable_tool
                and any(order < item["call_order"] for order in detail_orders)
            ]
        applied = bool(application_attempts) if executable_tool is not None else False
        applications.append(
            {
                "sop_id": sop_id,
                "executable_tool": executable_tool,
                "detail_call_orders": detail_orders,
                "application_call_orders": [
                    item["call_order"] for item in application_attempts
                ],
                "detail_before_application": applied,
                "applied": applied,
            }
        )
    return {
        "calculation_version": M09_CALCULATION_VERSION,
        "expected_sop_path": expected_path,
        "observed_successful_detail_path": observed_path,
        "first_detail_sop_id": first_sop_id,
        "first_detail_call_order": (
            first_detail["call_order"] if first_detail is not None else None
        ),
        "search_hit_correct": expected_set <= search_hits,
        "search_hit_sop_ids": sorted(search_hits),
        "final_correct": all(item["detail_call_orders"] for item in applications),
        "one_shot_correct": one_shot_correct,
        "k": k,
        "score_inverse_k": score_inverse_k,
        "applications": applications,
    }


def _correct_sop_open_count(
    successful_details: list[dict[str, Any]], expected_sop_ids: set[str]
) -> int | None:
    """Return successful detail opens through the first complete required path."""

    opened_expected: set[str] = set()
    for count, detail in enumerate(successful_details, start=1):
        sop_id = detail["sop_id"]
        if sop_id in expected_sop_ids:
            opened_expected.add(sop_id)
        if expected_sop_ids <= opened_expected:
            return count
    return None


def _result_sop_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        sop_id = value.get("sop_id")
        if isinstance(sop_id, str) and sop_id:
            found.add(sop_id)
        for child in value.values():
            found.update(_result_sop_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_result_sop_ids(child))
    return found


def _tool_attempts(events: Any) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        raise ValueError("M09 events must be an array")
    pending: dict[str, list[dict[str, Any]]] = {}
    attempts: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"M09 events[{index}] must be an object")
        if event.get("epoch", "call") != "call":
            continue
        tool = event.get("tool")
        if not isinstance(tool, str) or not tool:
            continue
        if event.get("event_type") == "tool_call":
            arguments = event.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            sop_id = arguments.get("sop_id")
            if not isinstance(sop_id, str):
                sop_id = None
            attempt = {
                "tool": tool,
                "arguments": deepcopy(arguments),
                "result": None,
                "call_order": _event_order(event, index),
                "success": False,
                "sop_id": sop_id,
            }
            attempts.append(attempt)
            pending.setdefault(tool, []).append(attempt)
            continue
        if event.get("event_type") != "tool_result":
            continue
        queued = pending.get(tool) or []
        if not queued:
            continue
        attempt = queued.pop(0)
        result = event.get("result")
        attempt["result"] = deepcopy(result)
        attempt["success"] = _result_succeeded(result)
        if isinstance(result, dict):
            result_id = result.get("sop_id")
            if attempt["tool"] == SOP_DETAIL_TOOL:
                sop = result.get("sop")
                result_id = sop.get("id") if isinstance(sop, dict) else None
            if isinstance(result_id, str):
                attempt["sop_id"] = result_id
    return attempts


def _event_order(event: dict[str, Any], fallback: int) -> int:
    seq = event.get("seq")
    if isinstance(seq, int) and not isinstance(seq, bool):
        return seq
    return fallback


def _result_succeeded(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for key in ("error", "error_type"):
        if value.get(key) not in (None, "", False):
            return False
    return not any(value.get(key) is False for key in SUCCESS_BOOLEAN_FIELDS)


def _executable_tool(details: list[dict[str, Any]]) -> str | None:
    """Return the last Tool step of the first opened SOP that declares one."""

    for detail in details:
        result = detail.get("result")
        sop = result.get("sop") if isinstance(result, dict) else None
        if isinstance(sop, dict):
            tool_ids = [
                step.get("tool_id")
                for step in sop.get("steps", [])
                if isinstance(step, dict)
                and step.get("kind") == "tool"
                and isinstance(step.get("tool_id"), str)
            ]
            if tool_ids:
                return tool_ids[-1]
    return None
