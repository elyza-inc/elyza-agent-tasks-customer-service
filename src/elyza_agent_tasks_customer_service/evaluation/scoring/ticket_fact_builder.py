"""Build a post-call fact ticket from frozen call events."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import json
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.engine.package_build import TICKET_CATALOG_ACTION_CODES


FACT_BUILDER_VERSION = "ticket_fact_builder"


def build_ticket_fact(
    call_events: list[dict[str, Any]],
    *,
    run_outcome: dict[str, Any] | None = None,
    claimed_evidence_refs: list[dict[str, str]] | None = None,
    target_candidates_out: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Build one fact ticket from object-root call-event rows.

    Every paired non-catalog Tool result becomes one performed action whose
    ``action_code`` is the Tool ID. Targets are the ``id``/``*_id`` string
    values of a successful result's ``rows``. ``handling_type`` comes from
    ``run_outcome`` and is ``None`` without it.

    ``claimed_evidence_refs`` may contain the ticket's object-row reference
    list. Valid run-local claims are retained in their original order and any
    missing mandatory SOP or successful Tool receipts are appended, so exact
    comparison rejects invalid references and evidence omissions. ``None``
    returns the canonical mandatory evidence only. Malformed call rows are
    ignored rather than treated as successful actions.
    """

    if not isinstance(call_events, list):
        raise ValueError("call_events must be an array")
    pending_calls: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    actions: list[dict[str, Any]] = []
    target_ids: list[dict[str, str]] = []
    required_evidence_refs: list[dict[str, str]] = []
    sop_refs: list[str] = []

    for event in call_events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        tool = event.get("tool")
        if event_type == "tool_call" and isinstance(tool, str):
            pending_calls[tool].append(event)
            continue
        if event_type != "tool_result" or not isinstance(tool, str):
            continue
        call = pending_calls[tool].popleft() if pending_calls[tool] else {}
        if tool in TICKET_CATALOG_ACTION_CODES and not _event_succeeded(event):
            # SOP catalog calls are never performed actions, whether they succeed or fail.
            continue
        sop_ref = _sop_evidence_ref(event)
        if sop_ref is not None:
            if sop_ref:
                _append_unique(required_evidence_refs, {"kind": "sop", "ref": sop_ref})
                if sop_ref not in sop_refs:
                    sop_refs.append(sop_ref)
            continue
        event_targets = _result_targets(event)
        for target in event_targets:
            _append_unique(target_ids, target)
        target_ref = _target_ref(event_targets)
        succeeded = _event_succeeded(event)
        outcome = "succeeded" if succeeded else "failed"
        receipt_ref = event.get("event_ref") or call.get("event_ref")
        receipt_refs = [str(receipt_ref)] if isinstance(receipt_ref, str) else []
        actions.append(
            {
                "action_code": tool,
                "target_ref": target_ref,
                "outcome": outcome,
                "receipt_refs": receipt_refs,
            }
        )
        if target_candidates_out is not None:
            # Any ID the result carried is a valid target_ref; the first one
            # depends only on the result's key order.
            target_candidates_out.append([target["value"] for target in event_targets])
        if succeeded and receipt_refs:
            _append_unique(
                required_evidence_refs,
                {"kind": "tool_receipt", "ref": receipt_refs[0]},
            )

    evidence_refs = _confirmed_evidence_refs(
        call_events=call_events,
        claimed=claimed_evidence_refs,
        required=required_evidence_refs,
        sop_refs=sop_refs,
    )
    handling_type = run_outcome["handling_type"] if isinstance(run_outcome, dict) else None

    return {
        "schema_version": FACT_BUILDER_VERSION,
        "handling_type": handling_type,
        "target_ids": target_ids,
        "performed_actions": actions,
        "answer_summary": None,
        "refusal_reason_code": None,
        "promises": [],
        "escalation_initial_response": [],
        "escalation": None,
        "evidence_refs": evidence_refs,
    }


def _sop_evidence_ref(event: dict[str, Any]) -> str | None:
    if not _event_succeeded(event):
        return None
    result = event.get("result")
    if not isinstance(result, dict):
        return None
    sop_id = result.get("sop_id")
    nested_sop = result.get("sop")
    if not isinstance(sop_id, str) and isinstance(nested_sop, dict):
        sop_id = nested_sop.get("id")
    if not isinstance(sop_id, str) or not sop_id:
        candidates = result.get("candidates")
        if isinstance(candidates, list) and all(
            isinstance(candidate, dict)
            and isinstance(candidate.get("sop_id"), str)
            for candidate in candidates
        ):
            return ""
        categories = result.get("categories")
        if isinstance(categories, list) and all(
            isinstance(category, dict)
            and isinstance(category.get("category_id"), str)
            for category in categories
        ):
            return ""
        return None
    sop_ref = result.get("sop_ref")
    if isinstance(sop_ref, str) and sop_ref:
        return sop_ref
    return sop_id if sop_id.lower().startswith("sop:") else f"SOP:{sop_id}"


def _result_targets(
    event: dict[str, Any],
) -> list[dict[str, str]]:
    if not _event_succeeded(event):
        return []
    result = event.get("result")
    if not isinstance(result, dict):
        return []
    target_ids: list[dict[str, str]] = []
    rows = result.get("rows")
    if isinstance(rows, list):
        for record in rows:
            if not isinstance(record, dict):
                continue
            for id_type, record_id in record.items():
                if (
                    isinstance(id_type, str)
                    and (id_type == "id" or id_type.endswith("_id"))
                    and isinstance(record_id, str)
                    and record_id
                ):
                    _append_unique(target_ids, {"id_type": id_type, "value": record_id})
    return target_ids


def _confirmed_evidence_refs(
    *,
    call_events: list[dict[str, Any]],
    claimed: list[dict[str, str]] | None,
    required: list[dict[str, str]],
    sop_refs: list[str],
) -> list[dict[str, str]]:
    if claimed is None:
        return required
    event_by_ref = {
        event["event_ref"]: event
        for event in call_events
        if isinstance(event, dict)
        and isinstance(event.get("event_ref"), str)
        and event["event_ref"]
    }
    source_refs = _source_refs(call_events)
    confirmed: list[dict[str, str]] = []
    for item in claimed:
        if not isinstance(item, dict) or set(item) != {"kind", "ref"}:
            continue
        kind = item.get("kind")
        ref = item.get("ref")
        if not isinstance(ref, str) or not ref:
            continue
        valid = kind == "event" and ref in event_by_ref
        if kind == "tool_receipt":
            valid = (
                ref in event_by_ref
                and event_by_ref[ref].get("event_type") == "tool_result"
            )
        elif kind == "sop":
            valid = _normalized_sop_ref(ref) in {
                _normalized_sop_ref(value) for value in sop_refs if value
            }
        elif kind == "source":
            valid = ref in source_refs
        if valid:
            _append_unique(confirmed, deepcopy(item))
    normalized_confirmed = {_normalized_evidence_ref(item) for item in confirmed}
    for item in required:
        if not item.get("ref"):
            continue
        if _normalized_evidence_ref(item) not in normalized_confirmed:
            _append_unique(confirmed, deepcopy(item))
    return confirmed


def _source_refs(call_events: list[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            source_ref = value.get("source_ref")
            if isinstance(source_ref, str) and source_ref:
                refs.add(source_ref)
            source_refs = value.get("source_refs")
            if isinstance(source_refs, list):
                refs.update(item for item in source_refs if isinstance(item, str) and item)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(call_events)
    return refs


def _normalized_sop_ref(value: str) -> str:
    return value.lower()


def _normalized_evidence_ref(value: dict[str, str]) -> tuple[str, str]:
    ref = value["ref"]
    if value["kind"] == "sop":
        ref = _normalized_sop_ref(ref)
    return value["kind"], ref


def _event_succeeded(event: dict[str, Any]) -> bool:
    result = event.get("result")
    if not isinstance(result, dict) or result.get("error") is not None:
        return False
    return result.get("ok") is not False


# World rows record the handling in the data vocabulary; tickets use the ticket enum.
WORLD_HANDLING_TYPES = {
    "change": "change_procedure",
    "guidance": "guidance_inquiry_answer",
    "refusal": "refusal",
    "escalation": "escalation",
    "callback": "callback_commitment",
}


def recorded_handling_type(initial_world: dict[str, Any], final_world: dict[str, Any]) -> str | None:
    """Return the ticket handling type of rows the run created or changed, or ``None``.

    Call events carry no world diff, so for incomplete runs the fact comes from
    ``values.handling_type`` of rows that differ between the initial and final
    world, with the precedence of ``_handling_type``.
    """

    roles: list[str | None] = []
    for table, rows in final_world.items():
        if not isinstance(rows, list):
            continue
        before = {
            json.dumps(row.get("row_identity"), sort_keys=True, ensure_ascii=False): row
            for row in initial_world.get(table) or []
            if isinstance(row, dict)
        }
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = json.dumps(row.get("row_identity"), sort_keys=True, ensure_ascii=False)
            if before.get(key) == row:
                continue
            value = (row.get("values") or {}).get("handling_type")
            if value in WORLD_HANDLING_TYPES:
                roles.append(WORLD_HANDLING_TYPES[value])
    return _handling_type(roles)


def _handling_type(successful_roles: list[str | None]) -> str | None:
    if not successful_roles:
        return None
    for handling_type in ("escalation", "callback_commitment"):
        if handling_type in successful_roles:
            return handling_type
    if all(role == "guidance_inquiry_answer" for role in successful_roles):
        return "guidance_inquiry_answer"
    if "refusal" in successful_roles:
        return "refusal"
    return "change_procedure"


def _target_ref(event_targets: list[dict[str, str]]) -> str | None:
    if not event_targets:
        return None
    return event_targets[0]["value"]


def _append_unique(values: list[dict[str, str]], value: dict[str, str]) -> None:
    if value not in values:
        values.append(value)
