"""Deterministic typed-ticket scoring from observed run facts."""

from __future__ import annotations

import json
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.scoring.ticket_conversation_questions import (
    build_ticket_conversation_questions,
)
from elyza_agent_tasks_customer_service.evaluation.engine.package_build import TICKET_CATALOG_ACTION_CODES
from elyza_agent_tasks_customer_service.evaluation.scoring.ticket_fact_builder import build_ticket_fact, recorded_handling_type


class TicketEnumSelectionError(ValueError):
    """Raised when a ticket value is outside its scenario-declared catalog."""



SCORER_VERSION = "post_call_ticket_scoring"
DETERMINISTIC_TICKET_FIELDS = (
    "handling_type",
    "target_ids",
    "performed_actions",
    "evidence_refs",
)
CONVERSATIONAL_TICKET_FIELDS = (
    "refusal_reason",
    "promises",
    "answer_summary",
)
DETERMINISTIC_ENUM_CATALOGS = (
    "handling_types",
    "action_codes",
    "id_types",
    "evidence_kinds",
)


def score_post_call_ticket(
    *,
    contract: dict[str, Any],
    artifact: dict[str, Any] | None,
    initial_world: dict[str, Any],
    final_world: dict[str, Any],
    call_events: list[dict[str, Any]],
    run_outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score the four deterministic operator-ticket fields against run-local facts.

    ``artifact`` is the post-call ticket generator result. Contract defects
    produce ``contract_invalid``; infrastructure and size failures produce
    ``not_measurable``; schema-valid model errors produce ``fail``. M15 is
    scored by the conversation-log judge, so it is ``N/A`` here.
    """

    try:
        _validate_deterministic_contract(contract)
    except ValueError as exc:
        return _contract_invalid(str(exc))
    if artifact is None:
        return _not_measurable("operator_ticket_missing")
    artifact_status = artifact.get("status")
    if artifact_status in {"infra_error", "not_measurable"}:
        return _not_measurable(str(artifact.get("reason") or artifact_status))
    if artifact_status == "contract_invalid":
        return _contract_invalid(str(artifact.get("reason") or "ticket_artifact_contract_invalid"))
    ticket = artifact.get("ticket")
    if artifact_status != "submitted" or not isinstance(ticket, dict):
        return _model_fail(str(artifact.get("reason") or "operator_ticket_invalid"))
    try:
        _validate_deterministic_ticket(ticket, contract["enum_catalog"])
    except TicketEnumSelectionError as exc:
        return _model_fail(str(exc))
    target_candidates: list[list[str]] = []
    fact_ticket = build_ticket_fact(
        call_events,
        run_outcome=run_outcome,
        claimed_evidence_refs=ticket["evidence_refs"],
        target_candidates_out=target_candidates,
    )
    if isinstance(run_outcome, dict) and run_outcome.get("completed") is False:
        # An incomplete run's handling is what the operator actually recorded in
        # the world, not the run outcome's "incomplete" (which the operator cannot know).
        fact_ticket["handling_type"] = recorded_handling_type(initial_world, final_world) or fact_ticket["handling_type"]
    # A call the persona interrupted was never executed, and build_ticket_fact leaves it
    # out; a ticket that lists it as a failed action (it saw "not executed") is not wrong.
    not_executed = {
        str(event.get("event_ref"))
        for event in call_events
        if isinstance(event, dict) and event.get("event_type") == "interrupted_by_persona"
    }
    observed_ticket = dict(ticket)
    if not_executed and isinstance(ticket.get("performed_actions"), list):
        observed_ticket["performed_actions"] = [
            item
            for item in ticket["performed_actions"]
            if not (
                isinstance(item, dict)
                and item.get("receipt_refs")
                and set(map(str, item["receipt_refs"])) <= not_executed
            )
        ]
    deterministic = compare_ticket_facts(
        expected=fact_ticket,
        observed=observed_ticket,
        target_candidates=target_candidates,
    )
    return {
        "schema_version": SCORER_VERSION,
        "status": "pass" if deterministic["passed"] else "fail",
        "ticket_fact": fact_ticket,
        "deterministic_comparison": deterministic,
        "ticket_fidelity_deterministic": deterministic["score"],
        "metric_results": {
            "M14": {
                "ticket_fidelity_deterministic": deterministic["score"],
                "deterministic_field_results": deterministic["field_results"],
                "passed": deterministic["passed"],
            },
            "M15": {
                "ticket_fidelity_conversational": "N/A",
            },
        },
        "artifact_receipt_id": (artifact.get("receipt") or {}).get("receipt_id"),
    }


def compare_ticket_facts(
    *,
    expected: dict[str, Any],
    observed: dict[str, Any],
    target_candidates: list[list[str]],
) -> dict[str, Any]:
    """Compare deterministic ticket fields; ``target_ids`` is an ID-pair multiset.

    ``target_ids`` accepts lists of ``{"id_type": str, "value": str}`` objects;
    malformed values are mismatches. ``performed_actions`` may name any ID in
    the matching ``target_candidates`` row as ``target_ref``. The other fields
    use exact JSON equality.
    """

    field_results: dict[str, bool] = {}
    mismatches: list[dict[str, Any]] = []
    observed = dict(observed)
    if isinstance(observed.get("performed_actions"), list):
        # SOP catalog calls are never expected actions (ticket_fact_builder drops them),
        # but they are declared action codes, so a ticket that lists them is not wrong.
        observed["performed_actions"] = [
            item
            for item in observed["performed_actions"]
            if not (isinstance(item, dict) and item.get("action_code") in TICKET_CATALOG_ACTION_CODES)
        ]
    for field in DETERMINISTIC_TICKET_FIELDS:
        if field == "target_ids":
            matched = _target_ids_match(observed.get(field), expected.get(field))
        elif field == "performed_actions":
            matched = _performed_actions_match(observed.get(field), expected.get(field), target_candidates)
        elif field == "evidence_refs":
            matched = _evidence_refs_match(observed.get(field), expected.get(field))
        else:
            matched = observed.get(field) == expected.get(field)
        field_results[field] = matched
        if not matched:
            mismatches.append(
                {
                    "field": field,
                    "expected": expected.get(field),
                    "observed": observed.get(field),
                }
            )
    matched_count = sum(field_results.values())
    return {
        "passed": matched_count == len(DETERMINISTIC_TICKET_FIELDS),
        "score": matched_count / len(DETERMINISTIC_TICKET_FIELDS),
        "field_results": field_results,
        "mismatches": mismatches,
    }


def _evidence_refs_match(observed: Any, expected: Any) -> bool:
    """Compare evidence references as a set; the expected side is already de-duplicated."""

    if not isinstance(observed, list) or not isinstance(expected, list):
        return observed == expected
    key = lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False)
    return {key(item) for item in observed} == {key(item) for item in expected}


def _performed_actions_match(observed: Any, expected: Any, target_candidates: list[list[str]]) -> bool:
    """Exact per-action match, except ``target_ref`` may be any ID the result carried."""

    if not isinstance(observed, list) or not isinstance(expected, list) or len(observed) != len(expected):
        return False
    for seen, fact, candidates in zip(observed, expected, target_candidates):
        if not isinstance(seen, dict) or not isinstance(fact, dict):
            return False
        target_ok = seen.get("target_ref") == fact.get("target_ref") or seen.get("target_ref") in candidates
        if not target_ok or {**seen, "target_ref": None} != {**fact, "target_ref": None}:
            return False
    return True


def _target_ids_match(observed: Any, expected: Any) -> bool:
    """Return whether valid target-ID lists contain the same ID/value pairs.

    Both inputs must be lists of exactly ``{"id_type": str, "value": str}``
    objects; invalid inputs return ``False`` instead of raising.
    """

    def pairs(value: Any) -> list[tuple[str, str]] | None:
        if not isinstance(value, list):
            return None
        result = []
        for item in value:
            if (
                not isinstance(item, dict)
                or set(item) != {"id_type", "value"}
                or not isinstance(item["id_type"], str)
                or not isinstance(item["value"], str)
            ):
                return None
            result.append((item["id_type"], item["value"]))
        return sorted(result)

    observed_pairs = pairs(observed)
    expected_pairs = pairs(expected)
    return observed_pairs is not None and observed_pairs == expected_pairs


def _validate_deterministic_contract(contract: Any) -> None:
    if not isinstance(contract, dict) or set(contract) != {
        "schema_version",
        "enum_catalog",
        "comparison_fields",
    }:
        raise ValueError("deterministic ticket contract fields are invalid")
    if contract["schema_version"] != "post_call_ticket_contract":
        raise ValueError("unsupported deterministic ticket contract schema")
    if contract["comparison_fields"] != list(DETERMINISTIC_TICKET_FIELDS):
        raise ValueError("ticket contract comparison_fields must be the four deterministic fields")
    catalog = contract["enum_catalog"]
    if not isinstance(catalog, dict) or set(catalog) != set(DETERMINISTIC_ENUM_CATALOGS):
        raise ValueError("deterministic ticket enum_catalog fields are invalid")
    for field in DETERMINISTIC_ENUM_CATALOGS:
        values = catalog[field]
        if not isinstance(values, list) or len(values) != len(set(values)) or any(
            not isinstance(item, str) or not item for item in values
        ):
            raise ValueError(f"enum_catalog.{field} must be a unique string array")
        if not values:
            raise ValueError(f"enum_catalog.{field} cannot be empty")


def _validate_deterministic_ticket(ticket: Any, catalog: dict[str, Any]) -> None:
    if not isinstance(ticket, dict) or frozenset(ticket) not in {
        frozenset(DETERMINISTIC_TICKET_FIELDS),
        frozenset(DETERMINISTIC_TICKET_FIELDS + CONVERSATIONAL_TICKET_FIELDS),
    }:
        raise ValueError("ticket fields are invalid")
    if ticket["handling_type"] not in catalog["handling_types"]:
        raise TicketEnumSelectionError("handling_type is not scenario-declared")
    for field in DETERMINISTIC_TICKET_FIELDS[1:]:
        if not isinstance(ticket[field], list):
            raise ValueError(f"{field} must be an array")
    for index, item in enumerate(ticket["target_ids"]):
        if not isinstance(item, dict) or set(item) != {"id_type", "value"}:
            raise ValueError(f"target_ids[{index}] fields are invalid")
        if item["id_type"] not in catalog["id_types"]:
            raise TicketEnumSelectionError(f"target_ids[{index}].id_type is not scenario-declared")
        if not isinstance(item["value"], str) or not item["value"]:
            raise ValueError(f"target_ids[{index}].value must be a non-empty string")
    for index, item in enumerate(ticket["performed_actions"]):
        if not isinstance(item, dict) or set(item) != {
            "action_code",
            "target_ref",
            "outcome",
            "receipt_refs",
        }:
            raise ValueError(f"performed_actions[{index}] fields are invalid")
        if item["action_code"] not in catalog["action_codes"]:
            raise TicketEnumSelectionError(f"performed_actions[{index}].action_code is not scenario-declared")
        if item["target_ref"] is not None and not isinstance(item["target_ref"], str):
            raise ValueError(f"performed_actions[{index}].target_ref must be a string or null")
        if item["outcome"] not in {"succeeded", "failed"}:
            raise ValueError(f"performed_actions[{index}].outcome is invalid")
        refs = item["receipt_refs"]
        if not isinstance(refs, list) or any(not isinstance(ref, str) or not ref for ref in refs):
            raise ValueError(f"performed_actions[{index}].receipt_refs must be a string array")
    for index, item in enumerate(ticket["evidence_refs"]):
        if not isinstance(item, dict) or set(item) != {"kind", "ref"}:
            raise ValueError(f"evidence_refs[{index}] fields are invalid")
        if item["kind"] not in catalog["evidence_kinds"]:
            raise TicketEnumSelectionError(f"evidence_refs[{index}].kind is not scenario-declared")
        if not isinstance(item["ref"], str) or not item["ref"]:
            raise ValueError(f"evidence_refs[{index}].ref must be a non-empty string")
    if set(ticket) != set(DETERMINISTIC_TICKET_FIELDS):
        build_ticket_conversation_questions(ticket)


def _contract_invalid(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCORER_VERSION,
        "status": "contract_invalid",
        "reason": reason,
        "metric_results": {},
    }


def _not_measurable(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCORER_VERSION,
        "status": "not_measurable",
        "reason": reason,
        "metric_results": {
            "M14": "N/M",
            "M15": "N/M",
        },
    }


def _model_fail(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCORER_VERSION,
        "status": "fail",
        "reason": reason,
        "metric_results": {
            "M14": {
                "ticket_fidelity_deterministic": 0.0,
                "deterministic_field_results": {
                    field: False for field in DETERMINISTIC_TICKET_FIELDS
                },
                "passed": False,
            },
            "M15": "N/M",
        },
    }
