"""Offline, instance-driven scoring for one completed conversation.

``score_record`` accepts three decoded values: an object-root scenario,
an object-root completed record, and an array of object event rows.  The
scenario must contain ``measurement_contract``.  The record must contain
``run_id``, ``initial_world``, ``final_world``, and
``operator_ticket_artifact``; optional ``extended_metric_artifacts`` is an
object holding the M17 ``m17_parse_bundle``. Malformed roots, duplicate
instance IDs, unsupported contract versions, and missing scorer inputs raise
``ValueError``; unimplemented metrics are emitted as explicit ``N/M`` rows.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.scoring.core_metric_scoring import (
    M04_CONTRACT_FIELD,
    M05_CONTRACT_FIELD,
    score_m04_customer_trigger_control,
    score_m05_obligations,
)
from elyza_agent_tasks_customer_service.evaluation.contracts.event_epoch import validate_event_epoch_order
from elyza_agent_tasks_customer_service.evaluation.contracts.event_identity import align_referenced_ideal_event_aliases
from elyza_agent_tasks_customer_service.evaluation.observers.japanese_parse_adapter import is_japanese_parser_dependency_error
from elyza_agent_tasks_customer_service.evaluation.contracts.metric_inventory import ACTIVE_METRIC_IDS, assert_no_single_overall_score
from elyza_agent_tasks_customer_service.evaluation.scoring.post_call_ticket_scoring import score_post_call_ticket
from elyza_agent_tasks_customer_service.evaluation.scoring.resolution_scoring import score_resolution
from elyza_agent_tasks_customer_service.evaluation.contracts.run_outcome import derive_run_outcome
from elyza_agent_tasks_customer_service.evaluation.scoring.extended_metric_scoring import (
    EXTENDED_METRIC_IDS,
    score_extended_metrics,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.m09 import calculate_m09
from elyza_agent_tasks_customer_service.evaluation.scoring.conversation_log_metrics import score_conversation_log_metrics
from elyza_agent_tasks_customer_service.evaluation.contracts.interaction_contract import interaction_event_refs


REPORT_SCHEMA_VERSION = "scoring_report"
MEASUREMENT_CONTRACT_SCHEMA_VERSION = "measurement_contract"
MEASUREMENT_CONTRACT_IMPL_VERSION = "measurement_contract_materializer"
IMPLEMENTED_METRIC_IDS = frozenset(
    (
        "M01",
        "M04",
        "M05",
        "M09",
        "M11",
        "M14",
        "M15",
        "M16",
        "M17",
        "M19",
    )
)
UNMEASURED_REASON_BY_METRIC = {
    "M20": "audio_mode_not_selected",
    "M21": "audio_mode_not_selected",
    "M22": "audio_mode_not_selected",
    "M23": "audio_mode_not_selected",
}
TEXT_AUDIO_METRIC_IDS = frozenset(UNMEASURED_REASON_BY_METRIC)
DEFAULT_UNMEASURED_REASON = "metric_not_implemented"
EXTENDED_ARTIFACT_RECORD_FIELD = "extended_metric_artifacts"
ARTIFACT_DEPENDENT_METRIC_IDS = frozenset(EXTENDED_METRIC_IDS)
RUNTIME_EVIDENCE_PRODUCER_METRIC_IDS = frozenset(
    IMPLEMENTED_METRIC_IDS - ARTIFACT_DEPENDENT_METRIC_IDS
)
TICKET_METRIC_IDS = frozenset(("M14", "M15"))
MISSING_TICKET_CONTRACT_REASON = "post_call_ticket_contract_missing"


def score_record(
    *,
    scenario: dict[str, Any],
    record: dict[str, Any],
    event_log: list[dict[str, Any]],
    frozen_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one scenario report without a combined score.

    Inputs are already-decoded Python values; this module loads no files.
    Every metric row is driven by one declared
    ``measurement_contract.metric_instances`` entry.
    """

    contract, instances = _validate_inputs(scenario, record, event_log)
    epoch = validate_event_epoch_order(event_log)
    call_events = align_referenced_ideal_event_aliases(
        scenario,
        epoch["call_events"],
        _core_metric_event_refs(scenario),
    )
    extended_rows_by_id = _extended_metric_rows(
        scenario=scenario,
        record=record,
        instances=instances,
    )
    context: dict[str, Any] = {
        "resolution_score": None,
        "ticket_score": None,
        "m09_score": None,
        "m11_score": None,
    }
    rows = [
        _score_instance(
            instance=instance,
            scenario=scenario,
            record=record,
            call_events=call_events,
            context=context,
            extended_rows_by_id=extended_rows_by_id,
            frozen_reference=frozen_reference,
        )
        for instance in instances
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scenario_id": scenario["scenario_id"],
        "run_id": record["run_id"],
        "measurement_contract_id": contract["contract_id"],
        "metric_results": rows,
        "violations": [
            violation
            for row in rows
            for violation in row["violations"]
        ],
        "not_measured": [
            {
                "metric_instance_id": row["metric_instance_id"],
                "metric_id": row["metric_id"],
                "reason": row["reason"],
            }
            for row in rows
            if row["status"] == "N/M"
        ],
        "diagnostics": {
            "implemented_metric_ids": sorted(IMPLEMENTED_METRIC_IDS),
            "artifact_dependent_metric_ids": sorted(ARTIFACT_DEPENDENT_METRIC_IDS),
            "runtime_evidence_producer_metric_ids": sorted(
                RUNTIME_EVIDENCE_PRODUCER_METRIC_IDS
            ),
            "call_event_count": len(epoch["call_events"]),
            "post_call_event_count": len(epoch["post_call_events"]),
            "call_event_log_hash": epoch["call_event_log_hash"],
            "post_call_ticket_contract": {
                "available": isinstance(
                    scenario.get("post_call_ticket_contract"), dict
                ),
                "dependent_metric_ids": sorted(TICKET_METRIC_IDS),
            },
            "frozen_final_world_reference": {
                "available": frozen_reference is not None,
                "dependent_metric_ids": ["M01"],
            },
        },
        "aggregation_policy": "single_overall_score_forbidden",
    }
    assert_no_single_overall_score(report)
    return report


def _core_metric_event_refs(scenario: dict[str, Any]) -> set[str]:
    """Return ideal-event refs declared by deterministic evaluation inputs."""

    refs: set[str] = set()
    m05 = scenario.get(M05_CONTRACT_FIELD)
    if isinstance(m05, dict):
        for obligation in m05.get("obligations", []):
            if not isinstance(obligation, dict):
                continue
            refs.update(
                ref
                for key in (
                    "deadline_event_id",
                    "consent_event_id",
                    "protected_action_event_id",
                )
                for ref in (obligation.get(key),)
                if isinstance(ref, str)
            )
            refs.update(
                ref
                for ref in obligation.get("required_event_ids", [])
                if isinstance(ref, str)
            )
    interaction = scenario.get("interaction_closed_questions")
    if isinstance(interaction, dict):
        refs.update(interaction_event_refs(interaction))
    return refs


def _validate_inputs(
    scenario: Any,
    record: Any,
    event_log: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be an object")
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    if not isinstance(event_log, list) or any(
        not isinstance(row, dict) for row in event_log
    ):
        raise ValueError("event_log must be an object array")
    scenario_id = scenario.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario.scenario_id must be a non-empty string")
    required_record_fields = {
        "run_id",
        "initial_world",
        "final_world",
        "operator_ticket_artifact",
    }
    missing_record_fields = required_record_fields - set(record)
    if missing_record_fields:
        raise ValueError(
            f"record fields missing: {sorted(missing_record_fields)}"
        )
    if not isinstance(record["run_id"], str) or not record["run_id"]:
        raise ValueError("record.run_id must be a non-empty string")
    if not isinstance(record["initial_world"], dict):
        raise ValueError("record.initial_world must be an object")
    if not isinstance(record["final_world"], dict):
        raise ValueError("record.final_world must be an object")
    artifact = record["operator_ticket_artifact"]
    if artifact is not None and not isinstance(artifact, dict):
        raise ValueError("record.operator_ticket_artifact must be an object or null")
    contract = scenario.get("measurement_contract")
    if not isinstance(contract, dict):
        raise ValueError("scenario.measurement_contract must be an object")
    required_contract_fields = {
        "schema_version",
        "impl_version",
        "contract_id",
        "metric_instances",
    }
    if set(contract) != required_contract_fields:
        raise ValueError("measurement_contract fields are invalid")
    if contract["schema_version"] != MEASUREMENT_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported measurement contract schema version")
    if contract["impl_version"] != MEASUREMENT_CONTRACT_IMPL_VERSION:
        raise ValueError("unsupported measurement contract implementation version")
    if not isinstance(contract["contract_id"], str) or not contract["contract_id"]:
        raise ValueError("measurement_contract.contract_id is invalid")
    instances = contract["metric_instances"]
    if not isinstance(instances, list):
        raise ValueError("measurement_contract.metric_instances must be an array")
    seen: set[str] = set()
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise ValueError(f"metric_instances[{index}] must be an object")
        if set(instance) != {"metric_instance_id", "metric_id", "contract"}:
            raise ValueError(f"metric_instances[{index}] fields are invalid")
        instance_id = instance["metric_instance_id"]
        metric_id = instance["metric_id"]
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"metric_instances[{index}].metric_instance_id is invalid")
        if instance_id in seen:
            raise ValueError(f"duplicate metric_instance_id: {instance_id}")
        if metric_id not in ACTIVE_METRIC_IDS:
            raise ValueError(f"metric_instances[{index}].metric_id is invalid")
        nullable_contract = metric_id in {"M04", "M05"} and instance["contract"] is None
        if not nullable_contract and not isinstance(instance["contract"], dict):
            raise ValueError(f"metric_instances[{index}].contract must be an object")
        seen.add(instance_id)
    return contract, instances


def _score_instance(
    *,
    instance: dict[str, Any],
    scenario: dict[str, Any],
    record: dict[str, Any],
    call_events: list[dict[str, Any]],
    context: dict[str, Any],
    extended_rows_by_id: dict[str, dict[str, Any]],
    frozen_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    metric_id = instance["metric_id"]
    if metric_id in EXTENDED_METRIC_IDS:
        return deepcopy(extended_rows_by_id[instance["metric_instance_id"]])
    if metric_id == "M01":
        return _resolution_metric_row(
            instance=instance,
            scenario=scenario,
            record=record,
            call_events=call_events,
            context=context,
            frozen_reference=frozen_reference,
        )
    if metric_id in TICKET_METRIC_IDS:
        return _ticket_metric_row(
            instance=instance,
            scenario=scenario,
            record=record,
            call_events=call_events,
            context=context,
        )
    if metric_id in {"M09", "M11"}:
        return _log_metric_row(
            instance=instance,
            scenario=scenario,
            record=record,
            call_events=call_events,
            context=context,
        )
    if metric_id in {"M04", "M05"}:
        return _core_metric_row(
            instance=instance,
            scenario=scenario,
            call_events=call_events,
        )
    return _not_measured_row(instance)


def _resolution_metric_row(
    *,
    instance: dict[str, Any],
    scenario: dict[str, Any],
    record: dict[str, Any],
    call_events: list[dict[str, Any]],
    context: dict[str, Any],
    frozen_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    if context["resolution_score"] is None:
        try:
            context["resolution_score"] = score_resolution(
                scenario_id=scenario["scenario_id"],
                record=record,
                call_events=call_events,
                frozen_reference=frozen_reference,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return _dependency_not_measured_row(
                instance=instance,
                reason="resolution_scorer_inputs_unavailable",
                dependency="record.final_world/private_frozen_reference/call_log",
                detail=str(exc),
            )
    score = context["resolution_score"]
    value = deepcopy(score["value"])
    status, reason = score["status"], score["reason"]
    # The right end state reached without ever opening the correct SOP is a guess, not a resolution.
    try:
        if context["m09_score"] is None:
            context["m09_score"] = calculate_m09(scenario=scenario, events=call_events)
        m09 = context["m09_score"]
        consulted = set(m09["expected_sop_path"]) <= set(m09["observed_successful_detail_path"])
    except (KeyError, TypeError, ValueError):
        consulted = None
    if isinstance(value, dict) and consulted is not None:
        value["gold_sop_consulted"] = consulted
    if status == "pass" and consulted is False:
        status, reason = "fail", "gold_sop_not_consulted"
    return _measured_row(
        instance=instance,
        status=status,
        value=value,
        reason=reason,
        violations=[],
        diagnostics={
            "source_scorer": score["schema_version"],
            "run_outcome": deepcopy(score["run_outcome"]),
            "final_world_sha256": score["final_world_sha256"],
            "expected_final_world_sha256": score["expected_final_world_sha256"],
            "final_world_root_keys": deepcopy(score["final_world_root_keys"]),
            "excluded_runtime_root_keys": deepcopy(
                score["excluded_runtime_root_keys"]
            ),
        },
    )


def _extended_metric_rows(
    *,
    scenario: dict[str, Any],
    record: dict[str, Any],
    instances: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    extended_instances = [
        instance
        for instance in instances
        if instance["metric_id"] in EXTENDED_METRIC_IDS
    ]
    if not extended_instances:
        return {}
    try:
        report = score_extended_metrics(
            scenario=scenario,
            artifacts=record.get(EXTENDED_ARTIFACT_RECORD_FIELD, {}),
        )
    except Exception as exc:  # noqa: BLE001 - dependency startup maps M17 to N/M.
        if is_japanese_parser_dependency_error(exc):
            dependency_rows: dict[str, dict[str, Any]] = {}
            for instance in extended_instances:
                if instance["metric_id"] == "M17":
                    row = _extended_dependency_nm_row(instance=instance)
                else:
                    row = _extended_contract_invalid_row(
                        instance=instance,
                        reason=str(exc),
                    )
                dependency_rows[instance["metric_instance_id"]] = row
            return dependency_rows
        if not isinstance(exc, ValueError):
            raise
        return {
            instance["metric_instance_id"]: _extended_contract_invalid_row(
                instance=instance,
                reason=str(exc),
            )
            for instance in extended_instances
        }
    return {
        row["metric_instance_id"]: row
        for row in report["metric_results"]
    }


def _extended_dependency_nm_row(*, instance: dict[str, Any]) -> dict[str, Any]:
    return _measured_row(
        instance=instance,
        status="N/M",
        value=None,
        reason="pinned_japanese_parser_dependency_unavailable",
        violations=[],
        diagnostics={"source_scorer": "extended_metric_scoring"},
    )


def _extended_contract_invalid_row(
    *,
    instance: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    violation = {
        "metric_instance_id": instance["metric_instance_id"],
        "metric_id": instance["metric_id"],
        "violation_type": "extended_metric_contract_invalid",
        "evidence": {"reason": reason},
    }
    return _measured_row(
        instance=instance,
        status="contract_invalid",
        value=None,
        reason=reason,
        violations=[violation],
        diagnostics={"source_scorer": "extended_metric_scoring"},
    )


def _core_metric_row(
    *,
    instance: dict[str, Any],
    scenario: dict[str, Any],
    call_events: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_id = instance["metric_id"]
    if metric_id == "M04":
        score = score_m04_customer_trigger_control(
            contract=scenario.get(M04_CONTRACT_FIELD),
            call_events=call_events,
        )
    elif metric_id == "M05":
        m05_contract = scenario.get(M05_CONTRACT_FIELD)
        if (
            m05_contract is None
            and _instance_declares_not_applicable(instance)
        ):
            return _measured_row(
                instance=instance,
                status="N/A",
                value=None,
                reason="m05_not_applicable_no_declared_obligation",
                violations=[],
                diagnostics={"source_scorer": "scoring_runner"},
            )
        score = score_m05_obligations(
            contract=m05_contract,
            call_events=call_events,
        )
    violations = _core_metric_violations(instance=instance, score=score)
    return _measured_row(
        instance=instance,
        status=score["status"],
        value=deepcopy(score["value"]),
        reason=score["reason"],
        violations=violations,
        diagnostics={"source_scorer": score["schema_version"]},
    )


def _instance_declares_not_applicable(instance: dict[str, Any]) -> bool:
    contract = instance.get("contract")
    applicability = None
    if isinstance(contract, dict):
        applicability = contract.get("scenario_applicability")
    return isinstance(applicability, dict) and applicability.get("applicable") is False


def _core_metric_violations(
    *,
    instance: dict[str, Any],
    score: dict[str, Any],
) -> list[dict[str, Any]]:
    if score["status"] not in {"fail", "contract_invalid"}:
        return []
    violation_type = "metric_check_failed"
    evidence: Any = deepcopy(score["value"])
    if score["status"] == "contract_invalid":
        violation_type = "metric_scoring_contract_invalid"
        evidence = {"reason": score["reason"]}
    return [
        {
            "metric_instance_id": instance["metric_instance_id"],
            "metric_id": instance["metric_id"],
            "violation_type": violation_type,
            "evidence": evidence,
        }
    ]


def _ticket_metric_row(
    *,
    instance: dict[str, Any],
    scenario: dict[str, Any],
    record: dict[str, Any],
    call_events: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    if context["ticket_score"] is None:
        contract = scenario.get("post_call_ticket_contract")
        if not isinstance(contract, dict):
            return _dependency_not_measured_row(
                instance=instance,
                reason=MISSING_TICKET_CONTRACT_REASON,
                dependency="scenario.post_call_ticket_contract",
            )
        try:
            context["ticket_score"] = score_post_call_ticket(
                contract=contract,
                artifact=record["operator_ticket_artifact"],
                initial_world=record["initial_world"],
                final_world=record["final_world"],
                call_events=call_events,
                run_outcome=derive_run_outcome(
                    record=record,
                    call_events=call_events,
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return _dependency_not_measured_row(
                instance=instance,
                reason="ticket_scorer_inputs_unavailable",
                dependency="ticket_artifact/world/call_log",
                detail=str(exc),
            )
    score = context["ticket_score"]
    metric_id = instance["metric_id"]
    value = deepcopy(score["metric_results"].get(metric_id, "N/M"))
    status = _status_from_value(value)
    violations: list[dict[str, Any]] = []
    if score["status"] == "contract_invalid":
        violations.append(
            {
                "metric_instance_id": instance["metric_instance_id"],
                "metric_id": metric_id,
                "violation_type": "ticket_scoring_contract_invalid",
                "evidence": {"reason": score.get("reason")},
            }
        )
    return _measured_row(
        instance=instance,
        status=status,
        value=value,
        reason=str(score.get("reason") or score["status"]),
        violations=violations,
        diagnostics={
            "source_scorer": score["schema_version"],
            "source_status": score["status"],
        },
    )


def _log_metric_row(
    *,
    instance: dict[str, Any],
    scenario: dict[str, Any],
    record: dict[str, Any],
    call_events: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    try:
        if instance["metric_id"] == "M09":
            if context["m09_score"] is None:
                context["m09_score"] = calculate_m09(
                    scenario=scenario,
                    events=call_events,
                )
            score = context["m09_score"]
            value = {
                "search_hit_correct": score["search_hit_correct"],
                "final_correct": score["final_correct"],
                "one_shot_correct": score["one_shot_correct"],
                "k": score["k"],
                "score_inverse_k": score["score_inverse_k"],
            }
            value["applicable_pass_rate"] = score["score_inverse_k"]
            diagnostics = {
                "source_scorer": score["calculation_version"],
                "source_result": deepcopy(score),
            }
            return _measured_row(
                instance=instance,
                status="measured",
                value=value,
                reason="inverse_opened_sop_count",
                violations=[],
                diagnostics=diagnostics,
            )
        if context["m11_score"] is None:
            context["m11_score"] = score_conversation_log_metrics(
                scenario,
                call_events,
                run_id=record["run_id"],
            )
        score = context["m11_score"]["M11"]
        facets = score["details"]["sub_facets"]
        value = {
            name: facet["passed"]
            for name, facet in facets.items()
        }
        value["applicable_pass_rate"] = score["value"]["applicable_pass_rate"]
        violations = [
            {
                "metric_instance_id": instance["metric_instance_id"],
                "metric_id": "M11",
                "violation_type": "tool_execution_item_failed",
                "evidence": {"item": name, "result": deepcopy(facet)},
            }
            for name, facet in facets.items()
            if facet["passed"] is False
        ]
        return _measured_row(
            instance=instance,
            status="measured",
            value=value,
            reason=str(score.get("reason") or "five_independent_tool_execution_results"),
            violations=violations,
            diagnostics={
                "source_scorer": context["m11_score"]["schema_version"],
                "source_result": deepcopy(score),
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        metric_id = instance["metric_id"]
        return _dependency_not_measured_row(
            instance=instance,
            reason=f"{metric_id.lower()}_scorer_inputs_unavailable",
            dependency="scenario/log/final_world",
            detail=str(exc),
        )


def _status_from_value(value: Any) -> str:
    if isinstance(value, str) and value in {"N/A", "N/M"}:
        return value
    if not isinstance(value, dict):
        return "measured"
    statuses = [
        item
        for item in value.values()
        if isinstance(item, str) and item in {"N/A", "N/M"}
    ]
    if "N/M" in statuses:
        return "N/M"
    if "N/A" in statuses:
        return "N/A"
    booleans = [item for item in value.values() if isinstance(item, bool)]
    if booleans:
        return "pass" if all(booleans) else "fail"
    return "measured"


def _measured_row(
    *,
    instance: dict[str, Any],
    status: str,
    value: Any,
    reason: str,
    violations: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "metric_instance_id": instance["metric_instance_id"],
        "metric_id": instance["metric_id"],
        "status": status,
        "value": value,
        "reason": reason,
        "violations": violations,
        "diagnostics": diagnostics,
    }


def _not_measured_row(instance: dict[str, Any]) -> dict[str, Any]:
    metric_id = instance["metric_id"]
    reason = UNMEASURED_REASON_BY_METRIC.get(
        metric_id,
        DEFAULT_UNMEASURED_REASON,
    )
    return _measured_row(
        instance=instance,
        status="N/A" if metric_id in TEXT_AUDIO_METRIC_IDS else "N/M",
        value=None,
        reason=reason,
        violations=[],
        diagnostics={
            "implemented": metric_id in TEXT_AUDIO_METRIC_IDS,
            "mode": "text" if metric_id in TEXT_AUDIO_METRIC_IDS else None,
        },
    )


def _dependency_not_measured_row(
    *,
    instance: dict[str, Any],
    reason: str,
    dependency: str,
    detail: str | None = None,
) -> dict[str, Any]:
    diagnostics = {
        "implemented": True,
        "missing_dependency": dependency,
    }
    if detail is not None:
        diagnostics["detail"] = detail
    return _measured_row(
        instance=instance,
        status="N/M",
        value=None,
        reason=reason,
        violations=[],
        diagnostics=diagnostics,
    )
