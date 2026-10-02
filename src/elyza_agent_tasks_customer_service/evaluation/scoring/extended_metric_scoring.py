"""Offline bridge for persisted M16/M17/M19 evidence.

``score_extended_metrics`` accepts decoded object values. M16 and M19 are
retained as contract-invalid placeholders; M17 remains scored from its
persisted parse bundle.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.observers.japanese_parse_adapter import is_japanese_parser_dependency_error
from elyza_agent_tasks_customer_service.evaluation.observers.japanese_rule_observer import (
    load_rule_catalog,
    score_japanese_register_from_parse,
    validate_register_context,
)


BRIDGE_SCHEMA_VERSION = "extended_metric_scoring"
EXTENDED_METRIC_IDS = frozenset(("M16", "M17", "M19"))
EXTENDED_ARTIFACT_KEYS = frozenset(("m17_parse_bundle",))
INTERACTION_METRIC_IDS = frozenset(("M16", "M19"))
INTERACTION_PLACEHOLDER_REASON = "extended_interaction_artifact_scoring_removed"


def score_extended_metrics(
    *,
    scenario: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Return M16/M19 placeholders and score M17 from persisted evidence.

    ``scenario`` must contain an object ``measurement_contract`` whose
    ``metric_instances`` is an object array, and ``artifacts`` must be an
    object containing only ``EXTENDED_ARTIFACT_KEYS``. Invalid roots raise
    ``ValueError``; invalid M17 inputs become per-metric ``contract_invalid``
    rows.
    """

    instances = _validate_bridge_inputs(scenario=scenario, artifacts=artifacts)
    rows: list[dict[str, Any]] = []
    for instance in instances:
        if instance["metric_id"] in INTERACTION_METRIC_IDS:
            row = _contract_invalid_row(instance, INTERACTION_PLACEHOLDER_REASON)
        else:
            row = _score_m17(instance=instance, scenario=scenario, artifacts=artifacts)
        rows.append(row)
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "scenario_id": scenario.get("scenario_id"),
        "metric_results": rows,
        "aggregation_policy": "single_overall_score_forbidden",
    }


def _validate_bridge_inputs(*, scenario: Any, artifacts: Any) -> list[dict[str, Any]]:
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be an object")
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be an object")
    unknown_artifacts = set(artifacts) - EXTENDED_ARTIFACT_KEYS
    if unknown_artifacts:
        raise ValueError(f"unknown extended artifacts: {sorted(unknown_artifacts)}")
    contract = scenario.get("measurement_contract")
    if not isinstance(contract, dict):
        raise ValueError("scenario.measurement_contract must be an object")
    raw_instances = contract.get("metric_instances")
    if not isinstance(raw_instances, list):
        raise ValueError("measurement_contract.metric_instances must be an array")
    instances: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, instance in enumerate(raw_instances):
        if not isinstance(instance, dict):
            raise ValueError(f"metric_instances[{index}] must be an object")
        metric_id = instance.get("metric_id")
        if metric_id not in EXTENDED_METRIC_IDS:
            continue
        instance_id = instance.get("metric_instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"metric_instances[{index}].metric_instance_id is invalid")
        if instance_id in seen:
            raise ValueError(f"duplicate extended metric_instance_id: {instance_id}")
        if not isinstance(instance.get("contract"), dict):
            raise ValueError(f"metric_instances[{index}].contract must be an object")
        instances.append(instance)
        seen.add(instance_id)
    return instances


def _score_m17(
    *,
    instance: dict[str, Any],
    scenario: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    try:
        context = validate_register_context(
            scenario.get("japanese_register_context")
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _contract_invalid_row(instance, str(exc))
    parse_bundle = artifacts.get("m17_parse_bundle")
    if parse_bundle is None:
        return _nm_row(instance, "persisted_m17_parse_bundle_missing")
    try:
        result = score_japanese_register_from_parse(
            parse_bundle=parse_bundle,
            context=context,
            rule_catalog=load_rule_catalog(),
        )
    except Exception as exc:  # noqa: BLE001 - dependency absence maps to N/M.
        if is_japanese_parser_dependency_error(exc):
            return _nm_row(instance, "pinned_japanese_parser_dependency_unavailable")
        return _contract_invalid_row(instance, str(exc))
    violations = [
        {
            "metric_instance_id": instance["metric_instance_id"],
            "metric_id": "M17",
            "violation_type": "confirmed_japanese_rule_violation",
            "evidence": deepcopy(row),
        }
        for row in result["rule_families"]
        if row["violation_count"] > 0
    ]
    violations.extend(
        {
            "metric_instance_id": instance["metric_instance_id"],
            "metric_id": "M17",
            "violation_type": "japanese_utterance_quality_violation",
            "evidence": deepcopy(row),
        }
        for row in result["utterance_assessments"]
        if row["violation_count"] > 0
    )
    return _result_row(
        instance=instance,
        status=result["status"],
        value=result,
        reason=result["reason"],
        violations=violations,
        diagnostics={
            "source_scorer": result["schema_version"],
            "parse_receipt_id": result["receipt"]["parse_receipt_id"],
        },
    )


def _result_row(
    *,
    instance: dict[str, Any],
    status: str,
    value: Any,
    reason: str,
    violations: list[dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "metric_instance_id": instance["metric_instance_id"],
        "metric_id": instance["metric_id"],
        "status": status,
        "value": value,
        "reason": reason,
        "violations": violations or [],
        "diagnostics": diagnostics or {},
    }


def _nm_row(instance: dict[str, Any], reason: str) -> dict[str, Any]:
    return _result_row(
        instance=instance,
        status="N/M",
        value=None,
        reason=reason,
        diagnostics={"offline_artifact_available": False},
    )


def _contract_invalid_row(
    instance: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    violation = {
        "metric_instance_id": instance["metric_instance_id"],
        "metric_id": instance["metric_id"],
        "violation_type": "extended_metric_contract_invalid",
        "evidence": {"reason": reason},
    }
    return _result_row(
        instance=instance,
        status="contract_invalid",
        value=None,
        reason=reason,
        violations=[violation],
        diagnostics={"bridge_schema_version": BRIDGE_SCHEMA_VERSION},
    )
