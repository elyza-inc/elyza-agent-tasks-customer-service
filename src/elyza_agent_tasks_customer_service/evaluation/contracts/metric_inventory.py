"""Validation for the reorganized metric inventory and human catalog."""

from __future__ import annotations

from elyza_agent_tasks_customer_service.evaluation.config_paths import config_path
from elyza_agent_tasks_customer_service.evaluation.contracts.run_outcome import HANDLING_TYPES
import json
from pathlib import Path
from typing import Any


INVENTORY_SCHEMA_VERSION = "metric_inventory"
CATALOG_VERSION = "metric_catalog_reorg_20260727"
ACTIVE_METRIC_IDS = (
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
    "M20",
    "M21",
    "M22",
    "M23",
    "M25",
    "M26",
    "M24",
)
RETIRED_METRIC_IDS = ("M02", "M03", "M06", "M08", "M10", "M12", "M13", "M18")
FORBIDDEN_SCORE_KEYS = {"overall_score", "single_overall_score"}
M16_M19_REQUIRED_SCENARIO_CONTRACTS = (
    "interaction_observation_contract",
    "interaction_closed_questions",
)
DEFAULT_INVENTORY_PATH = config_path("metric_inventory.json")


class MetricInventoryContractError(ValueError):
    """Raised when active and retired metric contracts are mixed."""


def load_metric_inventory(path: Path = DEFAULT_INVENTORY_PATH) -> dict[str, Any]:
    """Load an object-root JSON metric inventory.

    Only ``.json`` object roots are accepted. Objects may keep evaluator-
    derived rows in the object-row ``metric_row_extensions`` array; the loader
    appends them to ``metric_rows`` before validation. Missing files, malformed
    JSON, arrays/scalars, and invalid inventory contracts raise
    ``MetricInventoryContractError``.
    """

    if path.suffix.lower() != ".json":
        raise MetricInventoryContractError(f"metric inventory must be JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricInventoryContractError(f"cannot load metric inventory {path}: {exc}") from exc
    return validate_metric_inventory(value)


def validate_metric_inventory(value: Any) -> dict[str, Any]:
    """Validate exact active rows, retired mappings, gates, and mode contracts."""

    if not isinstance(value, dict):
        raise MetricInventoryContractError("metric inventory must be an object")
    value = _materialize_metric_row_extensions(value)
    required = {
        "schema_version",
        "catalog_version",
        "inventory_id",
        "single_overall_score",
        "metric_rows",
        "retired_metric_rows",
        "gates",
        "diagnostics",
        "measurement_modes",
    }
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise MetricInventoryContractError(
            f"metric inventory fields invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if value["schema_version"] != INVENTORY_SCHEMA_VERSION:
        raise MetricInventoryContractError("metric inventory schema version mismatch")
    if value["catalog_version"] != CATALOG_VERSION:
        raise MetricInventoryContractError("metric inventory catalog version mismatch")
    raw_rows = value["metric_rows"]
    retired = value["retired_metric_rows"]
    if (
        not isinstance(raw_rows, list)
        or any(not isinstance(row, dict) for row in raw_rows)
        or not isinstance(retired, list)
    ):
        raise MetricInventoryContractError("metric and retired rows must be arrays")
    known_ids = set(ACTIVE_METRIC_IDS) | set(RETIRED_METRIC_IDS)
    observed_ids = [row.get("metric_id") for row in raw_rows]
    observed_ids.extend(
        row.get("metric_id") for row in retired if isinstance(row, dict)
    )
    unknown_ids = [metric_id for metric_id in observed_ids if metric_id not in known_ids]
    if unknown_ids:
        raise MetricInventoryContractError(f"unknown metric IDs: {unknown_ids!r}")
    rows = [row for row in raw_rows if row.get("metric_id") in ACTIVE_METRIC_IDS]
    value["metric_rows"] = rows
    active_ids = [row.get("metric_id") for row in rows if isinstance(row, dict)]
    retired_ids = [row.get("metric_id") for row in retired if isinstance(row, dict)]
    if tuple(active_ids) != ACTIVE_METRIC_IDS:
        raise MetricInventoryContractError(
            f"active metric inventory mismatch: expected={ACTIVE_METRIC_IDS}, observed={tuple(active_ids)}"
        )
    if tuple(retired_ids) != RETIRED_METRIC_IDS:
        raise MetricInventoryContractError(
            f"retired metric inventory mismatch: expected={RETIRED_METRIC_IDS}, observed={tuple(retired_ids)}"
        )
    if set(active_ids) & set(retired_ids):
        raise MetricInventoryContractError("active and retired metric IDs overlap")
    overall_policy = value["single_overall_score"]
    if not isinstance(overall_policy, dict) or overall_policy.get("allowed") is not False:
        raise MetricInventoryContractError("single overall score must be forbidden")
    if set(overall_policy.get("forbidden_output_keys", [])) != FORBIDDEN_SCORE_KEYS:
        raise MetricInventoryContractError("forbidden overall score key inventory mismatch")
    retired_targets = {row["metric_id"]: row.get("absorbed_by") for row in retired}
    if any(not isinstance(target, str) or not target for target in retired_targets.values()):
        raise MetricInventoryContractError("every retired metric requires an explicit absorption target")
    m04 = next(row for row in rows if row["metric_id"] == "M04")
    required_edge = (m04.get("applicability") or {}).get("requires_edge")
    if required_edge != "customer_trigger_event_id -> critical_action_event_id":
        raise MetricInventoryContractError("M04 customer-trigger applicability edge mismatch")
    m01 = next(row for row in rows if row["metric_id"] == "M01")
    if tuple(m01.get("handling_types", [])) != HANDLING_TYPES:
        raise MetricInventoryContractError("M01 handling-type inventory mismatch")
    completion_contract = m01.get("completion_contract")
    if not isinstance(completion_contract, dict) or tuple(completion_contract) != HANDLING_TYPES:
        raise MetricInventoryContractError("M01 completion contract must cover all six handling types")
    if any(
        rule != "incomplete_is_not_resolution"
        and not rule.startswith("full_final_world_sha256")
        for rule in completion_contract.values()
    ):
        raise MetricInventoryContractError("M01 completion must use the full final-world hash")
    m01_shape = m01.get("result_shape")
    if not isinstance(m01_shape, dict) or set(m01_shape) != {
        "resolution_correct",
        "critical_violation_free",
    }:
        raise MetricInventoryContractError("M01 must report resolution and violations without a gate")
    m09 = next(row for row in rows if row["metric_id"] == "M09")
    m09_shape = m09.get("result_shape")
    if not isinstance(m09_shape, dict) or {
        "search_hit_correct",
        "final_correct",
        "one_shot_correct",
        "applicable_pass_rate",
        "search_hit_rate",
        "final_accuracy",
        "one_shot_rate",
    } - set(m09_shape):
        raise MetricInventoryContractError("M09 result shape is incomplete")
    m11 = next(row for row in rows if row["metric_id"] == "M11")
    if m11.get("measurement_epoch") != "post_hoc_reads_frozen_call_epoch_only":
        raise MetricInventoryContractError("M11 call-epoch cutoff mismatch")
    if set(m11.get("result_shape") or {}) != {
        "required_success",
        "argument_value",
        "argument_provenance",
        "dependency_order",
        "error_recovery",
        "applicable_pass_rate",
    }:
        raise MetricInventoryContractError("M11 result shape is incomplete")
    m16 = next(row for row in rows if row["metric_id"] == "M16")
    if m16.get("status") != "active" or set(m16.get("result_shape") or {}) != {
        "machine_observations",
        "closed_question_results",
        "fired_item_pass_rate",
    }:
        raise MetricInventoryContractError("M16 observation ledger is incomplete")
    if tuple(m16.get("required_scenario_contracts", ())) != (
        M16_M19_REQUIRED_SCENARIO_CONTRACTS
    ):
        raise MetricInventoryContractError("M16 scenario contract pair is incomplete")
    m17 = next(row for row in rows if row["metric_id"] == "M17")
    if m17.get("status") != "active" or set(m17.get("result_shape") or {}) != {
        "rule_applicable",
        "parser_measurable",
        "violation_count",
        "passed",
    }:
        raise MetricInventoryContractError("M17 fixed-rule result shape is incomplete")
    m24 = next(row for row in rows if row["metric_id"] == "M24")
    if m24.get("status") != "active" or set(m24.get("result_shape") or {}) != {
        "utterance_pass_ratio",
        "passed",
    }:
        raise MetricInventoryContractError("M24 spoken-style result shape is incomplete")
    m25 = next(row for row in rows if row["metric_id"] == "M25")
    if set(m25.get("result_shape") or {}) != {"accuracy", "observation_rate"}:
        raise MetricInventoryContractError("M25 accuracy and observation-rate shape is incomplete")
    m26 = next(row for row in rows if row["metric_id"] == "M26")
    if set(m26.get("result_shape") or {}) != {"grounded_use_rate"}:
        raise MetricInventoryContractError("M26 result shape is incomplete")
    m19 = next(row for row in rows if row["metric_id"] == "M19")
    if m19.get("status") != "active" or set(m19.get("result_shape") or {}) != {
        "machine_observations",
        "closed_question_results",
        "fired_item_pass_rate",
    }:
        raise MetricInventoryContractError("M19 observation ledger is incomplete")
    if tuple(m19.get("required_scenario_contracts", ())) != (
        M16_M19_REQUIRED_SCENARIO_CONTRACTS
    ):
        raise MetricInventoryContractError("M19 scenario contract pair is incomplete")
    m14 = next(row for row in rows if row["metric_id"] == "M14")
    if set((m14.get("result_shape") or {})) != {
        "ticket_fidelity_deterministic",
        "deterministic_field_results",
        "passed",
    }:
        raise MetricInventoryContractError("M14 deterministic ticket result shape mismatch")
    m15 = next(row for row in rows if row["metric_id"] == "M15")
    if set((m15.get("result_shape") or {})) != {"ticket_fidelity_conversational"}:
        raise MetricInventoryContractError("M15 conversational ticket result shape mismatch")
    if value["gates"] != []:
        raise MetricInventoryContractError("non-compensatory gates are forbidden")
    modes = value.get("measurement_modes")
    if not isinstance(modes, list) or len(modes) != 1 or modes[0].get("mode_id") != "passk_plan":
        raise MetricInventoryContractError("pass^k measurement mode is missing")
    return value


def _materialize_metric_row_extensions(value: dict[str, Any]) -> dict[str, Any]:
    """Merge the object-row extension array into ``metric_rows``.

    The accepted JSON object may contain ``metric_row_extensions`` as an
    array of metric row objects. Non-array extensions are left for the exact
    inventory-field validation to reject.
    """

    extensions = value.get("metric_row_extensions")
    if not isinstance(extensions, list):
        return value
    rows = value.get("metric_rows")
    if not isinstance(rows, list):
        return value
    materialized = dict(value)
    order = {metric_id: index for index, metric_id in enumerate(ACTIVE_METRIC_IDS)}
    materialized["metric_rows"] = [
        *rows,
        *sorted(extensions, key=lambda row: order.get(row.get("metric_id"), len(order))),
    ]
    del materialized["metric_row_extensions"]
    return materialized


def assert_no_single_overall_score(value: Any, *, path: str = "$") -> None:
    """Reject forbidden overall-score keys anywhere in a report object."""

    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_SCORE_KEYS:
                raise MetricInventoryContractError(f"single overall score key is forbidden: {path}.{key}")
            assert_no_single_overall_score(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_single_overall_score(item, path=f"{path}[{index}]")
