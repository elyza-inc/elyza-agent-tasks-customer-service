"""Score every package with perfect (gold) or zero-credit synthetic records.

The judge transport never makes a network request and its cache is temporary.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


REPORT_SCHEMA_VERSION_BY_MODE = {
    "perfect": "perfect_run_report",
    "zero": "zero_run_report",
}
MEASURED_FULL_FIELD = {
    "M09": "applicable_pass_rate",
    "M11": "applicable_pass_rate",
    "M16": "fired_item_pass_rate",
}
CAUSE_FIELD_KEYS = (
    "tool",
    "field",
    "expected",
    "observed",
    "derived_source_kind",
    "source_seq",
    "source_tool",
    "source_ref",
    "source_path",
    "call_seq",
)

from elyza_agent_tasks_customer_service.evaluation.engine.package_adapter import (
    load_package,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.package_scoring import (
    score_package_record,
)
from tests.helpers.gold_record_builder import (
    JUDGE_CONFIG,
    perfect_judge_transport,
    perfect_record,
    zero_record,
)


def metric_is_full(row: dict[str, Any]) -> bool:
    """Return whether one decoded text metric row is at its applicable ceiling."""

    status = row.get("status")
    if status in {"pass", "N/A"}:
        return True
    field = MEASURED_FULL_FIELD.get(row.get("metric_id"))
    value = row.get("value")
    return (
        status == "measured"
        and field is not None
        and isinstance(value, dict)
        and value.get(field) == 1.0
    )


def metric_has_credit(row: dict[str, Any]) -> bool:
    """Return whether one zero-run metric grants a pass or positive measured rate."""

    if row.get("status") == "pass":
        return True
    field = MEASURED_FULL_FIELD.get(row.get("metric_id"))
    value = row.get("value")
    measured = None
    if field is not None and isinstance(value, dict):
        measured = value.get(field)
    return (
        row.get("status") == "measured"
        and isinstance(measured, int | float)
        and not isinstance(measured, bool)
        and measured > 0
    )


def _m11_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    source = row.get("diagnostics", {}).get("source_result", {})
    facets = source.get("details", {}).get("sub_facets", {})
    failures = []
    for name, facet in facets.items():
        if not isinstance(facet, dict) or facet.get("passed") is not False:
            continue
        fields = facet.get("details", {}).get("fields", [])
        causes = [
            {key: deepcopy(field[key]) for key in CAUSE_FIELD_KEYS if key in field}
            for field in fields
            if isinstance(field, dict)
            and field.get("required")
            and (
                not field.get("matched_on_success")
                or field.get("derived_source_kind") == "unknown"
            )
        ]
        failures.append(
            {
                "sub_item": name,
                "reason": facet.get("reason"),
                "cause_fields": causes,
            }
        )
    return failures


def _m04_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "sub_item": edge.get("edge_id"),
            "reason": "required_precondition_missing_or_late",
            "cause_fields": [
                {
                    "field": "critical_action_tool_id",
                    "value": edge.get("critical_action_tool_id"),
                },
                {
                    "field": "precondition_failures",
                    "value": deepcopy(edge.get("precondition_failures", [])),
                },
            ],
        }
        for edge in row.get("value", {}).get("edges", [])
        if isinstance(edge, dict) and edge.get("passed") is False
    ]


def _utterance_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    failures = [
        {
            "sub_item": utterance.get("turn_ref"),
            "reason": row.get("reason"),
            "cause_fields": [
                {"field": "text", "value": utterance.get("text")},
                {
                    "field": "violations",
                    "value": deepcopy(
                        utterance.get("violations")
                        or utterance.get("japanese_correctness", {}).get("violations", [])
                        + utterance.get("polite_register", {}).get("violations", [])
                    ),
                },
            ],
        }
        for utterance in row.get("value", {}).get("utterance_assessments", [])
        if isinstance(utterance, dict) and utterance.get("passed") is False
    ]
    failures.extend(
        {
            "sub_item": family.get("rule_family"),
            "reason": row.get("reason"),
            "cause_fields": [
                {
                    "field": "violations",
                    "value": deepcopy(family.get("violations", [])),
                }
            ],
        }
        for family in row.get("value", {}).get("rule_families", [])
        if isinstance(family, dict) and family.get("passed") is False
    )
    return failures


def metric_failure_details(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact sub-item and cause-field failures for one metric row."""

    metric_id = row.get("metric_id")
    if metric_id == "M11":
        details = _m11_failures(row)
    elif metric_id == "M04":
        details = _m04_failures(row)
    elif metric_id in {"M17", "M24"}:
        details = _utterance_failures(row)
    else:
        details = []
    if details:
        return details
    return [
        {
            "sub_item": None,
            "reason": row.get("reason"),
            "cause_fields": [{"field": "value", "value": deepcopy(row.get("value"))}],
        }
    ]


def build_perfect_run_report(
    packages_dir: Path,
    *,
    mode: str = "perfect",
) -> dict[str, Any]:
    """Score every v3 package YAML in ``packages_dir`` for one sweep mode.

    The directory must exist and contain object-root package YAML files. Both
    ``.yaml`` and ``.yml`` are accepted. ``mode`` is ``perfect`` or ``zero``;
    malformed, unsupported, or unscorable inputs raise ``ValueError``.
    """

    if mode not in REPORT_SCHEMA_VERSION_BY_MODE:
        raise ValueError(f"unsupported sweep mode: {mode}")
    if mode == "perfect":
        scenario_status_field = "all_applicable_metrics_full"
    else:
        scenario_status_field = "all_metrics_no_credit"
    if not packages_dir.is_dir():
        raise ValueError(f"packages directory is not a directory: {packages_dir}")
    package_paths = sorted((*packages_dir.glob("*.yaml"), *packages_dir.glob("*.yml")))
    if not package_paths:
        raise ValueError(f"no package YAML files found: {packages_dir}")
    os.environ.setdefault(JUDGE_CONFIG["judge"]["api_key_env"], "offline-stub")
    scenario_rows = []
    metric_counts: dict[str, dict[str, int]] = {}
    failures = []
    with TemporaryDirectory(prefix="perfect-run-judge-cache-") as cache:
        cache_dir = Path(cache)
        for package_path in package_paths:
            package = load_package(package_path)
            if mode == "perfect":
                record = perfect_record(package)
            else:
                record = zero_record(package)
            report = score_package_record(
                package=package,
                record=record,
                judge_config=JUDGE_CONFIG,
                cache_dir=cache_dir,
                transport=perfect_judge_transport,
                mode="text",
            )
            scenario_full = True
            for row in report["metric_results"]:
                metric_id = row["metric_id"]
                if mode == "zero":
                    counts = metric_counts.setdefault(
                        metric_id,
                        {"no_credit": 0, "credited": 0, "not_applicable": 0},
                    )
                    if row["status"] == "N/A":
                        counts["not_applicable"] += 1
                    elif metric_has_credit(row):
                        counts["credited"] += 1
                        scenario_full = False
                        failures.append(
                            {
                                "scenario_id": package["scenario_id"],
                                "metric_id": metric_id,
                                "status": row["status"],
                                "reason": row.get("reason"),
                                "value": deepcopy(row.get("value")),
                            }
                        )
                    else:
                        counts["no_credit"] += 1
                    continue
                counts = metric_counts.setdefault(
                    metric_id, {"full": 0, "not_full": 0, "not_applicable": 0}
                )
                if row["status"] == "N/A":
                    counts["not_applicable"] += 1
                    continue
                if metric_is_full(row):
                    counts["full"] += 1
                    continue
                counts["not_full"] += 1
                scenario_full = False
                for detail in metric_failure_details(row):
                    failures.append(
                        {
                            "scenario_id": package["scenario_id"],
                            "metric_id": metric_id,
                            "status": row["status"],
                            **detail,
                        }
                    )
            scenario_rows.append(
                {
                    "scenario_id": package["scenario_id"],
                    scenario_status_field: scenario_full,
                }
            )
    if mode == "zero":
        return {
            "schema_version": REPORT_SCHEMA_VERSION_BY_MODE[mode],
            "mode": mode,
            "packages_dir": str(packages_dir.resolve()),
            "scenario_count": len(scenario_rows),
            "all_metrics_no_credit_scenario_count": sum(
                row["all_metrics_no_credit"] for row in scenario_rows
            ),
            "violation_scenario_count": sum(
                not row["all_metrics_no_credit"] for row in scenario_rows
            ),
            "violation_count": len(failures),
            "metric_summary": metric_counts,
            "scenarios": scenario_rows,
            "violations": failures,
            "judge": {"transport": "offline_stub", "cache": "temporary"},
        }
    return {
        "schema_version": REPORT_SCHEMA_VERSION_BY_MODE[mode],
        "packages_dir": str(packages_dir.resolve()),
        "scenario_count": len(scenario_rows),
        "all_metric_full_scenario_count": sum(
            row["all_applicable_metrics_full"] for row in scenario_rows
        ),
        "non_full_scenario_count": sum(
            not row["all_applicable_metrics_full"] for row in scenario_rows
        ),
        "metric_summary": metric_counts,
        "scenarios": scenario_rows,
        "failures": failures,
        "judge": {"transport": "offline_stub", "cache": "temporary"},
    }

