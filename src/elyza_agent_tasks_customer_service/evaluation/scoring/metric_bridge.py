"""Bridge completed runs to the metric scorers and derive M24 from M17.

The public functions accept already-decoded object-root scenario, record, and
score-report mappings plus an object-row event array.  Malformed metric rows,
duplicate active metric IDs, and missing active metric rows raise
``ValueError``. M24 is derived from M17's persisted spoken-style assessments.
Audio-only metrics are omitted from text results and the availability ledger.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.contracts.metric_inventory import ACTIVE_METRIC_IDS, assert_no_single_overall_score
from elyza_agent_tasks_customer_service.evaluation.scoring.scoring_runner import score_record


BRIDGE_SCHEMA_VERSION = "metric_bridge"
AVAILABILITY_SCHEMA_VERSION = "measurement_availability"
M24_RESULT_SCHEMA_VERSION = "spoken_style_compliance"
TEXT_EXCLUDED_METRIC_IDS = frozenset(("M20", "M21", "M22", "M23", "M25", "M26"))
UNMEASURABLE_RESULT_STATUSES = frozenset(("N/M", "contract_invalid", "pending"))


def score_metric_inventory(
    *,
    scenario: dict[str, Any],
    record: dict[str, Any],
    event_log: list[dict[str, Any]],
    frozen_reference: dict[str, Any] | None = None,
    include_audio: bool = False,
) -> dict[str, Any]:
    """Score one completed run with ``score_record``.

    ``scenario`` and ``record`` must be object-root mappings accepted by
    ``score_record``. ``event_log`` must be an array of object rows.  Unless
    ``include_audio`` is set, audio metric rows are dropped. The returned
    report adds the derived M24 result; callers build the
    measurement-availability ledger once their metric rows are final.
    """

    report = score_record(
        scenario=scenario,
        record=record,
        event_log=event_log,
        frozen_reference=frozen_reference,
    )
    if not include_audio:
        report["metric_results"] = [
            row
            for row in report["metric_results"]
            if row.get("metric_id") not in TEXT_EXCLUDED_METRIC_IDS
        ]
        report["violations"] = [
            row
            for row in report["violations"]
            if row.get("metric_id") not in TEXT_EXCLUDED_METRIC_IDS
        ]
        report["not_measured"] = [
            row
            for row in report["not_measured"]
            if row.get("metric_id") not in TEXT_EXCLUDED_METRIC_IDS
        ]
    _attach_m24_result(report)
    report["diagnostics"]["metric_bridge_version"] = BRIDGE_SCHEMA_VERSION
    assert_no_single_overall_score(report)
    return report


def _attach_m24_result(report: dict[str, Any]) -> None:
    """Add or replace M24 using the spoken-style evidence retained by M17."""

    m17 = next(
        (row for row in report["metric_results"] if row.get("metric_id") == "M17"),
        None,
    )
    if m17 is None:
        raise ValueError("M17 result is required to derive M24")
    m24 = _m24_result_from_m17(m17)
    report["metric_results"] = [
        row for row in report["metric_results"] if row.get("metric_id") != "M24"
    ]
    report["metric_results"].append(m24)
    report["violations"] = [
        row for row in report["violations"] if row.get("metric_id") != "M24"
    ]
    report["violations"].extend(deepcopy(m24["violations"]))
    report["not_measured"] = [
        row for row in report["not_measured"] if row.get("metric_id") != "M24"
    ]
    if m24["status"] == "N/M":
        report["not_measured"].append(
            {
                "metric_instance_id": m24["metric_instance_id"],
                "metric_id": "M24",
                "reason": m24["reason"],
            }
        )
    report["diagnostics"]["bridge_derived_metric_ids"] = ["M24"]


def _m24_result_from_m17(m17: dict[str, Any]) -> dict[str, Any]:
    instance_id = f"{m17['metric_instance_id']}:derived:M24"
    m17_value = m17.get("value")
    if isinstance(m17_value, dict):
        utterances = m17_value.get("utterance_assessments")
    else:
        utterances = None
    if not isinstance(utterances, list):
        status = m17.get("status")
        if status not in UNMEASURABLE_RESULT_STATUSES:
            status = "contract_invalid"
        return {
            "metric_instance_id": instance_id,
            "metric_id": "M24",
            "status": status,
            "value": {
                "schema_version": M24_RESULT_SCHEMA_VERSION,
                "metric_id": "M24",
                "status": status,
                "passed": None,
                "reason": "m17_utterance_assessments_unavailable",
                "utterance_assessments": [],
                "utterance_count": 0,
                "utterance_pass_count": 0,
                "utterance_pass_ratio": None,
                "violation_count": 0,
            },
            "reason": "m17_utterance_assessments_unavailable",
            "violations": [],
            "diagnostics": {
                "source_metric_id": "M17",
                "source_scorer": m17.get("diagnostics", {}).get("source_scorer"),
                "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
            },
        }
    assessments = []
    violations = []
    for row in utterances:
        if isinstance(row, dict):
            spoken_style = row.get("spoken_style")
        else:
            spoken_style = None
        if not isinstance(spoken_style, dict):
            raise ValueError("M17 spoken_style assessment is missing")
        style_violations = spoken_style.get("violations")
        if not isinstance(style_violations, list):
            raise ValueError("M17 spoken_style violations must be an array")
        assessment = {
            "turn_ref": row.get("turn_ref"),
            "text": row.get("text"),
            "passed": not style_violations,
            "violation_count": len(style_violations),
            "violations": deepcopy(style_violations),
        }
        assessments.append(assessment)
        if style_violations:
            violations.append(
                {
                    "metric_instance_id": instance_id,
                    "metric_id": "M24",
                    "violation_type": "spoken_style_violation",
                    "evidence": deepcopy(assessment),
                }
            )
    utterance_count = len(assessments)
    pass_count = sum(row["passed"] for row in assessments)
    ratio = pass_count / utterance_count if utterance_count else None
    if not utterance_count:
        status = "N/M"
        passed = None
        reason = "operator_utterances_unavailable"
    elif violations:
        status = "fail"
        passed = False
        reason = "written_style_markup_detected"
    else:
        status = "pass"
        passed = True
        reason = "all_operator_utterances_spoken_style_compliant"
    value = {
        "schema_version": M24_RESULT_SCHEMA_VERSION,
        "metric_id": "M24",
        "status": status,
        "passed": passed,
        "reason": reason,
        "scope": "all_operator_utterances_without_written_style_markup",
        "utterance_assessments": assessments,
        "utterance_count": utterance_count,
        "utterance_pass_count": pass_count,
        "utterance_pass_ratio": ratio,
        "violation_count": sum(row["violation_count"] for row in assessments),
    }
    return {
        "metric_instance_id": instance_id,
        "metric_id": "M24",
        "status": status,
        "value": value,
        "reason": reason,
        "violations": violations,
        "diagnostics": {
            "source_metric_id": "M17",
            "source_scorer": m17.get("diagnostics", {}).get("source_scorer"),
            "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
        },
    }


def build_measurement_availability(
    metric_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return ordered availability rows for text-mode metric IDs.

    ``metric_results`` must contain exactly one object row for every text
    metric ID. Audio-only rows are ignored; ``N/M``, ``pending``, and
    ``contract_invalid`` become ``not_measurable``;
    scorer ``N/A`` becomes ``not_applicable``; every other result status is
    ``measured``. Unknown, missing, or duplicate metric IDs raise
    ``ValueError``.
    """

    if not isinstance(metric_results, list) or any(
        not isinstance(row, dict) for row in metric_results
    ):
        raise ValueError("metric_results must be an object array")
    by_metric: dict[str, dict[str, Any]] = {}
    for row in metric_results:
        metric_id = row.get("metric_id")
        if metric_id not in ACTIVE_METRIC_IDS:
            raise ValueError(f"unknown active metric result: {metric_id}")
        if metric_id in by_metric:
            raise ValueError(f"duplicate active metric result: {metric_id}")
        by_metric[metric_id] = row
    text_metric_ids = tuple(
        metric_id
        for metric_id in ACTIVE_METRIC_IDS
        if metric_id not in TEXT_EXCLUDED_METRIC_IDS
    )
    missing = set(text_metric_ids) - set(by_metric)
    if missing:
        raise ValueError(f"active metric results missing: {sorted(missing)}")

    rows = [
        _availability_row(metric_id, by_metric[metric_id])
        for metric_id in text_metric_ids
    ]
    summary = {
        status: sum(row["availability"] == status for row in rows)
        for status in ("measured", "not_measurable", "not_applicable")
    }
    return {
        "schema_version": AVAILABILITY_SCHEMA_VERSION,
        "evaluation_mode": "text",
        "metrics": rows,
        "summary": summary,
        "not_measured": [
            {
                "metric_id": row["metric_id"],
                "status": "N/M",
                "reason": row["reason"],
            }
            for row in rows
            if row["availability"] == "not_measurable"
        ],
    }


def _availability_row(
    metric_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    result_status = result.get("status")
    reason = str(result.get("reason") or result_status or "result_status_missing")
    if result_status in UNMEASURABLE_RESULT_STATUSES:
        availability = "not_measurable"
    elif result_status == "N/A":
        availability = "not_applicable"
    else:
        availability = "measured"
    row = {
        "metric_id": metric_id,
        "availability": availability,
        "result_status": result_status,
        "reason": reason,
    }
    if metric_id == "M17" and availability == "not_measurable":
        blockers = _m17_blocking_families(result.get("value"))
        if blockers:
            row["blocking_rule_families"] = blockers
    return row


def _m17_blocking_families(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("rule_families"), list):
        return []
    return [
        {
            "rule_family": family.get("rule_family"),
            "rule_applicable": family.get("rule_applicable"),
            "parser_measurable": family.get("parser_measurable"),
        }
        for family in value["rule_families"]
        if isinstance(family, dict) and family.get("passed") is not True
    ]
