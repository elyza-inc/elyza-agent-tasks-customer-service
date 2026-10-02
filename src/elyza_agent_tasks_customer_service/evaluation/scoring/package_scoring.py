#!/usr/bin/env python3
"""Score one persisted package run through the complete text metric route.

The CLI accepts object-root package YAML, record JSON, and a fixed judge config
JSON/YAML. The record must contain an object-array ``event_log`` plus the
fields required by the core and conversation scorers. Malformed, missing, or
judge-incompatible inputs raise; no metric answer is substituted.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.engine.package_adapter import (
    TEXT_MODE_EXCLUDED_METRIC_IDS,
    convert_package,
    frozen_reference_from_package,
    load_package,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.metric_bridge import (
    build_measurement_availability,
    score_metric_inventory,
)
from elyza_agent_tasks_customer_service.evaluation.observers.japanese_parse_adapter import parse_japanese_turns_or_nm
from elyza_agent_tasks_customer_service.evaluation.llm.m05_consent_judge import judge_m05_consent
from elyza_agent_tasks_customer_service.evaluation.contracts.metric_inventory import (
    ACTIVE_METRIC_IDS,
    assert_no_single_overall_score,
)
from elyza_agent_tasks_customer_service.evaluation.contracts.variants import normalize_variant
from elyza_agent_tasks_customer_service.evaluation.scoring.post_call_ticket_scoring import score_post_call_ticket
from elyza_agent_tasks_customer_service.evaluation.contracts.run_outcome import derive_run_outcome
from elyza_agent_tasks_customer_service.evaluation.scoring.conversation_log_scoring import (
    load_judge_config,
    load_mapping,
    score_conversation,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


REPORT_VERSION = "package_text_scoring"
JUDGE_METRIC_IDS = frozenset(("M15", "M16", "M19"))
EXTENDED_ARTIFACT_FIELD = "extended_metric_artifacts"
FAILED_RUNTIME_STATUS = "failed_runtime"
FAILED_RUNTIME_TEXT_METRIC_IDS = tuple(
    metric_id
    for metric_id in ACTIVE_METRIC_IDS
    if metric_id not in TEXT_MODE_EXCLUDED_METRIC_IDS
)
FAILED_RUNTIME_OPTIONAL_CONTRACT_FIELDS = {
    "M04": "m04_customer_trigger_contract",
    "M05": "m05_obligation_contract",
}
PERSONA_TRIGGER_FIELDS = (
    "customer_pressure",
    "customer_misconception",
    "customer_correction",
    "customer_emotion",
)
FAILED_RUNTIME_VALUE_BY_METRIC = {
    "M01": {"resolution_correct": "N/M", "critical_violation_free": "N/M"},
    "M04": {"passed": "N/M"},
    "M05": {"passed": "N/M"},
    "M09": {
        "search_hit_correct": "N/M",
        "final_correct": "N/M",
        "one_shot_correct": "N/M",
        "k": None,
        "score_inverse_k": None,
        "applicable_pass_rate": None,
    },
    "M11": {
        "required_success": "N/M",
        "argument_value": "N/M",
        "argument_provenance": "N/M",
        "dependency_order": "N/M",
        "error_recovery": "N/M",
        "applicable_pass_rate": None,
    },
    "M14": {
        "ticket_fidelity_deterministic": "N/M",
        "deterministic_field_results": {
            "handling_type": "N/M",
            "target_ids": "N/M",
            "performed_actions": "N/M",
            "evidence_refs": "N/M",
        },
        "passed": "N/M",
    },
    "M15": {"ticket_fidelity_conversational": "N/M"},
    "M16": {
        "machine_observations": [
            {"criterion_id": "M16-M1", "status": "N/M", "passed": None},
            {"criterion_id": "M16-M2", "status": "N/M", "passed": None},
            {"criterion_id": "M16-M4", "status": "N/M", "passed": None},
            {"criterion_id": "M16-M5", "status": "N/M", "passed": None},
        ],
        "closed_question_results": [],
        "fired_item_pass_rate": None,
    },
    "M17": {
        "rule_applicable": "N/M",
        "parser_measurable": "N/M",
        "violation_count": None,
        "passed": "N/M",
        "utterance_pass_ratio": "N/M",
    },
    "M19": {
        "machine_observations": [],
        "closed_question_results": [],
        "fired_item_pass_rate": None,
    },
    "M24": {"passed": "N/M", "utterance_pass_ratio": "N/M"},
}


def score_package_record(
    *,
    package: dict[str, Any],
    record: dict[str, Any],
    judge_config: dict[str, Any],
    cache_dir: Path,
    transport: Any = None,
    mode: str = "text",
) -> dict[str, Any]:
    """Return the common metric report for one persisted package run.

    ``package`` and ``record`` are decoded object roots. A
    ``status=failed_runtime`` record requires a non-empty ``error_message`` and
    returns applicable text metrics as ``N/M`` and inapplicable metrics as
    ``N/A`` without invoking a judge. Other records require a complete
    object-array ``record.event_log``. ``judge_config`` must satisfy
    ``score_conversation``. Parser unavailability is retained as M17/M24
    ``N/M``; every other malformed or missing input raises.
    """

    if not isinstance(package, dict) or not isinstance(record, dict):
        raise ValueError("package and record must be objects")
    scenario = convert_package(
        package,
        mode=mode,
        variant_id=record.get("variant_id"),
    )
    if record.get("status") == FAILED_RUNTIME_STATUS:
        return _score_failed_runtime(scenario=scenario, record=record, mode=mode)
    event_log = record.get("event_log")
    if not isinstance(event_log, list) or any(not isinstance(row, dict) for row in event_log):
        raise ValueError("record.event_log must be a complete object array")
    scoring_record = _with_runtime_artifacts(record, event_log)
    core = score_metric_inventory(
        scenario=scenario,
        record=scoring_record,
        event_log=event_log,
        frozen_reference=frozen_reference_from_package(package),
        include_audio=mode != "text",
    )
    call_events = [row for row in event_log if row.get("epoch", "call") == "call"]
    ticket_score = score_post_call_ticket(
        contract=package["post_call_ticket_contract"],
        artifact=record.get("operator_ticket_artifact"),
        initial_world=record["initial_world"],
        final_world=record["final_world"],
        call_events=call_events,
        run_outcome=derive_run_outcome(record=record, call_events=call_events),
    )
    _merge_ticket_row(core, ticket_score)
    judge_kwargs = {
        "package": package,
        "record": record,
        "config": judge_config,
        "cache_dir": cache_dir,
    }
    if transport is not None:
        judge_kwargs["transport"] = transport
    judged = score_conversation(**judge_kwargs)
    _merge_judge_rows(core, judged)
    _merge_m05_judge_row(core, judged)
    consent_kwargs = {
        "package": package,
        "record": record,
        "config": judge_config,
        "cache_dir": cache_dir,
    }
    if transport is not None:
        consent_kwargs["transport"] = transport
    consent = judge_m05_consent(**consent_kwargs)
    _merge_m05_consent(core, consent)
    if mode != "text":
        _audio_m24_not_applicable(core)
    core["schema_version"] = REPORT_VERSION
    if "variant_id" in scenario:
        core["variant_id"] = scenario["variant_id"]
    core["judge"] = deepcopy(judged["judge"])
    core["m05_consent_judge"] = deepcopy(consent["judge"])
    core["measurement_availability"] = build_measurement_availability(
        core["metric_results"]
    )
    core["not_measured"] = _not_measured_rows(core["metric_results"])
    assert_no_single_overall_score(core)
    return core


def _not_measured_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "metric_instance_id": row["metric_instance_id"],
            "metric_id": row["metric_id"],
            "reason": row["reason"],
        }
        for row in rows
        if row["status"] == "N/M"
    ]


def _audio_m24_not_applicable(core: dict[str, Any]) -> None:
    """M24 checks written-style markup; an audio operator speaks, so its transcript format is not its choice."""

    for row in core["metric_results"]:
        if row.get("metric_id") == "M24":
            row.update({"status": "N/A", "reason": "audio_output_is_speech", "value": None})
    core["violations"] = [row for row in core.get("violations", []) if row.get("metric_id") != "M24"]


def _score_failed_runtime(
    *, scenario: dict[str, Any], record: dict[str, Any], mode: str
) -> dict[str, Any]:
    error_message = record.get("error_message")
    if not isinstance(error_message, str) or not error_message.strip():
        raise ValueError("failed_runtime record.error_message must be a non-empty string")
    contract = scenario["measurement_contract"]
    instances = contract["metric_instances"]
    by_metric = {row["metric_id"]: row for row in instances}
    metric_ids = ACTIVE_METRIC_IDS
    if mode == "text":
        metric_ids = FAILED_RUNTIME_TEXT_METRIC_IDS
    if tuple(by_metric) != metric_ids:
        raise ValueError("failed_runtime metric inventory is invalid")
    reason = f"run_failed: {error_message}"
    rows = []
    for metric_id in metric_ids:
        applicable, applicability_reason = _failed_runtime_applicability(
            metric_id=metric_id,
            scenario=scenario,
            record=record,
            mode=mode,
        )
        status = "N/A"
        value = None
        row_reason = applicability_reason
        if applicable:
            status = "N/M"
            value = deepcopy(
                FAILED_RUNTIME_VALUE_BY_METRIC.get(metric_id, {"passed": "N/M"})
            )
            row_reason = reason
        rows.append({
            "metric_instance_id": by_metric[metric_id]["metric_instance_id"],
            "metric_id": metric_id,
            "status": status,
            "value": value,
            "reason": row_reason,
            "violations": [],
            "diagnostics": {
                "source_scorer": REPORT_VERSION,
                "run_status": FAILED_RUNTIME_STATUS,
            },
        })
    report = {
        "schema_version": REPORT_VERSION,
        "scenario_id": scenario["scenario_id"],
        # 早期失敗の partial record は run_id を持たないことがある
        "run_id": record.get("run_id") or f"{scenario['scenario_id']}-failed",
        "measurement_contract_id": contract["contract_id"],
        "metric_results": rows,
        "violations": [],
        "not_measured": _not_measured_rows(rows),
        "diagnostics": {
            "run_status": FAILED_RUNTIME_STATUS,
            "error_type": record.get("error_type"),
            "error_message": error_message,
        },
        "aggregation_policy": "single_overall_score_forbidden",
    }
    if "variant_id" in scenario:
        report["variant_id"] = scenario["variant_id"]
    report["measurement_availability"] = build_measurement_availability(rows)
    assert_no_single_overall_score(report)
    return report


def _failed_runtime_applicability(
    *, metric_id: str, scenario: dict[str, Any], record: dict[str, Any], mode: str
) -> tuple[bool, str | None]:
    if mode == "text" and metric_id in {"M25", "M26"}:
        return False, "audio_only_metric"
    if mode != "text" and metric_id == "M24":
        # Same as completed audio runs (_audio_m24_not_applicable): speech has no written markup.
        return False, "audio_output_is_speech"
    contract_field = FAILED_RUNTIME_OPTIONAL_CONTRACT_FIELDS.get(metric_id)
    if contract_field is not None and scenario.get(contract_field) is None:
        return False, f"{metric_id.lower()}_contract_not_declared"
    if metric_id != "M19":
        return True, None
    try:
        variant_id = normalize_variant(record.get("variant_id"))
    except ValueError as exc:
        raise ValueError(f"failed_runtime record.{exc}") from exc
    if variant_id == "baseline":
        return False, "m19_baseline_not_applicable"
    persona = scenario.get("persona")
    persona_defined = isinstance(persona, dict) and any(
        isinstance(persona.get(field), dict) for field in PERSONA_TRIGGER_FIELDS
    )
    if not persona_defined:
        return False, "m19_trigger_persona_not_declared"
    return True, None


def _with_runtime_artifacts(
    record: dict[str, Any], event_log: list[dict[str, Any]]
) -> dict[str, Any]:
    result = deepcopy(record)
    artifacts = deepcopy(result.get(EXTENDED_ARTIFACT_FIELD, {}))
    if not isinstance(artifacts, dict):
        raise ValueError(f"record.{EXTENDED_ARTIFACT_FIELD} must be an object")
    if "m17_parse_bundle" not in artifacts:
        turns = [
            {"turn_ref": row["event_ref"], "text": row["content"]}
            for row in event_log
            if row.get("epoch", "call") == "call"
            and row.get("event_type") == "message"
            and row.get("actor") in {"operator", "assistant", "agent"}
            and isinstance(row.get("event_ref"), str)
            and isinstance(row.get("content"), str)
        ]
        attempt = parse_japanese_turns_or_nm(turns=turns)
        bundle = attempt.get("parse_bundle")
        if isinstance(bundle, dict):
            artifacts["m17_parse_bundle"] = bundle
        else:
            result["extended_metric_artifact_diagnostics"] = {
                "m17_parse_attempt": attempt
            }
    if artifacts:
        result[EXTENDED_ARTIFACT_FIELD] = artifacts
    return result


def _merge_judge_rows(core: dict[str, Any], judged: dict[str, Any]) -> None:
    instances = {
        row["metric_id"]: row["metric_instance_id"]
        for row in core["metric_results"]
        if row.get("metric_id") in JUDGE_METRIC_IDS
    }
    rows = {
        row["metric_id"]: {
            "metric_instance_id": instances[row["metric_id"]],
            "metric_id": row["metric_id"],
            "status": row["status"],
            "value": {
                key: deepcopy(value)
                for key, value in row.items()
                if key not in {"metric_id", "status", "reason"}
            },
            "reason": row["reason"],
            "violations": [],
            "diagnostics": {
                "source_scorer": judged["schema_version"],
                "judge_request_hash": judged["judge"]["requests"][row["metric_id"]]["request_hash"],
            },
        }
        for row in judged["metric_results"]
        if row["metric_id"] in JUDGE_METRIC_IDS
    }
    if set(rows) != JUDGE_METRIC_IDS or set(instances) != JUDGE_METRIC_IDS:
        raise ValueError("judge/core metric coverage must be exactly M15/M16/M19")
    core["metric_results"] = [
        rows.get(row["metric_id"], row) for row in core["metric_results"]
    ]
    core["violations"] = [
        row for row in core["violations"] if row.get("metric_id") not in JUDGE_METRIC_IDS
    ]


def _merge_ticket_row(core: dict[str, Any], ticket_score: dict[str, Any]) -> None:
    current = next(row for row in core["metric_results"] if row["metric_id"] == "M14")
    value = deepcopy(ticket_score.get("metric_results", {}).get("M14", "N/M"))
    passed = value.get("passed") if isinstance(value, dict) else None
    status = "pass" if passed is True else "fail" if passed is False else "N/M"
    replacement = {
        "metric_instance_id": current["metric_instance_id"],
        "metric_id": "M14",
        "status": status,
        "value": value,
        "reason": str(ticket_score.get("reason") or ticket_score["status"]),
        "violations": [],
        "diagnostics": {
            "source_scorer": ticket_score["schema_version"],
            "source_status": ticket_score["status"],
        },
    }
    core["metric_results"] = [
        replacement if row["metric_id"] == "M14" else row
        for row in core["metric_results"]
    ]
    core["violations"] = [
        row for row in core["violations"] if row.get("metric_id") != "M14"
    ]


def _merge_m05_judge_row(core: dict[str, Any], judged: dict[str, Any]) -> None:
    """Merge the conversation judge's M05 row (disclosure and, when declared, consent)."""

    disclosure = next(
        row for row in judged["metric_results"] if row["metric_id"] == "M05"
    )
    row = next(item for item in core["metric_results"] if item["metric_id"] == "M05")
    row["diagnostics"]["semantic_disclosure"] = deepcopy(disclosure)
    if row["status"] != "pass" or disclosure["status"] in {"pass", "N/A"}:
        return
    row["status"] = disclosure["status"]
    row["reason"] = disclosure["reason"]
    if disclosure["status"] != "fail":
        return
    violation = {
        "metric_instance_id": row["metric_instance_id"],
        "metric_id": "M05",
        "violation_type": "semantic_disclosure_not_observed",
        "evidence": deepcopy(disclosure["judgements"]),
    }
    row["violations"].append(violation)
    core["violations"].append(violation)


def _merge_m05_consent(core: dict[str, Any], consent: dict[str, Any]) -> None:
    if consent["status"] == "N/A":
        return
    row = next(item for item in core["metric_results"] if item["metric_id"] == "M05")
    row["diagnostics"]["semantic_consent"] = {
        key: deepcopy(consent[key])
        for key in ("schema_version", "status", "reason", "judgements", "judge")
    }
    if row["status"] != "pass" or consent["status"] == "pass":
        return
    row["status"] = consent["status"]
    row["reason"] = consent["reason"]
    if consent["status"] != "fail":
        return
    violation = {
        "metric_instance_id": row["metric_instance_id"],
        "metric_id": "M05",
        "violation_type": "semantic_consent_not_observed",
        "evidence": deepcopy(consent["judgements"]),
    }
    row["violations"].append(violation)
    core["violations"].append(violation)


def _load_dotenv(path: Path) -> None:
    """Load simple UTF-8 ``KEY=VALUE`` lines without expansion or commands."""

    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"malformed dotenv assignment at line {number}")
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.env_file is not None:
        _load_dotenv(args.env_file)
    elif (REPOSITORY_ROOT / ".env").is_file():
        _load_dotenv(REPOSITORY_ROOT / ".env")
    report = score_package_record(
        package=load_package(args.package),
        record=load_mapping(args.record),
        judge_config=load_judge_config(args.config),
        cache_dir=args.cache_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
