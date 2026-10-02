"""Remaining ledger boundaries that can be checked without a live runtime."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from elyza_agent_tasks_customer_service.evaluation.audio.audio_signal_metrics import (
    score_signal_metrics,
)
from elyza_agent_tasks_customer_service.evaluation.audio.audio_value_metrics import (
    score_value_metrics,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.package_scoring import (
    score_package_record,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.resolution_scoring import (
    score_resolution,
)
from tests.helpers.gold_record_builder import (
    JUDGE_CONFIG,
    load_source_package,
    perfect_judge_transport,
    perfect_record,
)


def _rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a package score report's object-array metric rows by metric ID."""

    return {row["metric_id"]: row for row in report["metric_results"]}


def _score(package: dict[str, Any], record: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Score one fixture package with the fixed local judge transport."""

    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")
    return _rows(
        score_package_record(
            package=package,
            record=record,
            judge_config=JUDGE_CONFIG,
            cache_dir=tmp_path,
            transport=perfect_judge_transport,
        )
    )


def test_failed_runtime_rows_are_explicit_not_measurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """台帳B/M01,M09,M11,M14,M15,M16,M17,M19,M24のNM1/NM2を固定する。"""

    package = load_source_package("hotel", "htl-001")
    record = perfect_record(package)
    record.update(status="failed_runtime", error_message="fixture transport failed")
    rows = _score(package, record, tmp_path, monkeypatch)

    for metric_id in ("M01", "M09", "M11", "M14", "M15", "M16", "M17", "M19", "M24"):
        assert rows[metric_id]["status"] == "N/M"
        assert rows[metric_id]["reason"] == "run_failed: fixture transport failed"
    assert rows["M04"]["reason"] == "run_failed: fixture transport failed"
    assert rows["M05"]["reason"] == "run_failed: fixture transport failed"


@pytest.mark.parametrize(
    ("ledger", "field", "replacement"),
    [
        ("B/M14 F2", "target_ids", []),
        ("B/M14 F3", "performed_actions", []),
        ("B/M14 F4", "evidence_refs", []),
    ],
)
def test_ticket_fact_mismatches_name_the_exact_field(
    ledger: str, field: str, replacement: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳B/M14 F2--F4は不一致fieldと期待値を保持して失敗する。"""

    del ledger
    package = load_source_package("hotel", "htl-001")
    record = perfect_record(package)
    expected = deepcopy(record["operator_ticket_artifact"]["ticket"][field])
    record["operator_ticket_artifact"]["ticket"][field] = replacement
    row = _score(package, record, tmp_path, monkeypatch)["M14"]

    assert row["status"] == "fail"
    assert row["value"]["deterministic_field_results"][field] is False
    assert row["value"]["ticket_fidelity_deterministic"] == 0.75
    assert expected != replacement


def test_missing_ticket_and_invalid_ticket_are_distinct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """台帳B/M14 NM1/F5は未提出とschema不正を別の理由コードで固定する。"""

    package = load_source_package("hotel", "htl-001")
    missing = perfect_record(package)
    missing["operator_ticket_artifact"] = None
    invalid = perfect_record(package)
    invalid["operator_ticket_artifact"]["ticket"]["target_ids"] = "not-an-array"

    missing_row = _score(package, missing, tmp_path / "missing", monkeypatch)["M14"]
    assert (missing_row["status"], missing_row["reason"]) == ("N/M", "operator_ticket_missing")
    with pytest.raises(ValueError, match="target_ids must be an array"):
        _score(package, invalid, tmp_path / "invalid", monkeypatch)


def test_m01_private_reference_and_final_world_shape_boundaries() -> None:
    """台帳B/M01 NM1/X1はprivate reference欠落とworld shape例外を固定する。"""

    record = {
        "final_world": {"table": []},
        "status": "success",
        "hard_fails": [],
        "run_outcome": {"schema_version": "run_outcome", "completed": True, "handling_type": "change_procedure", "reason": "fixture"},
    }
    missing = score_resolution(scenario_id="case", record=record, call_events=[], frozen_reference=None)

    assert (missing["status"], missing["reason"], missing["value"]) == (
        "N/M",
        "frozen_final_world_reference_missing",
        None,
    )
    with pytest.raises(ValueError, match="record.final_world must be an object"):
        score_resolution(scenario_id="case", record={"final_world": []}, call_events=[], frozen_reference=None)


def test_audio_missing_consensus_and_no_m05_contract_are_na() -> None:
    """台帳B/M20 NM1/NA2およびM22 NA1はv2の現行N/A契約を固定する。"""

    result = score_value_metrics(
        record={"event_log": [{"seq": 1, "event_type": "message", "actor": "user", "content": "09000000035"}]},
        scenario={"contracts": {"m20": {"important_values": [{"slot_id": "phone", "value": "09000000035", "value_type": "phone"}]}}},
    )["metrics"]

    assert (result["M20"]["status"], result["M20"]["units"][0]["unit_id"]) == ("failed", "phone")
    assert (result["M22"]["status"], result["M22"]["units"]) == ("N/A", [])


def test_legacy_audio_invalid_manifest_is_contract_invalid_for_every_metric() -> None:
    """台帳B/M21,M23 CI1はlegacy manifest不正で全metricがcontract_invalidになる。"""

    rows = score_signal_metrics(audio_manifest={}, event_log=[], config={})["metrics"]

    assert set(rows) == {"M21", "M23"}
    for metric_id in ("M21", "M23"):
        assert rows[metric_id]["status"] == "contract_invalid"
        assert rows[metric_id]["reason"] == "evidence_root is required when audio_manifest is supplied in memory"
