"""Keep the published metric descriptions aligned with emitted score rows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from elyza_agent_tasks_customer_service.evaluation.contracts.metric_inventory import (
    load_metric_inventory,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.package_scoring import (
    score_package_record,
)
from tests.helpers.gold_record_builder import (
    JUDGE_CONFIG,
    load_source_package,
    perfect_judge_transport,
    perfect_record,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS_DOCUMENT = ROOT / "docs" / "METRICS.md"
SAMPLE_ROOT = ROOT / ".run" / "records-claude-opus-5-baseline"
PACKAGES_DIR = ROOT / ".run" / "packages"
STATUS_VOCABULARY = {"pass", "fail", "measured", "N/M", "N/A", "contract_invalid"}
RATE_METRICS = {"M09", "M11", "M14", "M15", "M16", "M19", "M25", "M26"}
BINARY_METRICS = {"M01", "M04", "M05", "M17", "M20", "M22", "M23", "M24"}
RATE_VALUE_KEYS = {
    "M09": ("score_inverse_k", "applicable_pass_rate"),
    "M11": "applicable_pass_rate",
    "M14": "ticket_fidelity_deterministic",
    "M15": "score",
    "M16": "fired_item_pass_rate",
    "M19": "score",
    "M25": "accuracy",
    "M26": "grounded_use_rate",
}


def parse_metrics_document(path: Path = METRICS_DOCUMENT) -> dict[str, dict[str, Any]]:
    """Parse the metric list tables in METRICS.md.

    Each row is ``| [名前](#anchor) | M01 | 型 | ...``. A type cell without
    二値 or 率 (M21's ``—``) intentionally leaves ``score_type`` empty.
    """

    text = path.read_text(encoding="utf-8")
    rows_in_doc = re.findall(r"^\|\s*\[([^\]]+)\]\(#[^)]*\)[^|]*\|\s*(M\d{2})\s*\|\s*([^|]+?)\s*\|", text, re.MULTILINE)
    rules = "N/Mは0点、N/Aだけ除く" in text
    paired = "hard - baseline の差を平均" in text
    rows = {}
    for name, metric_id, qualifier in rows_in_doc:
        score_type = "binary" if "二値" in qualifier else "rate" if "率" in qualifier else None
        rows[metric_id] = {
            "name": name,
            "score_type": score_type,
            "na_condition": "N/A" in text,
            "nm_condition": "N/M" in text,
            "aggregation": "nm_zero_na_excluded" if rules else None,
            "paired_difference": paired and metric_id not in {"M04", "M19"},
        }
    return rows


def _sample_rows() -> list[dict[str, Any]]:
    paths = sorted(SAMPLE_ROOT.rglob("score.json"))[:3]
    if not paths:
        pytest.skip("baseline sample score.json is not part of the checkout")
    return [row for path in paths for row in json.loads(path.read_text(encoding="utf-8"))["metric_results"]]


def _perfect_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    package = load_source_package("hotel", "htl-001")
    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")
    report = score_package_record(
        package=package,
        record=perfect_record(package),
        judge_config=JUDGE_CONFIG,
        cache_dir=tmp_path,
        transport=perfect_judge_transport,
    )
    return report["metric_results"]


def test_document_metric_ids_match_active_inventory() -> None:
    documented = parse_metrics_document()
    inventory_ids = {row["metric_id"] for row in load_metric_inventory()["metric_rows"]}
    assert set(documented) == inventory_ids


def test_document_declares_na_nm_and_aggregation_rules() -> None:
    documented = parse_metrics_document()
    assert all(row["na_condition"] and row["nm_condition"] for row in documented.values())
    assert {row["aggregation"] for row in documented.values()} == {"nm_zero_na_excluded"}
    assert all(documented[metric_id]["paired_difference"] for metric_id in documented if metric_id not in {"M04", "M19"})


@pytest.mark.xfail(reason="M21 is listed as a reference value without a score type.", strict=True)
def test_document_has_a_contract_for_every_inventory_metric() -> None:
    documented = parse_metrics_document()
    assert all(row["score_type"] is not None for row in documented.values())


def test_documented_types_and_statuses_match_samples_and_perfect_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documented = parse_metrics_document()
    rows = _sample_rows() + _perfect_rows(tmp_path, monkeypatch)
    for row in rows:
        metric_id, status = row["metric_id"], row["status"]
        assert metric_id in documented
        assert status in STATUS_VOCABULARY
        if status in {"N/A", "N/M", "contract_invalid"}:
            continue
        if metric_id in BINARY_METRICS:
            assert documented[metric_id]["score_type"] == "binary"
            assert status in {"pass", "fail"}
        if metric_id in RATE_METRICS:
            assert documented[metric_id]["score_type"] == "rate"
            assert isinstance(row.get("value"), dict)
            required_key = RATE_VALUE_KEYS[metric_id]
            if isinstance(required_key, tuple):
                assert any(key in row["value"] for key in required_key)
            else:
                assert required_key in row["value"]
