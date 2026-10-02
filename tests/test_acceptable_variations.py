"""Metamorphic checks for business-valid variations of gold conversations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from elyza_agent_tasks_customer_service.evaluation.engine.package_runtime import (
    PackageRuntime,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.package_scoring import (
    score_package_record,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.resolution_scoring import (
    _canonical_final_world,
)
from tests.helpers.acceptable_variation_transformer import TRANSFORM_KINDS, transform
from tests.helpers.gold_record_builder import (
    JUDGE_CONFIG,
    perfect_judge_transport,
    perfect_record,
    project_record,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = ROOT / ".run" / "packages"
SCENARIO_IDS = (
    "ecf-002",
    "htl-002",
    "par-001",
    "tel-002",
    "htl-001",
)
MEASURED_RATE_KEYS = {
    "M09": "applicable_pass_rate",
    "M11": "applicable_pass_rate",
    "M16": "fired_item_pass_rate",
}
M24_REJECTED_VARIATIONS = frozenset(
    {
        ("htl-001", "insert_confirmation"),
        ("htl-002", "insert_confirmation"),
        ("tel-002", "insert_confirmation"),
        ("par-001", "insert_confirmation"),
    }
)
VARIATION_CASES = tuple(
    pytest.param(scenario_id, kind, id=f"{scenario_id}-{kind}")
    for scenario_id in SCENARIO_IDS
    for kind in TRANSFORM_KINDS
)


def _load_package(scenario_id: str) -> dict[str, Any]:
    """Load one JSON-object package stored with a ``.yaml`` suffix.

    Array/scalar roots and malformed JSON raise ``ValueError`` or
    ``JSONDecodeError``; a missing package raises ``FileNotFoundError``.
    """

    package = json.loads((PACKAGES_DIR / f"{scenario_id}.yaml").read_text(encoding="utf-8"))
    if not isinstance(package, dict):
        raise ValueError("package root must be an object")
    return package


def _replay_world(package: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    runtime = PackageRuntime(package, scenario_id=package["scenario_id"])
    for event in record["event_log"]:
        if event.get("event_type") == "tool_call" and event.get("tool") in package["tools"]:
            runtime.call(event["tool"], event["arguments"])
    return _canonical_final_world(runtime.world)


def _score(
    package: dict[str, Any], record: dict[str, Any], cache_dir: Path
) -> list[dict[str, Any]]:
    return score_package_record(
        package=package,
        record=record,
        judge_config=JUDGE_CONFIG,
        cache_dir=cache_dir,
        transport=perfect_judge_transport,
    )["metric_results"]


def _at_ceiling(row: dict[str, Any]) -> bool:
    if row["status"] == "pass":
        return True
    rate_key = MEASURED_RATE_KEYS.get(row["metric_id"])
    return row["status"] == "measured" and row["value"].get(rate_key) == 1.0


@pytest.mark.parametrize(
    ("scenario_id", "kind"),
    VARIATION_CASES,
)
def test_business_valid_variation_keeps_every_metric_at_ceiling(
    scenario_id: str,
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _load_package(scenario_id)
    original = perfect_record(package)
    try:
        varied = transform(original, kind)
    except ValueError as exc:
        pytest.skip(str(exc))
    project_record(varied, package)

    expected_world = _canonical_final_world(original["final_world"])
    replayed_world = _replay_world(package, varied)
    if replayed_world != expected_world:
        pytest.skip("variation does not replay to the original final world")
    assert replayed_world == expected_world

    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")
    baseline = _score(package, original, tmp_path / "baseline")
    actual = _score(package, varied, tmp_path / "varied")
    baseline_na = {row["metric_id"] for row in baseline if row["status"] == "N/A"}
    actual_na = {row["metric_id"] for row in actual if row["status"] == "N/A"}

    assert actual_na == baseline_na
    non_ceiling = {
        row["metric_id"]: (row["status"], row.get("reason"))
        for row in actual
        if row["metric_id"] not in actual_na and not _at_ceiling(row)
    }
    if (scenario_id, kind) in M24_REJECTED_VARIATIONS:
        assert non_ceiling == {"M24": ("fail", "written_style_markup_detected")}
    else:
        assert not non_ceiling, {
            row["metric_id"]: (row["status"], row.get("reason"))
            for row in actual
            if row["metric_id"] not in actual_na and not _at_ceiling(row)
        }
