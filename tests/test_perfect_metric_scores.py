"""D9: package-declared gold evidence reaches text metric ceilings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.perfect_run_report import (
    build_perfect_run_report,
)
from elyza_agent_tasks_customer_service.evaluation.engine.package_adapter import (
    _japanese_register_context,
    convert_package,
    load_package,
)
from elyza_agent_tasks_customer_service.evaluation.observers.japanese_parse_adapter import (
    parse_japanese_turns,
)
from elyza_agent_tasks_customer_service.evaluation.observers.japanese_rule_observer import (
    load_rule_catalog,
    score_japanese_register_from_parse,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.core_metric_scoring import (
    score_m04_customer_trigger_control,
)
from elyza_agent_tasks_customer_service.evaluation.scoring.package_scoring import (
    score_package_record,
)
from tests import SCENARIO_COUNT
from tests.helpers.gold_record_builder import (
    JUDGE_CONFIG,
    load_source_package,
    perfect_judge_transport,
    perfect_record,
    project_record,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = ROOT / ".run" / "packages"
REMAINING_PROVENANCE_FIX_CASES = (
    ("htl-003", "plan_name"),
    ("htl-008", "reissue_count"),
)
M04_SELF_PRECONDITION_CASES = ("tel-014", "tel-018", "tel-022")
CORPORATE_REPRESENTATIVE_CASES = (
    "htl-022",
    "tel-022",
)
EXPLICIT_EXTERNAL_REPRESENTATIVE_CASES = {"tel-022"}
IN_GROUP_STAFF_CASES = {"par-018"}


def _score(
    package: dict,
    record: dict,
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")
    return score_package_record(
        package=package,
        record=record,
        judge_config=JUDGE_CONFIG,
        cache_dir=cache_dir,
        transport=perfect_judge_transport,
    )


def _by_metric(report: dict) -> dict[str, dict]:
    return {row["metric_id"]: row for row in report["metric_results"]}


def test_all_96_packages_are_swept_with_structured_failure_causes() -> None:
    report = build_perfect_run_report(PACKAGES_DIR)

    assert report["scenario_count"] == SCENARIO_COUNT
    assert report["all_metric_full_scenario_count"] == SCENARIO_COUNT, report["failures"]
    assert report["non_full_scenario_count"] == 0, report["failures"]
    assert report["judge"] == {"transport": "offline_stub", "cache": "temporary"}
    assert all(
        failure["sub_item"] is not None and isinstance(failure["cause_fields"], list)
        for failure in report["failures"]
    )


def test_all_96_zero_records_award_no_credit() -> None:
    report = build_perfect_run_report(PACKAGES_DIR, mode="zero")

    assert report["scenario_count"] == SCENARIO_COUNT
    assert report["all_metrics_no_credit_scenario_count"] == SCENARIO_COUNT, report["violations"]
    assert report["violation_count"] == 0, report["violations"]
    assert report["violations"] == [], report["violations"]


def test_all_96_packages_classify_tantosha_from_declared_inputs() -> None:
    classifications = {
        "out_group": set(),
        "in_group": set(),
        "unresolved": set(),
    }
    paths = sorted(PACKAGES_DIR.glob("*.yaml"))

    assert len(paths) == SCENARIO_COUNT
    for path in paths:
        package = json.loads(path.read_text(encoding="utf-8"))
        context = _japanese_register_context(package, package["scenario_id"])
        representatives = [
            person for person in context["persons"] if "担当者" in person["surface_forms"]
        ]
        assert len(representatives) <= 1
        relation = "unresolved"
        if representatives:
            relation = representatives[0]["relation_to_operator"]
        classifications[relation].add(package["scenario_id"])

    assert classifications["out_group"] == EXPLICIT_EXTERNAL_REPRESENTATIVE_CASES
    assert classifications["in_group"] == IN_GROUP_STAFF_CASES
    assert len(classifications["unresolved"]) == SCENARIO_COUNT - len(
        EXPLICIT_EXTERNAL_REPRESENTATIVE_CASES | IN_GROUP_STAFF_CASES
    )


@pytest.mark.parametrize(("scenario_id", "field_name"), REMAINING_PROVENANCE_FIX_CASES)
def test_remaining_provenance_cases_reach_m11_ceiling(
    scenario_id: str,
    field_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = load_package(PACKAGES_DIR / f"{scenario_id}.yaml")
    metrics = _by_metric(_score(package, perfect_record(package), tmp_path, monkeypatch))
    provenance = metrics["M11"]["diagnostics"]["source_result"]["details"][
        "sub_facets"
    ]["argument_provenance"]
    matching = [
        row
        for row in provenance["details"]["fields"]
        if row["field"] == field_name
    ]

    assert matching
    assert all(row["derived_source_kind"] != "unknown" for row in matching)
    assert metrics["M11"]["value"]["applicable_pass_rate"] == 1.0


@pytest.mark.parametrize("scenario_id", M04_SELF_PRECONDITION_CASES)
def test_m04_self_precondition_packages_reach_ceiling(
    scenario_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = load_package(PACKAGES_DIR / f"{scenario_id}.yaml")
    metrics = _by_metric(_score(package, perfect_record(package), tmp_path, monkeypatch))

    assert metrics["M04"]["status"] == "pass"


def test_tel_022_external_representative_honorific_is_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = load_package(PACKAGES_DIR / "tel-022.yaml")
    metrics = _by_metric(_score(package, perfect_record(package), tmp_path, monkeypatch))
    family = next(
        row
        for row in metrics["M17"]["value"]["rule_families"]
        if row["rule_family"] == "in_out_group"
    )

    assert metrics["M17"]["status"] == "pass"
    assert family["passed"] is True
    assert family["violations"] == []


def test_corporate_representative_honorific_cases_remain_valid() -> None:
    parse_bundle = parse_japanese_turns(
        turns=[{"turn_ref": "operator:1", "text": "ご担当者様がいらっしゃいます。"}]
    )
    rule_catalog = load_rule_catalog()

    for scenario_id in CORPORATE_REPRESENTATIVE_CASES:
        package = load_package(PACKAGES_DIR / f"{scenario_id}.yaml")
        result = score_japanese_register_from_parse(
            parse_bundle=parse_bundle,
            context=_japanese_register_context(package, scenario_id),
            rule_catalog=rule_catalog,
        )

        assert result["status"] == "pass", scenario_id
        assert result["violation_count"] == 0, scenario_id


def test_in_group_staff_humble_subject_cases_are_valid() -> None:
    parse_bundle = parse_japanese_turns(
        turns=[{"turn_ref": "operator:1", "text": "担当者がお伺いしました。"}]
    )
    rule_catalog = load_rule_catalog()

    for scenario_id in sorted(IN_GROUP_STAFF_CASES):
        package = load_package(PACKAGES_DIR / f"{scenario_id}.yaml")
        result = score_japanese_register_from_parse(
            parse_bundle=parse_bundle,
            context=_japanese_register_context(package, scenario_id),
            rule_catalog=rule_catalog,
        )

        assert result["status"] == "pass", scenario_id
        assert result["violation_count"] == 0, scenario_id


def test_tel_001_postal_code_has_a_declared_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = load_package(PACKAGES_DIR / "tel-001.yaml")
    metrics = _by_metric(_score(package, perfect_record(package), tmp_path, monkeypatch))
    provenance = metrics["M11"]["diagnostics"]["source_result"]["details"][
        "sub_facets"
    ]["argument_provenance"]
    postal_fields = [
        field
        for field in provenance["details"]["fields"]
        if field["field"] == "postal_code"
    ]

    assert postal_fields
    assert all(field["derived_source_kind"] != "unknown" for field in postal_fields)
    assert provenance["passed"] is True


def test_removing_one_required_tool_fails_m11(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = load_source_package("hotel", "htl-001")
    record = perfect_record(package)
    removed_tool = package["execution_trace"][-1]["tool_id"]
    record["event_log"] = [
        event for event in record["event_log"] if event.get("tool") != removed_tool
    ]
    project_record(record, package)
    metrics = _by_metric(_score(package, record, tmp_path, monkeypatch))

    assert metrics["M11"]["value"]["applicable_pass_rate"] < 1.0
    assert metrics["M01"]["status"] == "pass"
    assert metrics["M14"]["status"] == "pass"


def test_changing_final_world_fails_only_m01(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = load_source_package("hotel", "htl-001")
    record = perfect_record(package)
    table = next(iter(record["final_world"]))
    record["final_world"][table] = []
    metrics = _by_metric(_score(package, record, tmp_path, monkeypatch))

    assert metrics["M01"]["status"] == "fail"
    assert metrics["M11"]["value"]["applicable_pass_rate"] == 1.0
    assert metrics["M14"]["status"] == "pass"


def test_moving_identity_after_action_fails_m04() -> None:
    package = load_source_package("hotel", "htl-001")
    scenario = convert_package(package, mode="text")
    events = perfect_record(package)["event_log"]
    identity = package["contracts"]["m04"]["required_precondition_tool_ids"][0]
    action = package["contracts"]["m04"]["critical_action_tool_id"]
    identity_events = [event for event in events if event.get("tool") == identity]
    remaining = [event for event in events if event.get("tool") != identity]
    action_result = max(
        index for index, event in enumerate(remaining) if event.get("tool") == action
    )
    moved = remaining[: action_result + 1] + identity_events + remaining[action_result + 1 :]
    for seq, event in enumerate(moved, 1):
        event["seq"] = seq

    score = score_m04_customer_trigger_control(
        contract=scenario["m04_customer_trigger_contract"],
        call_events=moved,
    )

    assert score["status"] == "fail"


def test_changing_one_ticket_field_fails_only_m14(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = load_source_package("hotel", "htl-001")
    record = perfect_record(package)
    ticket = record["operator_ticket_artifact"]["ticket"]
    alternatives = package["post_call_ticket_contract"]["enum_catalog"][
        "handling_types"
    ]
    ticket["handling_type"] = next(
        value for value in alternatives if value != ticket["handling_type"]
    )
    metrics = _by_metric(_score(package, record, tmp_path, monkeypatch))

    assert metrics["M14"]["status"] == "fail"
    assert metrics["M01"]["status"] == "pass"
    assert metrics["M11"]["value"]["applicable_pass_rate"] == 1.0
    assert metrics["M15"]["status"] == "pass"
