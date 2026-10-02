"""Field-ledger effects at the package-to-scorer boundary."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from elyza_agent_tasks_customer_service.evaluation.engine.package_adapter import (
    convert_package,
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


def _package() -> dict[str, Any]:
    """Load the complete representative source package used by ledger checks."""

    return load_source_package("hotel", "htl-001")


def _remove(value: dict[str, Any], *path: str) -> None:
    """Delete one required object field; test fixtures use exact object paths."""

    current: Any = value
    for part in path[:-1]:
        current = current[part]
    del current[path[-1]]


def _metric_rows(package: dict[str, Any], cache_dir: Any, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Score all text metrics with stable fixture answers for projection comparison."""

    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")
    return score_package_record(
        package=package,
        record=perfect_record(_package()),
        judge_config=JUDGE_CONFIG,
        cache_dir=cache_dir,
        transport=perfect_judge_transport,
    )["metric_results"]


@pytest.mark.parametrize(
    ("ledger", "path", "error_text"),
    [
        ("A/T01 scenario_id", ("scenario_id",), "package.scenario_id"),
        ("A/T02 business.initial_user_request", ("business", "initial_user_request"), None),
        ("A/T06 domain.domain_id", ("domain", "domain_id"), None),
        ("A/T10 world_schema.customer_table", ("world_schema", "customer_table"), None),
        ("A/S01 answer.expected_mutations", ("answer", "expected_mutations"), "package.answer.expected_mutations"),
        ("A/S05 contracts.completion.required_consent", ("contracts", "completion", "required_consent"), None),
        ("A/S21 post_call_ticket_contract", ("post_call_ticket_contract",), None),
        ("A/S23 use_sops", ("use_sops",), "package.use_sops"),
        ("A/S24 world_hashes.final", ("world_hashes", "final"), None),
    ],
)
def test_read_field_deletion_is_an_explicit_adapter_error(
    ledger: str, path: tuple[str, ...], error_text: str | None
) -> None:
    """台帳Aのread field削除が採点契約の明示エラーになることを固定する。"""

    del ledger
    package = _package()
    _remove(package, *path)

    baseline = convert_package(_package(), mode="text")
    if error_text:
        with pytest.raises(Exception, match=error_text):
            convert_package(package, mode="text")
    else:
        assert convert_package(package, mode="text") != baseline


@pytest.mark.parametrize(
    ("ledger", "path"),
    [
        ("A/T09 world_schema.entities.*.description", ("world_schema", "entities", "reservations", "description")),
        ("A/T11 identity_policy.question_grouping", ("world_schema", "identity_policy", "question_grouping")),
        ("A/T11 identity_policy.match_mode", ("world_schema", "identity_policy", "match_mode")),
        ("A/T11 identity_policy.question", ("world_schema", "identity_policy", "question")),
        ("A/T12 search_policy.require_filter", ("world_schema", "search_policy", "require_filter")),
    ],
)
def test_task_unread_field_deletion_does_not_change_scorer_projection(
    ledger: str, path: tuple[str, ...], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳Aの未読task field削除後も全metric入力projectionは同一である。"""

    del ledger
    package = _package()
    baseline = _metric_rows(package, tmp_path / "baseline", monkeypatch)
    _remove(package, *path)

    assert _metric_rows(package, tmp_path / "candidate", monkeypatch) == baseline


def test_solution_unread_fields_are_not_in_scorer_projection(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳Aのsolution未読65 leafはprojectionに出ず、全metric入力は不変である。"""

    package = _package()
    baseline = _metric_rows(package, tmp_path / "baseline", monkeypatch)
    candidate = deepcopy(package)
    for turn in candidate["gold_dialogue"]["turns"]:
        turn.pop("turn_id", None)
        turn.pop("organization", None)
        turn.pop("source", None)
        if turn["kind"] == "tool_result":
            turn.pop("returned_rows", None)
            turn.pop("world_changes", None)
    candidate["gold_dialogue"].pop("scenario_id", None)
    candidate["gold_dialogue"].pop("final_world_hash", None)
    candidate["gold_dialogue"].pop("tool_trace", None)
    candidate["world_hashes"].pop("initial", None)
    candidate["contracts"]["completion"].pop("handling_type", None)

    assert _metric_rows(candidate, tmp_path / "candidate", monkeypatch) == baseline
