"""Judge response boundary cases for M15, M16, and M19."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from elyza_agent_tasks_customer_service.evaluation.scoring.package_scoring import (
    score_package_record,
)
from tests.helpers.gold_record_builder import (
    JUDGE_CONFIG,
    load_source_package,
    perfect_judge_transport,
    perfect_record,
)


def _score(
    transport: Callable[..., dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, dict[str, Any]]:
    """Score the representative package with one deterministic judge transport."""

    monkeypatch.setenv(JUDGE_CONFIG["judge"]["api_key_env"], "test")
    package = load_source_package("hotel", "htl-001")
    report = score_package_record(
        package=package,
        record=perfect_record(package),
        judge_config=JUDGE_CONFIG,
        cache_dir=tmp_path,
        transport=transport,
    )
    return {row["metric_id"]: row for row in report["metric_results"]}


def test_invalid_judge_json_is_a_fixed_integration_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳B/M15・M16・M19 X1: judge不正JSONは例外として固定する。"""

    def transport(
        _url: str, payload: dict[str, Any], _headers: dict[str, str], _timeout: float
    ) -> dict[str, Any]:
        return {"model": payload["model"], "choices": [{"message": {"content": "{"}}]}

    with pytest.raises(ValueError, match="conversation-log judge failed: Expecting property name"):
        _score(transport, tmp_path, monkeypatch)


def test_partial_judge_response_is_a_fixed_integration_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳B/M15・M16・M19 NM2: partial judge responseは例外として固定する。"""

    def transport(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        response = perfect_judge_transport(url, payload, headers, timeout)
        content = json.loads(response["choices"][0]["message"]["content"])
        content["judgements"] = content["judgements"][:-1]
        response["choices"][0]["message"]["content"] = json.dumps(content)
        return response

    with pytest.raises(ValueError, match="conversation-log judge failed: judge response violates schema"):
        _score(transport, tmp_path, monkeypatch)


def test_contradictory_judge_answers_demote_m16_and_m19(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """台帳B/M16・M19 F/NM: contradictory valid judge answers demote the rows."""

    def transport(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        response = perfect_judge_transport(url, payload, headers, timeout)
        content = json.loads(response["choices"][0]["message"]["content"])
        for judgement in content["judgements"]:
            if ":M16:" in judgement["question_id"] or ":M19:" in judgement["question_id"]:
                judgement.update(answer="no", evidence=[])
        response["choices"][0]["message"]["content"] = json.dumps(content)
        return response

    rows = _score(transport, tmp_path, monkeypatch)

    assert rows["M16"]["status"] == "measured"
    assert rows["M16"]["value"]["fired_item_pass_rate"] == 0.0
    assert [item["criterion_id"] for item in rows["M16"]["value"]["machine_observations"]] == ["M16-M1", "M16-M2", "M16-M4", "M16-M5"]
    assert rows["M19"]["status"] == "fail"
    assert rows["M19"]["reason"] == "one_or_more_judgements_mismatch"
    assert rows["M19"]["value"]["judgements"][0]["question_id"] == "htl-001:M19:CQ1"
