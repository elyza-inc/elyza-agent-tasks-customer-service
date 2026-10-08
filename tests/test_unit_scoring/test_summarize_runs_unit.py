"""Direct unit tests for published run-summary tables."""

from __future__ import annotations

import json

import pytest

from elyza_agent_tasks_customer_service.evaluation.aggregate import summarize_runs as summary


def _entry(**metrics):
    return {"metrics": metrics, "outcome": "終話到達", "audio": None,
            "customer_turns": 2, "tool_calls": 3}


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ({"metric_id": "M01", "status": "pass"}, 1.0),
        ({"metric_id": "M01", "status": "fail"}, 0.0),
        ({"metric_id": "M01", "status": "N/M"}, 0.0),
        ({"metric_id": "M01", "status": "contract_invalid"}, 0.0),
        ({"metric_id": "M01", "status": "N/A"}, None),
        ({"metric_id": "M14", "status": "pass", "value": {"ticket_fidelity_deterministic": .25}}, .25),
        ({"metric_id": "M15", "status": "fail", "value": {"score": .5}}, .5),
        ({"metric_id": "M11", "status": "measured", "value": {"applicable_pass_rate": .75}}, .75),
        ({"metric_id": "M16", "status": "measured", "value": {}}, 0.0),
    ],
)
def test_entity_score_statuses(metric, expected):
    assert summary.entity_score(metric) == expected


def test_entity_score_unknown_status_and_small_statistics():
    with pytest.raises(ValueError, match="unknown metric status"):
        summary.entity_score({"metric_id": "M01", "status": "wat"})
    assert summary.mean([]) is None
    assert summary.mean([2.0]) == 2.0
    assert summary.boot_ci([]) == (None, None)
    assert summary.boot_ci([.5], n=4, seed=1) == (.5, .5)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (None, "終話到達"),
        ("run_failed: max_turns reached", "上限打ち切り"),
        ("run_failed: max_tool_rounds reached", "上限打ち切り"),
        ("run_failed: HTTP 500", "測定エラー"),
    ],
)
def test_run_outcome_uses_run_failed_reason(reason, expected):
    metric_results = [{"reason": reason}] if reason else [{"reason": "pass"}]
    assert summary.run_outcome(metric_results) == expected


def test_load_runs_and_subfacets_with_all_optional_artifacts(tmp_path):
    run = tmp_path / "case-1" / "baseline" / "run_001"
    (run / "audio_user").mkdir(parents=True)
    (run / "score.json").write_text(json.dumps({"scenario_id": "actual-1", "metric_results": [
        {"metric_id": "M01", "status": "pass"},
        {"metric_id": "M11", "status": "measured", "diagnostics": {"source_result": {"details": {"sub_facets": {
            "required_success": {"status": "pass"}, "bad": {"status": 1}}}}}},
    ]}))
    (run / "record.json").write_text('{"status":"success"}')
    (run / "event_log.json").write_text(json.dumps([
        {"actor": "user", "event_type": "message"}, {"event_type": "tool_call"}, {}]))
    (run / "audio_user" / "audio_metric_results.json").write_text('{"metrics":{"M20":{"status":"passed"}}}')
    loaded = summary.load_runs([tmp_path])
    assert loaded["baseline"]["actual-1"] == {
        "metrics": {"M01": (1.0, "pass"), "M11": (0.0, "measured")}, "outcome": "終話到達",
        "audio": {"M20": {"status": "passed"}}, "customer_turns": 1, "tool_calls": 1,
    }
    assert summary.load_m11_subfacets([tmp_path]) == {"baseline": {"actual-1": {"required_success": "pass"}}}


def test_load_runs_counts_a_scored_cutoff_as_limit_outcome(tmp_path):
    run = tmp_path / "case-1" / "hard" / "run_001"
    run.mkdir(parents=True)
    (run / "score.json").write_text('{"scenario_id":"case-1","metric_results":[{"metric_id":"M01","status":"fail"}]}')
    (run / "record.json").write_text('{"status":"success","call_limit_reached":"max_turns"}')
    entry = summary.load_runs([tmp_path])["hard"]["case-1"]
    assert entry["outcome"] == "上限打ち切り" and entry["metrics"]["M01"] == (0.0, "fail")


def test_load_runs_rejects_duplicate_scenario(tmp_path):
    for variant in ("baseline", "hard"):
        path = tmp_path / "same" / variant / "run_001"
        path.mkdir(parents=True)
        (path / "score.json").write_text('{"scenario_id":"same","metric_results":[]}')
    # Duplicates are scoped per variant, so the two entries are valid.
    assert set(summary.load_runs([tmp_path])) == {"baseline", "hard"}
    duplicate = tmp_path / "other" / "baseline" / "run_001"
    duplicate.mkdir(parents=True)
    (duplicate / "score.json").write_text('{"scenario_id":"same","metric_results":[]}')
    with pytest.raises(ValueError, match="duplicate scenario"):
        summary.load_runs([tmp_path])
    with pytest.raises(ValueError, match="duplicate scenario"):
        summary.load_m11_subfacets([tmp_path])


def test_summary_tables_cover_empty_paired_hard_only_and_audio_paths(capsys, tmp_path):
    base = {"x-1": _entry(M01=(1., "pass"), M04=(None, "N/A"), M05=(0., "N/M"), M11=(None, "N/A"))}
    hard = {"x-1": _entry(M01=(0., "fail"), M04=(1., "pass"), M05=(1., "pass"), M11=(.5, "measured"))}
    metric_lines = summary.metric_table(base, hard)
    category_lines = summary.category_table(base, hard)
    assert any("M04 | — | 1.000 | —(構造的N/A)" in line for line in metric_lines)
    assert any("タスク遂行 | 1.000 | 0.000 | -1.000" in line for line in category_lines)
    facets = summary.m11_subfacet_table({"a": {"required_success": "N/A"}}, {"a": {"required_success": "pass"}})
    assert "必須操作の成功 | — | 1.000 | 0/1" in facets[2]
    assert summary.audio_table({"x": _entry()}) is None
    audio_entry = _entry()
    audio_entry["audio"] = {"M20": {"units": [{"status": "passed"}, {"status": "failed"}]},
                            "M25": {"status": "passed", "observation": {"target_value_count": 2, "opportunity_count": 3}}}
    audio_lines = summary.audio_table({"x": audio_entry})
    assert "| M20 | 0.500 | 2 | 0 | 0 |" in audio_lines
    assert "| M25観測率 | 1.000 | 2 | 0 | 0 |" in audio_lines
    complete_audio = _entry()
    complete_audio["audio"] = {metric: {"status": "passed"} for metric in summary.AUDIO_CATEGORY_METRICS}
    assert any("| 音声カテゴリ | 1.000" in line for line in summary.audio_table({"x": complete_audio}))
    assert summary.domain_table({"zzz-1": hard["x-1"]})[-1].startswith("| zzz |")
    assert "| baseline | 1/1 | 2.0 | 3.0 |" in summary.side_table({"baseline": base})[-1]
    run = tmp_path / "s" / "hard" / "run_001"
    run.mkdir(parents=True)
    (run / "score.json").write_text('{"scenario_id":"s","metric_results":[]}')
    assert summary.main(["--results-root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "実体数 baseline=0 hard=1" in output
    assert "終話到達 b=0/0 h=1/1 | 上限打ち切り b=0 h=0 | 測定エラー b=0 h=0" in output
    assert "### ドメイン別(hard)" in output
