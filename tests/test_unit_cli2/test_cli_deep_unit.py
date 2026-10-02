"""Filesystem-only coverage for evaluator CLI validation and scoring paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from elyza_agent_tasks_customer_service.evaluation.cli import run_eval


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _package(path: Path, scenario_id: str = "s1") -> Path:
    path.write_text(f"scenario_id: {scenario_id}\nvariants: [baseline]\n", encoding="utf-8")
    return path


def test_run_eval_score_only_counts_one_persisted_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exactly one existing score-only record increments the persisted count once."""

    run_dir = run_eval.run_output_dir(tmp_path, scenario_id="s1", variant_id="hard", run_number=1)
    _json(run_dir / "record.json", {"scenario_id": "s1", "status": "success"})
    monkeypatch.setattr(run_eval, "_write_package_score", lambda **_: run_dir / "score.json")
    assert run_eval._run_package_score_only(
        entries={"s1": {"scenario": {}}},
        output_dir=tmp_path,
        variant_id="hard",
        io_mode="text-text",
        judge_config={},
        judge_cache_dir=tmp_path / "cache",
        model="m",
    ) is True
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["scored_count"] == 1
    assert summary["skipped_count"] == 0


def test_run_score_only_validation_and_main_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = tmp_path / "p"
    package_dir.mkdir()
    _package(package_dir / "s1.yaml")
    judge = tmp_path / "judge.yaml"
    judge.write_text("judge: ok\n", encoding="utf-8")
    config = {"scenario_dir": str(package_dir), "output_dir": str(tmp_path / "out"), "variant_id": "baseline", "score_only": True, "conversation_log_scoring": {"config_path": str(judge), "cache_dir": str(tmp_path / "cache")}}
    monkeypatch.setattr(run_eval, "load_judge_config", lambda _: {})
    with pytest.raises(ValueError, match="score_only"):
        run_eval.run_package_eval({**config, "score_only": "yes"})
    assert run_eval.classify_runtime_failure(error_type="TimeoutError", error_message="timed out") == "timeout_failure"
    assert run_eval.classify_runtime_failure(error_type="ValueError", error_message="ASR missing") == "asr_failure"

    captured: dict[str, object] = {}
    monkeypatch.setattr(run_eval, "run_package_eval", lambda cfg, endpoint: captured.update(config=cfg, endpoint=endpoint) or True)
    monkeypatch.setattr(sys, "argv", ["run_eval", json.dumps({"mode": "chat_tools"}), "http://transport/"])
    assert run_eval.main() == 0
    assert captured["endpoint"] == "http://transport"


def test_run_package_executes_one_stubbed_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = tmp_path / "packages"
    package_dir.mkdir()
    _package(package_dir / "s1.yaml")

    class Runtime:
        def __init__(self, *args, **kwargs): pass
        def tool_definitions(self): return []

    captured: dict[str, object] = {}

    def runner(**kwargs):
        captured.update(kwargs)
        run_dir = run_eval.run_output_dir(kwargs["output_dir"], scenario_id=kwargs["scenario_id"], variant_id=kwargs["variant_id"], run_number=1)
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {"scenario_id": kwargs["scenario_id"], "status": "success", "run_id": "r", "event_log": [], "wall_time_sec": 0.1}
        _json(run_dir / "record.json", record)
        return [record]

    monkeypatch.setattr(run_eval, "PackageRuntime", Runtime)
    monkeypatch.setattr(run_eval, "run_package_chat", runner)
    assert run_eval.run_package_eval({"scenario_dir": str(package_dir), "output_dir": str(tmp_path / "out"), "variant_id": "hard", "api_model": "model", "max_workers": 1}, "http://endpoint")
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))["success_count"] == 1
    assert captured["max_tool_rounds"] == 40
    assert captured["variant_id"] == "hard"
    assert (tmp_path / "out" / "s1" / "hard" / "run_001" / "record.json").is_file()
    assert captured["max_turns"] == 28
    assert captured["timeout_sec"] == 600.0
    assert captured["operator_uses_responses"] is False
    assert captured["user_controller_endpoint"] == "http://endpoint"
