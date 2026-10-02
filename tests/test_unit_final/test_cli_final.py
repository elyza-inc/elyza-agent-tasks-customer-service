"""CLI resume, parser, validation, and worker stubs without external services."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from elyza_agent_tasks_customer_service.evaluation.cli import run_eval


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _scenario(path: Path, scenario_id: str = "s1") -> Path:
    return _json(path, {"scenario_id": scenario_id, "measurement_contract": {}})


def _report() -> dict:
    return {"metric_results": [{"metric_id": "M01", "status": "pass"}], "diagnostics": {"implemented_metric_ids": []}, "violations": []}


def test_run_eval_workers_score_only_and_failure_categories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel worker accounting and score-only replay use only synthetic packages."""
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "s1.yaml").write_text("scenario_id: s1\n", encoding="utf-8")
    monkeypatch.setattr(run_eval, "load_judge_config", lambda _: {})
    monkeypatch.setattr(run_eval, "score_package_record", lambda **_: _report())
    config = {"scenario_dir": str(packages), "output_dir": str(tmp_path / "out"), "variant_id": "baseline", "score_only": True, "conversation_log_scoring": {"config_path": str(tmp_path / "judge.json"), "cache_dir": str(tmp_path / "cache")}}
    (tmp_path / "judge.json").write_text("{}", encoding="utf-8")
    run_dir = run_eval.run_output_dir(Path(config["output_dir"]), scenario_id="s1", variant_id="baseline", run_number=1)
    _json(run_dir / "record.json", {"scenario_id": "s1", "status": "success", "event_log": []})
    assert run_eval.run_package_eval(config)
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))["scenario_count"] == 1
    assert run_eval.classify_runtime_failure(error_type="WebSocketError", error_message="no close frame") == "transport_failure"
    assert run_eval.classify_runtime_failure(error_type="RuntimeError", error_message="TTS unavailable") == "tts_failure"
    assert run_eval.classify_runtime_failure(error_type="ValueError", error_message="bad JSON schema") == "judge_or_controller_json_failure"


def test_run_eval_worker_success_and_runtime_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The regular worker pool records both completed and failed local stubs."""
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "s1.yaml").write_text("scenario_id: s1\n", encoding="utf-8")
    monkeypatch.setattr(run_eval, "PackageRuntime", lambda *args, **kwargs: type("Runtime", (), {"tool_definitions": lambda self: []})())
    monkeypatch.setattr(run_eval, "rebuild_summary", lambda _root, summary: summary)
    monkeypatch.setattr(run_eval, "write_summary", lambda root, summary: _json(root / "summary.json", summary))
    monkeypatch.setattr(run_eval, "run_package_chat", lambda **kwargs: [{"scenario_id": kwargs["scenario_id"], "status": "success", "wall_time_sec": 0.1}])
    config = {"scenario_dir": str(packages), "output_dir": str(tmp_path / "out"), "variant_id": "baseline", "api_model": "m", "max_workers": 1}
    assert run_eval.run_package_eval(config, "http://operator")
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))["success_count"] == 1
    monkeypatch.setattr(run_eval, "run_package_chat", lambda **_: (_ for _ in ()).throw(RuntimeError("ASR missing")))
    assert not run_eval.run_package_eval({**config, "output_dir": str(tmp_path / "failed")}, "http://operator")
