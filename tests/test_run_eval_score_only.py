import json


def test_score_only_rescores_existing_record_without_runtime_config(tmp_path, monkeypatch):
    from elyza_agent_tasks_customer_service.evaluation.cli import run_eval
    from elyza_agent_tasks_customer_service.evaluation.contracts.metric_inventory import (
        ACTIVE_METRIC_IDS,
    )

    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "existing.yaml").write_text("scenario_id: existing\n", encoding="utf-8")
    (scenario_dir / "missing.yaml").write_text("scenario_id: missing\n", encoding="utf-8")
    run_dir = tmp_path / "output" / "existing" / "baseline" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "record.json").write_text(
        json.dumps({"scenario_id": "existing", "status": "success", "event_log": []}),
        encoding="utf-8",
    )
    audio_dir = run_dir / "audio_user"
    audio_dir.mkdir()
    (audio_dir / "audio_metric_results.json").write_text(
        json.dumps(
            {
                "schema_version": "fixture-audio-v1",
                "metrics": {
                    metric_id: {"status": "passed", "reason": "fixture"}
                    for metric_id in ("M21", "M23")
                },
            }
        ),
        encoding="utf-8",
    )
    judge_config_path = tmp_path / "judge.yaml"
    judge_config_path.write_text("judge: fixture\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(run_eval, "load_judge_config", lambda path: {"fixture": str(path)})

    def score_fixture(**kwargs):
        calls.append(kwargs)
        return {
            "scenario_id": "existing",
            "metric_results": [
                {
                    "metric_instance_id": f"{metric_id}-fixture",
                    "metric_id": metric_id,
                    "status": "N/M",
                    "reason": "fixture",
                    "violations": [],
                }
                for metric_id in ACTIVE_METRIC_IDS
            ],
            "violations": [],
            "diagnostics": {},
        }

    monkeypatch.setattr(
        run_eval,
        "score_package_record",
        score_fixture,
    )
    monkeypatch.setattr(
        run_eval,
        "run_package_chat",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    assert run_eval.run_package_eval(
        {
            "scenario_dir": str(scenario_dir),
            "output_dir": str(tmp_path / "output"),
            "variant_id": "baseline",
            "io_mode": "audio-text",
            "score_only": True,
            "conversation_log_scoring": {
                "config_path": str(judge_config_path),
                "cache_dir": str(tmp_path / "judge-cache"),
            },
        },
    )

    assert (run_dir / "score.json").is_file()
    assert calls[0]["record"]["scenario_id"] == "existing"
    assert calls[0]["mode"] == "audio-text"
    score = json.loads((run_dir / "score.json").read_text(encoding="utf-8"))
    assert next(row for row in score["metric_results"] if row["metric_id"] == "M21")["status"] == "pass"
    summary = json.loads(
        (tmp_path / "output" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["scored_count"] == 1
    assert summary["skipped_count"] == 1
