"""Artifact builder and contract-materialization unit checks."""

from __future__ import annotations

import json
import wave
from pathlib import Path


from elyza_agent_tasks_customer_service.evaluation.audio import audio_contract_materializer as materializer
from elyza_agent_tasks_customer_service.evaluation.audio.audio_evidence_builder import (
    build_audio_evidence_artifacts,
)
from elyza_agent_tasks_customer_service.evaluation.audio.audio_signal_metrics import score_signal_metrics


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0" * 80)


def test_builder_materializes_measured_segment_and_explicit_gaps(tmp_path: Path) -> None:
    """Measured runtime rows write self-contained scorer artifacts without inferred plans."""

    wav_path = tmp_path / "user.wav"
    _wav(wav_path)
    row = {
        "event_type": "message", "actor": "user", "content": "hello", "clock_id": "clock", "monotonic_ns": 10,
        "metadata": {"diagnostic_user_audio_asr": {"asr": {"profiles": [{"profile": "p1", "transcript": "hello"}]}}, "audio_metric_segment": {"audio_segment_id": "s1", "speaker": "user", "direction": "input", "boundary": "provider", "audio_path": str(wav_path), "clock_id": "clock", "monotonic_range_ns": [1, 10], "complete": True, "chunk_sequence_ids": ["c1"]}},
    }
    refs = build_audio_evidence_artifacts(run_dir=tmp_path, run_id="run", scenario_id="scenario", route="route", source_rows=[row])

    audio = json.loads((tmp_path / refs["audio_manifest"]).read_text())
    config = json.loads((tmp_path / refs["config"]).read_text())
    assert audio["audio_segments"][0]["relative_path"] == "user.wav"
    assert config["evidence_gaps"]["M23"] == ["eligible_signal_segment_boundary_or_threshold_profile_missing"]

    report = score_signal_metrics(
        audio_manifest=tmp_path / refs["audio_manifest"],
        event_log=tmp_path / refs["event_log"],
        config=config,
    )
    assert {key: value["status"] for key, value in report["metrics"].items()} == {key: "N/M" for key in ("M21", "M23")}

    contract = materializer.materialize_audio_measurement_contract(
        audio_events=tmp_path / refs["event_log"],
        audio_evidence_manifest=tmp_path / refs["audio_manifest"],
        output_dir=tmp_path / "materialized",
    )
    assert contract["schema_version"] == "audio_measurement_contract"
    assert (tmp_path / "materialized" / materializer.CONTRACT_FILE).is_file()


