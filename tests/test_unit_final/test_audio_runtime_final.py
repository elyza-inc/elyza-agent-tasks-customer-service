"""Stubbed audio boundaries and synthetic-WAV coverage checks."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from elyza_agent_tasks_customer_service.evaluation.audio import audio_user_channel as channel


def _wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0" * 80)
    return path


def test_audio_channel_telephony_and_finalize_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Channel output uses synthetic audio and local scorer/materializer stubs."""
    profile = tmp_path / "profiles.json"
    profile.write_text(json.dumps({"profiles": [{"model": "whisper-1", "profile": "p1"}, {"model": "other", "profile": "p2"}]}), encoding="utf-8")
    monkeypatch.setattr(channel, "config_path", lambda _: profile)
    monkeypatch.setattr(channel, "synthesize_openai_tts", lambda **kwargs: _wav(kwargs["output_wav"]))
    monkeypatch.setattr(channel, "_to_telephony_band", lambda _: None)
    monkeypatch.setattr(channel, "transcribe_audio_profile", lambda **kwargs: {"profile": kwargs["profile"]["profile"], "transcript": "確認"})
    backend = channel.AudioUserChannel(out_dir=tmp_path, tts_cache_dir=tmp_path / "cache")
    converted = backend.convert("確認")
    assert converted["asr_text"] == "確認" and converted["metadata"]["audio_metric_segment"]["speaker"] == "user"
    _wav(tmp_path / "operator.wav")
    operator = backend.capture_operator_audio(source_wav=tmp_path / "operator.wav", transcript="案内", started_ns=1, completed_ns=2)
    assert operator["metadata"]["audio_metric_segment"]["speaker"] == "operator"
    monkeypatch.setattr(channel, "build_audio_evidence_artifacts", lambda **_: {"audio_manifest": "b.json", "event_log": "e.json", "config": "c.json"})
    monkeypatch.setattr(channel, "extract_measured_audio_evidence", lambda **_: ([], [], {}))
    monkeypatch.setattr(channel, "materialize_audio_measurement_contract", lambda **_: {"ok": True})
    monkeypatch.setattr(backend, "_select_telephony_profile", lambda: None)
    monkeypatch.setattr(backend, "_load", lambda _name: {})
    monkeypatch.setattr(channel, "score_signal_metrics", lambda **_: {"metrics": {}})
    assert backend.finalize(run_id="r", scenario_id="s", source_rows=[]) == {"metrics": {}}


