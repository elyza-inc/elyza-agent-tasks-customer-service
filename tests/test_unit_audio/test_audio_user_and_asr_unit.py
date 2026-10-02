"""Local AudioUserChannel and ASR-server utility checks with stubs."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from elyza_agent_tasks_customer_service.evaluation.audio import audio_user_channel as channel


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0" * 80)


def test_capture_operator_audio_uses_local_stubs(tmp_path: Path, monkeypatch) -> None:
    """Captured model audio becomes a persisted operator segment without TTS/ASR calls."""

    source = tmp_path / "source.wav"
    _wav(source)
    backend = object.__new__(channel.AudioUserChannel)
    backend.run_dir = tmp_path / "audio_user"
    backend.run_dir.mkdir()
    backend.profiles = [{"profile": "p1", "model": "whisper-1"}, {"profile": "p2", "model": "whisper-1"}]
    backend.turn = 4
    monkeypatch.setattr(channel, "_to_telephony_band", lambda path: None)
    monkeypatch.setattr(channel, "transcribe_audio_profile", lambda **kwargs: {"transcript": f"案内-{kwargs['profile']['profile']}"})

    result = backend.capture_operator_audio(source_wav=source, transcript="案内です", started_ns=1, completed_ns=2)

    assert result["metadata"]["audio_metric_segment"]["speaker"] == "operator"
    assert len(result["metadata"]["audio_metric_events"]) == 2
    assert result["metadata"]["audio_segment_id"] == "audio-operator-0005"
    assert result["metadata"]["diagnostic_user_audio_asr"]["asr"] == {
        "transcript": "案内-p1",
        "profiles": [
            {"profile": "p1", "model": "whisper-1", "transcript": "案内-p1"},
            {"profile": "p2", "model": "whisper-1", "transcript": "案内-p2"},
        ],
    }


def test_audio_channel_signal_profile_requires_object_root(tmp_path: Path, monkeypatch) -> None:
    backend = object.__new__(channel.AudioUserChannel)
    backend.run_dir = tmp_path / "audio_user"
    backend.run_dir.mkdir()
    (backend.run_dir / "audio_metric_config.json").write_text("{}", encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text('{"selected_profile": []}', encoding="utf-8")
    monkeypatch.setattr(channel, "config_path", lambda _name: profile)
    with pytest.raises(ValueError, match="invalid telephony"):
        backend._select_telephony_profile()


def test_capture_operator_audio_profile_index_fallbacks(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.wav"
    _wav(source)
    backend = object.__new__(channel.AudioUserChannel)
    backend.run_dir, backend.profiles, backend.turn = tmp_path / "audio_user", [{}, {}], 0
    backend.run_dir.mkdir()
    monkeypatch.setattr(channel, "_to_telephony_band", lambda _path: None)
    monkeypatch.setattr(channel, "transcribe_audio_profile", lambda **_kwargs: {"transcript": "ok"})
    profiles = backend.capture_operator_audio(source_wav=source, transcript="案内", started_ns=1, completed_ns=2)["metadata"]["diagnostic_user_audio_asr"]["asr"]["profiles"]
    assert profiles == [
        {"profile": "profile-1", "model": "whisper-1", "transcript": "ok"},
        {"profile": "profile-2", "model": "whisper-1", "transcript": "ok"},
    ]


