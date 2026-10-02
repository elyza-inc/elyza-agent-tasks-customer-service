"""Synthetic WAV, HTTP, and provider-event checks for audio runtime code."""

from __future__ import annotations

import json
import sys
import types
import wave
from pathlib import Path

import pytest

from elyza_agent_tasks_customer_service.evaluation.audio import audio_user_channel as channel
from elyza_agent_tasks_customer_service.evaluation.audio import gemini_live_operator_backend as gemini
from elyza_agent_tasks_customer_service.evaluation.audio import realtime_operator_backend as realtime


def _wav(path: Path, rate: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(b"\0\0" * 80)
    return path


def _events() -> tuple[dict, dict, dict]:
    events = {"events": [
        {"event_id": "u", "event_type": "message", "actor": "user", "content": "090-1234", "audio_segment_id": "user"},
        {"event_id": "i", "event_type": "tool_call", "arguments": {"phone": "0901234"}},
        {"event_id": "w", "event_type": "tool_result", "result": {"phone": "0901234"}},
        {"event_id": "op", "event_type": "message", "actor": "operator", "content": "料金は1200円です", "audio_segment_id": "operator"},
        {"event_id": "d", "event_type": "audio_chunk_played_or_virtual_played", "audio_segment_id": "operator"},
    ]}
    asr = {"observations": [{"status": "accepted", "audio_segment_id": "user", "profile_id": "a", "transcript": "0901234"}]}
    audio = {"audio_segments": [
        {"audio_segment_id": "user", "speaker": "user", "direction": "input", "relative_path": "user.wav"},
        {"audio_segment_id": "operator", "speaker": "operator", "direction": "output", "boundary": "provider", "relative_path": "operator.wav", "sample_range": [0, 80]},
    ]}
    return events, asr, audio


def test_audio_user_channel_convert_finalize_and_config_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = object.__new__(channel.AudioUserChannel)
    backend.run_dir = tmp_path / "audio_user"
    backend.run_dir.mkdir()
    backend.profiles = [{"profile": "one", "model": "whisper-1"}, {"profile": "two", "model": "other"}]
    backend.tts_config = {"model": channel.DEFAULT_CUSTOMER_TTS_MODEL, "voice": "alloy"}
    backend.turn = 0
    def synthesize(**kwargs): _wav(kwargs["output_wav"])
    monkeypatch.setattr(channel, "synthesize_openai_tts", synthesize)
    monkeypatch.setattr(channel, "_to_telephony_band", lambda _: None)
    monkeypatch.setattr(channel, "transcribe_audio_profile", lambda **kw: {"profile": kw["profile"]["profile"], "transcript": "確認"})
    user = backend.convert("raw")
    assert user["asr_text"] == "確認"
    for name, value in {"audio_evidence_manifest.json": {}, "audio_events.json": {}, "audio_metric_config.json": {"m23": {}}}.items():
        (backend.run_dir / name).write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(channel, "config_path", lambda _: tmp_path / "profile.json")
    (tmp_path / "profile.json").write_text(json.dumps({"selected_profile": {"name": "p"}}), encoding="utf-8")
    backend._select_telephony_profile()
    assert backend._load("audio_metric_config.json")["m21"]["selected_profile"]["name"] == "p"
    with pytest.raises(ValueError, match="object root"):
        (backend.run_dir / "bad.json").write_text("[]", encoding="utf-8")
        backend._load("bad.json")


def test_audio_user_channel_preserves_profile_fallbacks_timeouts_and_json_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = object.__new__(channel.AudioUserChannel)
    backend.run_dir, backend.turn = tmp_path / "audio_user", 0
    backend.run_dir.mkdir()
    backend.profiles = [{"profile": "one", "model": "whisper-1"}, {"profile": "two", "model": "other"}, {"profile": "ignored", "model": "ignored"}]
    backend.tts_config = {"model": channel.DEFAULT_CUSTOMER_TTS_MODEL, "voice": "alloy"}
    calls = []
    monkeypatch.setattr(channel, "synthesize_openai_tts", lambda **kwargs: calls.append(kwargs["timeout_sec"]) or _wav(kwargs["output_wav"]))
    monkeypatch.setattr(channel, "_to_telephony_band", lambda _path: None)
    monkeypatch.setattr(channel, "transcribe_audio_profile", lambda **_kwargs: {"transcript": " ok "})
    user = backend.convert("raw")
    assert calls == [900]
    assert user["metadata"]["diagnostic_user_audio_asr"]["asr"]["profiles"] == [
        {"profile": "one", "model": "whisper-1", "transcript": "ok"},
        {"profile": "two", "model": "other", "transcript": "ok"},
        {"profile": "ignored", "model": "ignored", "transcript": "ok"},
    ]
    assert (backend.run_dir / "audio-user-0001.json").read_text(encoding="utf-8").startswith("{\n  ")

    profile = tmp_path / "invalid.json"
    profile.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(channel, "config_path", lambda _name: profile)
    with pytest.raises(ValueError, match="profile"):
        channel.AudioUserChannel(out_dir=tmp_path / "out", tts_cache_dir=tmp_path / "cache")


def test_backend_send_context_audio_text_and_chat_stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = _wav(tmp_path / "input.wav", rate=24000)
    realtime_backend = object.__new__(realtime.RealtimeOperatorBackend)
    realtime_backend.out_dir, realtime_backend._instructions = tmp_path, "base"
    realtime_sent: list[dict] = []
    realtime_backend._send, realtime_backend._wait_for = realtime_sent.append, lambda _: {}
    realtime_backend.send_user_audio(wav)
    realtime_backend.send_user_text("hello")
    realtime_backend.submit_tool_result("c", {"ok": True})
    assert [item["type"] for item in realtime_sent] == ["input_audio_buffer.append", "input_audio_buffer.commit", "conversation.item.create", "conversation.item.create"]

    gemini_backend = object.__new__(gemini.GeminiLiveOperatorBackend)
    gemini_backend.out_dir, gemini_backend._audio_pending, gemini_backend._pending_calls = tmp_path, False, {"c": "tool"}
    gemini_sent: list[dict] = []
    gemini_backend._send = gemini_sent.append
    gemini_backend.send_user_audio(_wav(tmp_path / "gemini.wav", rate=16000))
    gemini_backend.send_user_text("hello")
    gemini_backend.submit_tool_result("c", "value")
    assert gemini_backend._audio_pending and gemini_sent[-1]["toolResponse"]["functionResponses"][0]["name"] == "tool"
    gemini_backend._post_call_model, gemini_backend._post_call_url = "post", "http://post"
    fake_io = types.ModuleType("elyza_agent_tasks_customer_service.evaluation.llm.llm_io")
    fake_io.post_json = object()
    fake_io.chat_with_context_retry = lambda *args, **kwargs: {"choices": []}
    monkeypatch.setitem(sys.modules, fake_io.__name__, fake_io)
    assert gemini_backend.chat_transport("", {"messages": [{"role": "user", "content": "x"}], "tools": []}, {}, 1) == {"choices": []}


