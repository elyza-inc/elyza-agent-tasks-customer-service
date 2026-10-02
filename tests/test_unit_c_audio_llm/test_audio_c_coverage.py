"""Focused no-network checks for untested audio validation paths."""

from __future__ import annotations

import base64
from pathlib import Path
import wave

import pytest

from elyza_agent_tasks_customer_service.evaluation.audio import (
    audio_value_metrics as deterministic,
)
from elyza_agent_tasks_customer_service.evaluation.audio import omni_operator_backend as omni
from elyza_agent_tasks_customer_service.evaluation.audio import audio_user_channel as user_channel
from elyza_agent_tasks_customer_service.evaluation.audio import gemini_live_operator_backend as gemini


def test_deterministic_full_scoring_path() -> None:
    scenario = {
        "contracts": {
            "m20": {"important_values": [{"slot_id": "phone", "value": "09012", "value_type": "phone"}]},
            "m05": {"disclosure": "返金は12時まで", "deadline_tool_id": "save"},
        }
    }
    profiles = [{"transcript": "09012"}, {"transcript": "09012"}]
    record = {"run_id": "r", "event_log": [
        {"seq": 1, "event_type": "message", "actor": "user", "content": "電話090-12"},
        {"seq": 2, "event_type": "message", "actor": "operator", "content": "09012。返金は12時まで", "metadata": {"diagnostic_user_audio_asr": {"asr": {"profiles": profiles}}}},
        {"seq": 3, "event_type": "tool_call", "arguments": {"phone": "09012"}, "argument_provenance": {"phone": {"source_kind": "customer_utterance"}}},
    ]}
    results = deterministic.score_value_metrics(record=record, scenario=scenario)
    assert results["metrics"]["M20"]["status"] == "passed"
    assert results["metrics"]["M26"]["status"] == "passed"


def test_deterministic_rejects_unagreed_and_invalid_audio_evidence() -> None:
    slot = {"slot_id": "phone", "value": "09012", "value_type": "phone"}
    scenario = {"contracts": {"m20": {"important_values": [slot]}, "m05": {"disclosure": "返金は12時まで"}}}
    profiles = [{"transcript": "09012"}, {"transcript": "09013"}]
    record = {"event_log": [
        {"seq": 1, "event_type": "message", "actor": "user", "content": "09012"},
        {"seq": 2, "event_type": "message", "actor": "operator", "metadata": {"diagnostic_user_audio_asr": {"asr": {"profiles": profiles}}}},
        {"seq": 3, "event_type": "tool_call", "arguments": {"phone": "0901"}, "argument_provenance": {"phone": {"source_kind": "customer_utterance"}}},
    ]}
    result = deterministic.score_value_metrics(record=record, scenario=scenario)
    assert result["metrics"]["M20"]["status"] == "failed"
    assert result["metrics"]["M22"]["status"] == "failed"
    assert result["metrics"]["M26"]["status"] == "N/A"
    assert deterministic._typed_profile_evidence({"kind": "digits", "normalized": "09012"}, ["09012", "09012"]) == [{"kind": "audio_asr", "correct": True}]
    assert deterministic._typed_profile_evidence({"kind": "digits", "normalized": "09012"}, ["09012"]) == []
    assert deterministic._same_shape({"kind": "kana", "normalized": "ヤマダ"}, "ヤマダ")
    assert not deterministic._same_shape({"kind": "kana", "normalized": "ヤマダ"}, "YAMADA")


def test_deterministic_customer_contract_and_boundary_helpers() -> None:
    assert deterministic._slots({"contracts": {"m20": {"important_values": [{"name": "phone", "value": "09012"}]}}})[0]["kind"] == "digits"
    assert deterministic._m20_contract_source({"contracts": {"m20": {"important_values": []}}}) == "scenario.contracts.m20.important_values"
    assert deterministic._events({"event_log": [{"seq": 2}, {"seq": 1}]})[0]["seq"] == 1
    with pytest.raises(ValueError, match="object array"):
        deterministic._events({"event_log": [{} , "bad"]})
    assert deterministic._tokens("digits", "1 A 12") == {"12"}
    assert deterministic._agreed_contains("返金", ["返金", "返金します"])
    assert not deterministic._agreed_contains("返金", ["返金"])


def test_deterministic_guard_and_kana_boundaries() -> None:
    assert deterministic._slots({"contracts": {"m20": {"important_values": {"bad": 1}}}}) == []
    with pytest.raises(ValueError, match="requires name"):
        deterministic._slot({"name": "", "value": "x"})
    assert deterministic._operator_evidence({"kind": "digits", "normalized": "09012"}, [{"actor": "user", "profiles": ["09013", "09013"]}, {"actor": "operator", "profiles": ["09012", "09012"]}]) == [{"kind": "audio_asr", "correct": True}]
    assert deterministic._tool_evidence({"kind": "digits", "normalized": "09012"}, [{"event_type": "tool_call", "arguments": {"phone": "bad"}, "argument_provenance": {"phone": {"source_kind": "customer_utterance"}}}]) == []
    assert deterministic._tool_evidence({"kind": "digits", "normalized": "09012"}, [{"event_type": "tool_call", "arguments": {"phone": "09012"}, "argument_provenance": {"phone": "unlinked"}}]) == []
    assert deterministic._same_shape({"kind": "kana", "normalized": "ヤマダ"}, "タナカ")
    assert deterministic._same_shape({"kind": "kana", "normalized": "ヶ"}, "ヶ")
    assert deterministic._typed_profile_evidence({"kind": "kana", "normalized": "ヤマダ"}, ["ヤマダ", "ヤマダ"]) == [{"kind": "audio_asr", "correct": True}]
    assert deterministic._kind("name", "abc") == "kana"
    assert deterministic._kind(None, "ヶ") == "kana"
    assert deterministic._kind(None, "ァ") == "kana"
    invalid = deterministic.score_value_metrics(record={"event_log": []}, scenario={"contracts": {"m05": "invalid"}})
    assert invalid["metrics"]["M22"]["status"] == "N/M"
    scenario = {"contracts": {"m20": {"important_values": [{"slot_id": "phone", "value": "09012"}]}}}
    record = {"event_log": [
        {"seq": 1, "event_type": "message", "actor": "user", "content": "09012"},
        {"seq": 2, "event_type": "message", "actor": "operator", "metadata": {"diagnostic_user_audio_asr": {"asr": {"profiles": [{"transcript": "09012"}, {"transcript": "09012"}]}}}},
    ]}
    assert deterministic.score_value_metrics(record=record, scenario=scenario)["metrics"]["M25"]["status"] == "passed"
    assert deterministic._score_m25([{"slot_id": "phone", "kind": "digits", "normalized": "09012"}], [{"seq": 1, "actor": "operator", "source": "09012", "profiles": ["09012", "09012"]}, {"seq": 2, "actor": "operator", "source": "09012", "profiles": ["09012", "09012"]}])["status"] == "N/A"
    assert deterministic._typed_profile_evidence({"kind": "text", "normalized": "x"}, ["12", "12"]) == []


def test_audio_channel_init_and_omni_audio_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = tmp_path / "profiles.json"
    profile.write_text('{"profiles":[{"model":"whisper-1","profile":"p"}]}', encoding="utf-8")
    monkeypatch.setattr(user_channel, "config_path", lambda _name: profile)
    channel = user_channel.AudioUserChannel(out_dir=tmp_path, tts_cache_dir=tmp_path / "cache")
    assert channel.profiles == [{"model": "whisper-1", "profile": "p"}]
    converted = gemini._json_schema_to_gemini(
        {"type": "object", "properties": {"x": {"type": "string"}}}
    )
    assert converted == {"type": "OBJECT", "properties": {"x": {"type": "STRING"}}}

    wav = tmp_path / "reply.wav"
    with wave.open(str(wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0")
    audio = wav.read_bytes()
    response = {
        "choices": [{"message": {"role": "assistant", "content": "ok", "audio": {"data": base64.b64encode(audio).decode()}}}]
    }
    message, extracted = omni._extract_omni_response(response)
    assert message["content"] == "ok" and extracted == audio
    backend = object.__new__(omni.OmniOperatorBackend)
    backend.out_dir, backend._turn = tmp_path, 0
    assert backend._persist_audio(audio).is_file()
