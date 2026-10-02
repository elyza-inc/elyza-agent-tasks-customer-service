from __future__ import annotations

import wave

import pytest

from elyza_agent_tasks_customer_service.evaluation.core import (
    asr_runtime,
)


def test_asr_runtime_chunking_and_response_boundaries(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "long.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(10)
        wav.writeframes(b"\0\0" * 30)
    assert asr_runtime.wav_duration_sec(wav_path) == 3
    assert len(asr_runtime.split_pcm16_wav(wav_path, tmp_path, max_chunk_sec=1)) == 3
    client = asr_runtime.OpenAICompatibleAsrClient(asr_runtime.AsrProfileConfig("a", "http://x", "m"))
    monkeypatch.setattr(client, "_post_audio", lambda **_: {"text": "ok"})
    monkeypatch.setattr(asr_runtime, "MAX_CHUNK_SEC", 1)
    assert client._post_audio_with_optional_chunking(audio_path=wav_path, fields={})["chunk_count"] == 3
    monkeypatch.setattr(asr_runtime, "post_multipart_json", lambda **kwargs: {"text": "done", "headers": kwargs["headers"]})
    monkeypatch.setenv("ASR_TOKEN", " token ")
    client = asr_runtime.OpenAICompatibleAsrClient(asr_runtime.AsrProfileConfig("a", "http://x/", "m", api_key_env="ASR_TOKEN"))
    assert client._post_audio(audio_path=wav_path, fields={})["text"] == "done"
    assert b'filename="long.wav"' in asr_runtime._multipart_body(boundary="b", fields={"model": "m"}, file_field_name="file", file_path=wav_path)


