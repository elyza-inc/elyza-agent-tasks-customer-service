import wave
from pathlib import Path
from io import BytesIO
from urllib.error import HTTPError

import pytest

from elyza_agent_tasks_customer_service.evaluation.core import (
    asr_runtime,
)


def _write_wav(path: Path, *, frames: int = 1600, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\0\0" * frames)


def test_asr_runtime_client_handles_short_audio(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    _write_wav(path, frames=1)
    client = asr_runtime.OpenAICompatibleAsrClient(asr_runtime.AsrProfileConfig("one", "http://one", "m"))
    assert client.transcribe(path).raw_response["audio_too_short"] is True


def test_asr_runtime_exact_boundaries_and_retry_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ASR cutoffs, retry schedule, and response classification observable."""
    wav_path = tmp_path / "exact.wav"
    _write_wav(wav_path, frames=1600)
    client = asr_runtime.OpenAICompatibleAsrClient(asr_runtime.AsrProfileConfig("one", "http://one", "m"))
    multipart = asr_runtime.post_multipart_json
    monkeypatch.setattr(asr_runtime, "post_multipart_json", lambda **_kwargs: {"text": "exact"})
    assert client._post_audio(audio_path=wav_path, fields={}) == {"text": "exact"}
    monkeypatch.setattr(asr_runtime, "post_multipart_json", multipart)
    (tmp_path / "empty.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        client.transcribe(tmp_path / "empty.wav")
    with pytest.raises(ValueError, match="too small"):
        asr_runtime.split_pcm16_wav(wav_path, tmp_path, max_chunk_sec=0.00001)

    delays: list[float] = []
    attempts = [
        HTTPError("http://one", 500, "down", {}, BytesIO(b"retry")),
        HTTPError("http://one", 500, "down", {}, BytesIO(b"retry")),
    ]
    monkeypatch.setattr(asr_runtime, "sleep", delays.append)
    monkeypatch.setattr(asr_runtime, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(attempts.pop(0)) if attempts else _Response())
    assert asr_runtime.post_multipart_json(url="http://one", fields={}, file_field_name="file", file_path=wav_path, timeout_sec=600) == {"text": "ok"}
    assert delays == [2, 8]

    short_error = HTTPError("http://one", 400, "bad", {}, BytesIO(b"audio_too_short"))
    monkeypatch.setattr(asr_runtime, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(short_error))
    assert asr_runtime.post_multipart_json(url="http://one", fields={}, file_field_name="file", file_path=wav_path, timeout_sec=600)["audio_too_short"] is True
    server_error = HTTPError("http://one", 600, "bad", {}, BytesIO(b"bad"))
    monkeypatch.setattr(asr_runtime, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(server_error))
    with pytest.raises(RuntimeError, match="600"):
        asr_runtime.post_multipart_json(url="http://one", fields={}, file_field_name="file", file_path=wav_path, timeout_sec=600)


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return b'{"text":"ok"}'


