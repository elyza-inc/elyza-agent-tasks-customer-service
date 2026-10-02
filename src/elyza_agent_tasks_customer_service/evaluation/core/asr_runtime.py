from __future__ import annotations

import http.client
import json
import mimetypes
import os
import wave
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


MIN_AUDIO_DURATION_SEC = 0.1
MAX_CHUNK_SEC = 25.0
ASR_RETRY_DELAYS_SEC = (2, 8)


@dataclass(frozen=True)
class AsrProfileConfig:
    profile: str
    endpoint: str
    model: str
    file_field_name: str = "file"
    request_fields: dict[str, str] | None = None
    api_key_env: str | None = None
    timeout_sec: float = 600.0


@dataclass(frozen=True)
class AsrProfileResult:
    profile: str
    model: str
    endpoint: str
    transcript: str
    latency_ms: float
    raw_response: dict[str, Any]


class OpenAICompatibleAsrClient:
    """ASR client for OpenAI-compatible `/v1/audio/transcriptions` endpoints."""

    def __init__(self, config: AsrProfileConfig) -> None:
        if not config.profile.strip():
            raise ValueError("ASR profile must be non-empty")
        if not config.endpoint.strip():
            raise ValueError(f"ASR endpoint is required for profile={config.profile}")
        if not config.model.strip():
            raise ValueError(f"ASR model is required for profile={config.profile}")
        self.config = config

    def transcribe(self, audio_path: Path) -> AsrProfileResult:
        if not audio_path.exists():
            raise FileNotFoundError(f"ASR audio file does not exist: {audio_path}")
        if not audio_path.is_file():
            raise ValueError(f"ASR audio path is not a file: {audio_path}")
        if audio_path.stat().st_size <= 0:
            raise ValueError(f"ASR audio file is empty: {audio_path}")
        fields = dict(self.config.request_fields or {})
        fields.setdefault("model", self.config.model)
        started = perf_counter()
        response = self._post_audio_with_optional_chunking(audio_path=audio_path, fields=fields)
        transcript = response.get("text")
        if not isinstance(transcript, str):
            raise ValueError(
                f"ASR response missing text string for profile={self.config.profile}, "
                f"model={self.config.model}: {response}"
            )
        return AsrProfileResult(
            profile=self.config.profile,
            model=self.config.model,
            endpoint=self.config.endpoint,
            transcript=transcript.strip(),
            latency_ms=round((perf_counter() - started) * 1000, 3),
            raw_response=response,
        )

    def _post_audio_with_optional_chunking(self, *, audio_path: Path, fields: dict[str, str]) -> dict[str, Any]:
        duration = wav_duration_sec(audio_path)
        if duration <= MAX_CHUNK_SEC:
            return self._post_audio(audio_path=audio_path, fields=fields)
        if not is_pcm16_wav(audio_path):
            return self._post_audio(audio_path=audio_path, fields=fields)
        transcripts: list[str] = []
        raw_chunks: list[dict[str, Any]] = []
        with TemporaryDirectory(prefix="asr-chunks-") as tmp:
            chunk_paths = split_pcm16_wav(audio_path, Path(tmp), max_chunk_sec=MAX_CHUNK_SEC)
            for chunk_index, chunk_path in enumerate(chunk_paths):
                response = self._post_audio(audio_path=chunk_path, fields=fields)
                text = response.get("text")
                if not isinstance(text, str):
                    raise ValueError(
                        f"ASR chunk response missing text string for profile={self.config.profile}, "
                        f"chunk={chunk_path}: {response}"
                    )
                transcripts.append(text.strip())
                raw_chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "chunk_path": str(chunk_path),
                        "duration_sec": wav_duration_sec(chunk_path),
                        "transcript_empty": not text.strip(),
                        "response": response,
                    }
                )
        return {
            "text": " ".join(text for text in transcripts if text),
            "chunked": True,
            "source_audio": str(audio_path),
            "source_duration_sec": duration,
            "chunk_count": len(raw_chunks),
            "empty_chunk_indexes": [
                row["chunk_index"] for row in raw_chunks if row["transcript_empty"]
            ],
            "chunks": raw_chunks,
        }

    def _post_audio(self, *, audio_path: Path, fields: dict[str, str]) -> dict[str, Any]:
        if wav_duration_sec(audio_path) < MIN_AUDIO_DURATION_SEC:
            return {"text": "", "audio_too_short": True}
        headers: dict[str, str] = {}
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env, "").strip()
            if not api_key:
                raise ValueError(
                    f"ASR API key environment variable is not set: {self.config.api_key_env} "
                    f"for profile={self.config.profile}"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        return post_multipart_json(
            url=self.config.endpoint.rstrip("/") + "/v1/audio/transcriptions",
            fields=fields,
            file_field_name=self.config.file_field_name,
            file_path=audio_path,
            timeout_sec=self.config.timeout_sec,
            headers=headers,
        )


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        if rate <= 0:
            raise ValueError(f"wav sample rate must be positive: {path}")
        return wav.getnframes() / rate


def is_pcm16_wav(path: Path) -> bool:
    with wave.open(str(path), "rb") as wav:
        return wav.getsampwidth() == 2 and wav.getnchannels() == 1


def split_pcm16_wav(path: Path, out_dir: Path, *, max_chunk_sec: float) -> list[Path]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        frame_count = wav.getnframes()
        if channels != 1 or sample_width != 2:
            raise ValueError(f"split_pcm16_wav supports mono PCM16 only: {path}")
        frames_per_chunk = int(frame_rate * max_chunk_sec)
        if frames_per_chunk <= 0:
            raise ValueError(f"max_chunk_sec too small: {max_chunk_sec}")
        paths: list[Path] = []
        index = 0
        while wav.tell() < frame_count:
            frames = wav.readframes(frames_per_chunk)
            if not frames:
                break
            chunk_path = out_dir / f"{path.stem}.chunk_{index:03d}.wav"
            with wave.open(str(chunk_path), "wb") as chunk:
                chunk.setnchannels(channels)
                chunk.setsampwidth(sample_width)
                chunk.setframerate(frame_rate)
                chunk.writeframes(frames)
            paths.append(chunk_path)
            index += 1
    if not paths:
        raise ValueError(f"no ASR chunks created for: {path}")
    return paths


def post_multipart_json(
    *,
    url: str,
    fields: dict[str, str],
    file_field_name: str,
    file_path: Path,
    timeout_sec: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    boundary = "----asr-" + uuid.uuid4().hex
    body = _multipart_body(boundary=boundary, fields=fields, file_field_name=file_field_name, file_path=file_path)
    request_headers = dict(headers or {})
    request_headers.update(
        {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        }
    )
    request = Request(
        url,
        data=body,
        headers=request_headers,
        method="POST",
    )
    for retry_delay_sec in (*ASR_RETRY_DELAYS_SEC, None):
        try:
            with urlopen(request, timeout=timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code == 400 and "audio_too_short" in detail:
                return {"text": "", "audio_too_short": True}
            if not 500 <= error.code < 600 or retry_delay_sec is None:
                raise RuntimeError(f"ASR request failed {error.code} for {url}: {detail}") from error
        except TimeoutError as error:
            if "read operation timed out" not in str(error).lower() or retry_delay_sec is None:
                raise
        except (http.client.IncompleteRead, ConnectionError):
            # A connection dropped mid-response is resent like a 5xx.
            if retry_delay_sec is None:
                raise
        sleep(retry_delay_sec)
    if not isinstance(payload, dict):
        raise ValueError(f"ASR response must be a JSON object for {url}: {payload}")
    return payload


def _multipart_body(*, boundary: str, fields: dict[str, str], file_field_name: str, file_path: Path) -> bytes:
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field_name}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks)
