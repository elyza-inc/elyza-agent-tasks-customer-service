"""Audio conversion and TTS helpers for evaluator runs."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.audio.omni_audio_common import resolve_path
from elyza_agent_tasks_customer_service.evaluation.llm.llm_io import is_transient_http_code


DEFAULT_CUSTOMER_TTS_MODEL = "gpt-4o-mini-tts-2025-12-15"


def require_file(path: Path | None, label: str) -> Path:
    if path is None or not path.exists():
        raise FileNotFoundError(f"{label} is required and must exist: {path}")
    return path


def require_dir(path: Path | None, label: str) -> Path:
    if path is None or not path.is_dir():
        raise FileNotFoundError(f"{label} is required and must be a directory: {path}")
    return path


def synthesize_openai_tts(*, text: str, output_wav: Path, config: dict[str, Any], timeout_sec: float) -> None:
    """Synthesize and deterministically cache OpenAI-compatible WAV audio."""

    if not text.strip():
        raise ValueError(f"cannot synthesize empty text for {output_wav}")
    model = str(config.get("model") or DEFAULT_CUSTOMER_TTS_MODEL)
    voice = str(config.get("voice") or "alloy")
    cache_key = hashlib.sha256(
        json.dumps([text, model, voice], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_dir = resolve_path(config.get("cache_dir")) or output_wav.parent / ".openai_tts_cache"
    cache_wav = cache_dir / f"{cache_key}.wav"
    if not cache_wav.is_file() or cache_wav.stat().st_size == 0:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI TTS")
        payload = json.dumps(
            {
                "model": model,
                "voice": voice,
                "input": text,
                "response_format": "wav",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/v1/audio/speech",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        # Same resend rule as llm_io.post_json: transient HTTP and connection errors, 3 attempts.
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                    audio_bytes = response.read()
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if not is_transient_http_code(exc.code) or attempt == 3:
                    raise RuntimeError(f"OpenAI TTS failed with HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, http.client.IncompleteRead, ConnectionError):
                if attempt == 3:
                    raise
            time.sleep(min(2.0 * attempt, 5.0))
        if not audio_bytes:
            raise RuntimeError("OpenAI TTS returned empty audio")
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_wav = cache_dir / f".{cache_key}.{os.getpid()}.tmp"
        remuxed_wav = cache_dir / f".{cache_key}.{os.getpid()}.wav"
        temporary_wav.write_bytes(audio_bytes)
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to convert audio to PCM16")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(temporary_wav),
                "-acodec",
                "pcm_s16le",
                str(remuxed_wav),
            ],
            check=True,
        )
        remuxed_wav.replace(cache_wav)
        temporary_wav.unlink(missing_ok=True)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if output_wav != cache_wav:
        shutil.copyfile(cache_wav, output_wav)
    if not output_wav.is_file() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"OpenAI TTS did not create output wav: {output_wav}")


