from __future__ import annotations

import json
import subprocess
import traceback
import wave
from pathlib import Path
from typing import Any

# audio adapter 共通処理

EVALUATION_PACKAGE_DIR = Path(__file__).resolve().parent.parent


def resolve_path(value: str | None) -> Path | None:
    """相対 path を evaluation パッケージ dir 基準で解決する (repo ルート基準ではない)."""
    if value in (None, "", "null"):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (EVALUATION_PACKAGE_DIR / path).resolve()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """append 専用 writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    """Write sorted, indented UTF-8 JSON with a trailing newline."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def ffmpeg_to_mono_pcm16(
    source: Path,
    target: Path,
    *,
    sample_rate: int | str,
    audio_filter: str | None = None,
) -> None:
    """Convert ``source`` to mono PCM16 WAV at ``sample_rate`` with ffmpeg."""
    filter_args = ["-af", audio_filter] if audio_filter else []
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(source),
            *filter_args,
            "-ar", str(sample_rate), "-ac", "1", "-acodec", "pcm_s16le",
            str(target),
        ],
        check=True,
    )


def write_mono_pcm16_wav(path: Path, frames: bytes, *, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(frames)


def format_exception(exc: Exception) -> tuple[str, str]:
    return type(exc).__name__, f"{exc}\n{traceback.format_exc()}"
