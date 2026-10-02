"""Customer-controller and customer-TTS default model checks."""

from __future__ import annotations

import ast
import inspect
import json
import wave
from pathlib import Path

import pytest

from elyza_agent_tasks_customer_service.evaluation.audio import audio_user_channel as channel
from elyza_agent_tasks_customer_service.evaluation.audio.audio_runtime import (
    DEFAULT_CUSTOMER_TTS_MODEL,
)


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0" * 80)


def test_customer_model_defaults_and_audio_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_eval_path = (
        Path(inspect.getfile(channel)).resolve().parents[1] / "cli" / "run_eval.py"
    )
    tree = ast.parse(run_eval_path.read_text(encoding="utf-8"))

    default_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "DEFAULT_USER_CONTROLLER_MODEL"
            for target in node.targets
        )
    ]
    assert len(default_assignments) == 1
    assert isinstance(default_assignments[0].value, ast.Constant)
    assert default_assignments[0].value.value == "gpt-5.6-luna"

    controller_defaults = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.Or)
        and len(node.values) == 2
        and isinstance(node.values[0], ast.Call)
        and isinstance(node.values[0].func, ast.Attribute)
        and isinstance(node.values[0].func.value, ast.Name)
        and node.values[0].func.value.id == "config"
        and node.values[0].func.attr == "get"
        and len(node.values[0].args) == 1
        and isinstance(node.values[0].args[0], ast.Constant)
        and node.values[0].args[0].value == "user_controller_model"
        and isinstance(node.values[1], ast.Name)
        and node.values[1].id == "DEFAULT_USER_CONTROLLER_MODEL"
    ]
    assert len(controller_defaults) == 1

    profile_path = tmp_path / "asr_profiles.api.json"
    profile_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {"profile": "primary", "model": "whisper-1"},
                    {"profile": "secondary", "model": "whisper-1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(channel, "config_path", lambda _name: profile_path)

    default_channel = channel.AudioUserChannel(
        out_dir=tmp_path / "default-run",
        tts_cache_dir=tmp_path / "default-cache",
    )
    assert default_channel.tts_config["model"] == DEFAULT_CUSTOMER_TTS_MODEL
    assert default_channel.tts_config["voice"] == "alloy"

    custom_channel = channel.AudioUserChannel(
        out_dir=tmp_path / "custom-run",
        tts_cache_dir=tmp_path / "custom-cache",
        tts_model="custom-tts-model",
    )
    assert custom_channel.tts_config["model"] == "custom-tts-model"

    with pytest.raises(ValueError, match="tts_model must be a non-empty string"):
        channel.AudioUserChannel(
            out_dir=tmp_path / "empty-model-run",
            tts_cache_dir=tmp_path / "cache",
            tts_model="",
        )
    with pytest.raises(ValueError, match="tts_model must be a non-empty string"):
        channel.AudioUserChannel(
            out_dir=tmp_path / "non-string-model-run",
            tts_cache_dir=tmp_path / "cache",
            tts_model=1,  # type: ignore[arg-type]
        )

    observed_configs: list[dict[str, object]] = []

    def synthesize(**kwargs: object) -> None:
        observed_configs.append(dict(kwargs["config"]))  # type: ignore[arg-type]
        _wav(kwargs["output_wav"])  # type: ignore[arg-type]

    monkeypatch.setattr(channel, "synthesize_openai_tts", synthesize)
    monkeypatch.setattr(channel, "_to_telephony_band", lambda _path: None)
    monkeypatch.setattr(
        channel,
        "transcribe_audio_profile",
        lambda **kwargs: {
            "profile": kwargs["profile"]["profile"],
            "model": kwargs["profile"]["model"],
            "transcript": "確認しました",
        },
    )

    converted = default_channel.convert("予約内容を確認したいです")
    assert observed_configs[0]["model"] == DEFAULT_CUSTOMER_TTS_MODEL
    assert converted["metadata"]["tts_model"] == DEFAULT_CUSTOMER_TTS_MODEL
    assert converted["metadata"]["tts_voice"] == "alloy"

    evidence = json.loads(
        (
            default_channel.run_dir / "audio-user-0001.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["tts_model"] == DEFAULT_CUSTOMER_TTS_MODEL
    assert evidence["tts_voice"] == "alloy"