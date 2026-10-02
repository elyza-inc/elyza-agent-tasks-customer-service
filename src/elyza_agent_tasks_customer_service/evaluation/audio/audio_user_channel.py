"""TTS/ASR customer channel and M21/M23 audio evidence scoring for audio runs."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.config_paths import config_path
from elyza_agent_tasks_customer_service.evaluation.audio.audio_contract_materializer import materialize_audio_measurement_contract
from elyza_agent_tasks_customer_service.evaluation.audio.audio_evidence_builder import (
    build_audio_evidence_artifacts,
    extract_measured_audio_evidence,
    transcribe_audio_profile,
)
from elyza_agent_tasks_customer_service.evaluation.audio.audio_signal_metrics import score_signal_metrics
from elyza_agent_tasks_customer_service.evaluation.audio.audio_runtime import (
    DEFAULT_CUSTOMER_TTS_MODEL,
    synthesize_openai_tts,
)
from elyza_agent_tasks_customer_service.evaluation.audio.omni_audio_common import ffmpeg_to_mono_pcm16


def _to_telephony_band(wav_path: Path) -> None:
    """Convert one WAV in place to the simulated telephone channel (8kHz narrowband)."""

    narrow = wav_path.with_suffix(".telephony.tmp.wav")
    ffmpeg_to_mono_pcm16(
        wav_path,
        narrow,
        sample_rate=os.environ.get("AUDIO_CHANNEL_SAMPLE_RATE", "8000"),
        audio_filter="highpass=f=300,lowpass=f=3400",
    )
    narrow.replace(wav_path)


_DIGIT_READINGS = {"0": "ゼロ", "1": "イチ", "2": "ニー", "3": "サン", "4": "ヨン", "5": "ゴー", "6": "ロク", "7": "ナナ", "8": "ハチ", "9": "キュー"}
_SEP = r"[-\u2010-\u2015\u2212ー－]"


def _spell_groups(value: str) -> str:
    groups = []
    for part in (p for p in re.split(_SEP, value) if p):
        while len(part) > 4:
            groups.append(part[:4])
            part = part[4:]
        groups.append(part)
    return "の、".join("".join(_DIGIT_READINGS[d] for d in group) for group in groups)


def spell_digit_strings_for_tts(text: str) -> str:
    """Read phone numbers and digit strings of five or more digits digit by digit, in groups of up to four.

    Amounts (followed by 円) and dates written as YYYY-MM-DD keep their usual reading.
    """

    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    def run(match: re.Match[str]) -> str:
        value = match.group(0)
        if re.fullmatch(rf"\d{{4}}{_SEP}\d{{1,2}}{_SEP}\d{{1,2}}", value):
            return value
        return _spell_groups(value)

    return re.sub(rf"(?<![\d,.])\d(?:{_SEP}?\d){{4,}}(?![\d,.]|\s*円)", run, text)


class AudioUserChannel:
    """Convert customer text to WAV and expose only its ASR transcript upstream."""

    def __init__(
        self,
        *,
        out_dir: Path,
        tts_cache_dir: Path,
        tts_model: str = DEFAULT_CUSTOMER_TTS_MODEL,
    ) -> None:
        if not isinstance(tts_model, str) or not tts_model:
            raise ValueError("tts_model must be a non-empty string")
        self.run_dir = out_dir / "audio_user"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        profile_path = config_path("asr_profiles.api.json")
        root = json.loads(profile_path.read_text(encoding="utf-8"))
        profiles = root.get("profiles") if isinstance(root, dict) else None
        if not isinstance(profiles, list) or not profiles or not isinstance(profiles[0], dict):
            raise ValueError(f"ASR profile list is empty: {profile_path}")
        if str(profiles[0].get("model") or "") != "whisper-1":
            raise ValueError("the first ASR profile must use whisper-1")
        self.profiles = [row for row in profiles if isinstance(row, dict)][:2]
        cache_dir = tts_cache_dir.expanduser().resolve()
        if cache_dir == self.run_dir.resolve() or self.run_dir.resolve() in cache_dir.parents:
            raise ValueError("tts_cache_dir must be outside out-dir/audio_user")
        self.tts_config = {
            "provider": "openai",
            "model": tts_model,
            "voice": "alloy",
            "cache_dir": str(cache_dir),
        }
        self.turn = 0

    def _transcribe_profiles(self, wav_path: Path) -> list[dict[str, str]]:
        asr_results = [
            transcribe_audio_profile(wav_path=wav_path, profile=profile)
            for profile in self.profiles
        ]
        return [
            {
                "profile": str(result.get("profile") or profile.get("profile") or f"profile-{index}"),
                "model": str(result.get("model") or profile.get("model") or "whisper-1"),
                "transcript": str(result.get("transcript") or "").strip(),
            }
            for index, (result, profile) in enumerate(zip(asr_results, self.profiles), start=1)
        ]

    def convert(self, text: str) -> dict[str, Any]:
        self.turn += 1
        segment_id = f"audio-user-{self.turn:04d}"
        wav_path = self.run_dir / f"{segment_id}.wav"
        started_ns = time.monotonic_ns()
        spoken_text = spell_digit_strings_for_tts(text)
        synthesize_openai_tts(
            text=spoken_text,
            output_wav=wav_path,
            config=self.tts_config,
            timeout_sec=900,
        )
        _to_telephony_band(wav_path)
        completed_ns = time.monotonic_ns()
        profile_results = self._transcribe_profiles(wav_path)
        asr_text = profile_results[0]["transcript"]
        if not asr_text:
            raise ValueError(f"whisper-1 returned an empty transcript: {wav_path}")
        evidence = {
            "wav": wav_path.name,
            "tts_text": text,
            "tts_input_text": spoken_text,
            "asr_text": asr_text,
            "tts_model": self.tts_config["model"],
            "tts_voice": self.tts_config["voice"],
        }
        (self.run_dir / f"{segment_id}.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "audio_segment_id": segment_id,
            "tts_text": text,
            "tts_input_text": spoken_text,
            "asr_text": asr_text,
            "tts_model": self.tts_config["model"],
            "tts_voice": self.tts_config["voice"],
            "operator_input_text": asr_text,
            "wav": wav_path.name,
            "diagnostic_user_audio_asr": {
                "asr": {
                    "transcript": asr_text,
                    "profiles": profile_results,
                }
            },
            "audio_metric_segment": {
                "audio_segment_id": segment_id,
                "speaker": "user",
                "direction": "input",
                "boundary": "provider",
                "audio_path": str(wav_path.resolve()),
                "clock_id": "python_monotonic_ns",
                "monotonic_range_ns": [started_ns, completed_ns],
                "complete": True,
                "chunk_sequence_ids": [segment_id],
            },
            "audio_metric_events": [
                {
                    "event_type": "audio_chunk_received",
                    "clock_id": "python_monotonic_ns",
                    "monotonic_ns": completed_ns,
                    "audio_segment_id": segment_id,
                    "chunk_sequence_id": segment_id,
                }
            ],
        }
        return {
            "asr_text": asr_text,
            "wav_path": wav_path,
            "metadata": metadata,
        }

    def capture_operator_audio(
        self,
        *,
        source_wav: Path,
        transcript: str,
        started_ns: int,
        completed_ns: int,
    ) -> dict[str, Any]:
        """Persist realtime model audio as an operator output segment."""

        self.turn += 1
        segment_id = f"audio-operator-{self.turn:04d}"
        wav_path = self.run_dir / f"{segment_id}.wav"
        shutil.copy2(source_wav, wav_path)
        _to_telephony_band(wav_path)
        evidence = {
            "wav": wav_path.name,
            "operator_text": transcript,
            "source": "realtime_model_audio",
        }
        (self.run_dir / f"{segment_id}.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        measured_profiles = self._transcribe_profiles(wav_path)
        return {
            "metadata": {
                "audio_segment_id": segment_id,
                "operator_text": transcript,
                "wav": wav_path.name,
                "audio_source": "realtime_model_audio",
                "model_output_transcript": transcript,
                "diagnostic_user_audio_asr": {
                    "asr": {
                        "transcript": measured_profiles[0]["transcript"],
                        "profiles": measured_profiles,
                    }
                },
                "audio_metric_segment": {
                    "audio_segment_id": segment_id,
                    "speaker": "operator",
                    "direction": "output",
                    "boundary": "provider",
                    "audio_path": str(wav_path.resolve()),
                    "clock_id": "python_monotonic_ns",
                    "monotonic_range_ns": [started_ns, completed_ns],
                    "complete": True,
                    "chunk_sequence_ids": [segment_id],
                },
                "audio_metric_events": [
                    {
                        "event_type": "response_started",
                        "clock_id": "python_monotonic_ns",
                        "monotonic_ns": started_ns,
                        "audio_segment_id": segment_id,
                        "chunk_sequence_id": segment_id,
                        "control": "yield_and_respond",
                    },
                    {
                        "event_type": "audio_chunk_played_or_virtual_played",
                        "clock_id": "python_monotonic_ns",
                        "monotonic_ns": completed_ns,
                        "audio_segment_id": segment_id,
                        "chunk_sequence_id": segment_id,
                    },
                ],
            }
        }

    def finalize(
        self,
        *,
        run_id: str,
        scenario_id: str,
        source_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        segments, events, _ = extract_measured_audio_evidence(
            run_dir=self.run_dir,
            rows=source_rows,
        )
        contract = materialize_audio_measurement_contract(
            audio_events={"events": events},
            audio_evidence_manifest={"audio_segments": segments},
            output_dir=self.run_dir,
        )
        build_audio_evidence_artifacts(
            run_dir=self.run_dir,
            run_id=run_id,
            scenario_id=scenario_id,
            route="audio-user",
            source_rows=source_rows,
            scenario_contract=contract,
        )
        self._select_telephony_profile()
        result = score_signal_metrics(
            audio_manifest=self._load("audio_evidence_manifest.json"),
            event_log=self._load("audio_events.json"),
            config=self._load("audio_metric_config.json"),
            evidence_root=self.run_dir,
        )
        (self.run_dir / "audio_metric_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    def _select_telephony_profile(self) -> None:
        """Select the telephony channel profile for both M21 and M23."""

        run_config_path = self.run_dir / "audio_metric_config.json"
        config = self._load(run_config_path.name)
        profile_path = config_path("audio_channel_profile.telephony.json")
        profile_config = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(profile_config, dict) or not isinstance(
            profile_config.get("selected_profile"), dict
        ):
            raise ValueError(f"invalid telephony audio channel profile: {profile_path}")
        config["m23"] = profile_config
        config["m21"] = {"selected_profile": profile_config["selected_profile"]}
        run_config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _load(self, name: str) -> dict[str, Any]:
        value = json.loads((self.run_dir / name).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"audio artifact must have an object root: {name}")
        return value
