"""Synthetic WAV checks for deterministic M21/M23 signal helpers."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from elyza_agent_tasks_customer_service.evaluation.audio import audio_signal_metrics as metrics


def _wav(path: Path, *, amplitude: float = 0.2, rate: int = 8_000) -> None:
    """Write a non-empty mono PCM16 sine WAV for audio-only unit tests."""

    samples = (amplitude * np.sin(np.linspace(0, 2 * math.pi * 440, rate, endpoint=False)) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())


def _policy() -> dict[str, object]:
    return {
        "profile_id": "unit", "required_facets": ["decode_format", "rms_peak", "continuity"],
        "sample_rate_hz": [8_000], "channels": [1], "sample_width_bits": [16],
        "rms_dbfs_min": -30, "rms_dbfs_max": -5, "peak_dbfs_max": -1,
        "clip_ratio_max": 0, "flat_top_max_ms": 1, "near_zero_abs_max": 0.001,
        "dropout_max_ms": 1, "snr_min_db": 10, "rolloff_min_hz": 0,
        "rolloff_max_hz": 4_000, "dc_max_dbfs": -40, "active_level_dbov_min": -99,
        "active_level_dbov_max": 0, "inapplicable_facets": {},
    }


def test_inspect_wav_and_m23_facets_for_synthetic_signal(tmp_path: Path) -> None:
    """A valid synthesized WAV produces measurable M23 facets without an API."""

    path = tmp_path / "tone.wav"
    _wav(path)
    decoded = metrics.inspect_wav(path)
    segment = {"audio_segment_id": "s1", "frame_count": 8_000, "complete": True, "boundary": "provider", "direction": "output", "chunk_sequence_ids": ["c1"], "sample_range": [0, 8_000]}
    context = {"segments": {"s1": segment}, "decoded": {"s1": decoded}, "events": {"played": {"audio_segment_id": "s1", "event_type": "audio_chunk_played_or_virtual_played", "data": {"chunk_sequence_id": "c1"}}}}
    plan = {"quality_id": "q1", "applicable": True, "audio_segment_id": "s1", "speech_sample_ranges": [[0, 8_000]], "noise_sample_ranges": [], "p56_observation": None}

    facets, observations = metrics._signal_facets(context, plan, _policy())

    assert decoded["duration_ns"] == 1_000_000_000
    assert facets["decode_format"]["status"] == "passed"
    assert facets["continuity"]["status"] == "passed"
    assert observations["rms_dbfs"] < -10


def test_wav_rejects_non_wave(tmp_path: Path) -> None:
    """Malformed audio cannot silently become signal evidence."""

    path = tmp_path / "bad.wav"
    path.write_bytes(b"not-a-wave")

    with pytest.raises(metrics.AudioMetricContractError, match="truncated"):
        metrics.inspect_wav(path)


def test_m21_declared_turn_evidence_path(tmp_path: Path) -> None:
    """Declared turn evidence scores without transport services."""

    events = {
        "control": {"data": {"control": "yield"}, "monotonic_ns": 10, "audio_segment_id": None},
        "end": {"monotonic_ns": 20, "audio_segment_id": None},
    }
    context = {
        "plan": {
            "turn_probes": [{"probe_id": "turn", "applicable": True, "expected_control": "yield", "observed_control_event_id": "control", "latency_kind": "response", "start_event_id": "control", "end_event_id": "end"}],
            "signal_quality_segments": [],
        },
        "segments": {}, "decoded": {}, "events": events,
        "audio": {"clock": {"measured_uncertainty_ns": 0}},
    }

    assert metrics._score_m21(context, {"selected_profile": {"profile_id": "p", "response_latency_max_ms": 1, "clock_uncertainty_max_ms": 1}})["status"] == "passed"
