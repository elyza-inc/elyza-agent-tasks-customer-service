"""Unmarked audio-ledger boundaries for the direct deterministic scorers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import struct
import wave

import pytest

from elyza_agent_tasks_customer_service.evaluation.audio.audio_signal_metrics import (
    inspect_wav,
    score_signal_metrics,
)
from elyza_agent_tasks_customer_service.evaluation.audio.audio_value_metrics import (
    score_value_metrics,
)


HASH = "0" * 64
SEGMENT_ID = "operator-1"
TARGET_ID = "claim-1"
QUALITY_ID = "quality-1"
CLOCK_ID = "clock-1"
WAV_NAME = "audio.wav"
SAMPLE_RATE = 8_000
SAMPLE_COUNT = 800


def _samples(kind: str = "tone") -> list[int]:
    """Return one PCM fixture shape: tone, silence, clipped, or DC."""

    if kind == "silence":
        return [0] * SAMPLE_COUNT
    if kind == "clipped":
        return [32767] * SAMPLE_COUNT
    if kind == "dc":
        return [8_000] * SAMPLE_COUNT
    return [8_000 if index % 20 < 10 else -8_000 for index in range(SAMPLE_COUNT)]


def _legacy_inputs(tmp_path: Path, *, sample_kind: str = "tone") -> tuple[dict, list[dict], dict]:
    """Build valid in-memory legacy manifests plus a relative PCM WAV fixture."""

    samples = _samples(sample_kind)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / WAV_NAME
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    inspected = inspect_wav(path)
    segment = {
        "audio_segment_id": SEGMENT_ID,
        "speaker": "operator",
        "direction": "output",
        "boundary": "playback",
        "relative_path": WAV_NAME,
        "wav_sha256": inspected["wav_sha256"],
        "canonical_pcm_sha256": inspected["canonical_pcm_sha256"],
        "sample_rate_hz": SAMPLE_RATE,
        "channels": 1,
        "sample_width_bits": 16,
        "frame_count": SAMPLE_COUNT,
        "duration_ns": inspected["duration_ns"],
        "sample_range": [0, SAMPLE_COUNT],
        "monotonic_range_ns": [10_000_000, 110_000_000],
        "complete": True,
        "chunk_sequence_ids": ["chunk-1"],
    }
    audio = {
        "schema_version": "audio_evidence_manifest.v1",
        "run_id": "run-1",
        "scenario_id": "scenario-1",
        "route": "fixture",
        "condition_hashes": dict.fromkeys(("runtime", "scorer", "audio_policy"), HASH),
        "clock": {"clock_id": CLOCK_ID, "measured_uncertainty_ns": 0},
        "audio_segments": [segment],
        "observer_versions": {},
        "metric_plan": {"turn_probes": [], "signal_quality_segments": []},
    }
    events = [
        _event("intent", 1, 0, data={"value": "09000000035"}),
        _event("sink", 2, 1_000_000, data={"value": "09000000035"}),
        _event("observed", 3, 2_000_000, data={"control": "stop"}),
        _event("start", 4, 3_000_000),
        _event("end", 5, 23_000_000),
        _event("deadline", 6, 120_000_000),
        _event("chunk", 7, 100_000_000, segment_id=SEGMENT_ID, data={"chunk_sequence_id": "chunk-1"}),
    ]
    return audio, events, {}


def _event(event_id: str, seq: int, monotonic_ns: int, *, segment_id: str | None = None, data: dict | None = None) -> dict:
    """Build an exact audio_events.v1 event object for the legacy scorer."""

    return {"event_id": event_id, "seq": seq, "event_type": "audio_chunk_played_or_virtual_played" if event_id == "chunk" else "fixture", "clock_id": CLOCK_ID, "monotonic_ns": monotonic_ns, "audio_segment_id": segment_id, "sample_range": [0, SAMPLE_COUNT] if segment_id else None, "data": data or {}}


def _legacy_score(tmp_path: Path, *, sample_kind: str = "tone", mutate=None) -> dict:
    """Score valid legacy artifact mappings; ``mutate`` receives all three inputs."""

    audio, events, config = _legacy_inputs(tmp_path, sample_kind=sample_kind)
    if mutate:
        mutate(audio, events, config)
    return score_signal_metrics(audio_manifest=audio, event_log=events, config=config, evidence_root=tmp_path)["metrics"]


def _probe(kind: str | None = "stop", *, expected: str = "stop") -> dict:
    """Return the exact legacy M21 turn-control contract for TARGET_ID."""

    return {"probe_id": TARGET_ID, "applicable": True, "expected_control": expected, "observed_control_event_id": "observed", "latency_kind": kind, "start_event_id": "start", "end_event_id": "end"}


def _m21_policy(**values: float) -> dict:
    """Return the selected M21 threshold profile with overridable numeric bounds."""

    policy = {"profile_id": "m21", "stop_latency_max_ms": 10, "response_latency_max_ms": 10, "resume_latency_max_ms": 10, "pause_hold_min_ms": 30, "clock_uncertainty_max_ms": 5}
    policy.update(values)
    return {"selected_profile": policy}


def _quality(*, applicable: bool = True, speech: list[list[int]] | None = None, noise: list[list[int]] | None = None, p56: dict | None = None) -> dict:
    """Return the exact M23 plan row with explicit local sample ranges."""

    return {"quality_id": QUALITY_ID, "applicable": applicable, "audio_segment_id": SEGMENT_ID, "speech_sample_ranges": speech if speech is not None else [[0, 400]], "noise_sample_ranges": noise if noise is not None else [[400, 800]], "p56_observation": p56}


def _m23_policy(required: str, **values: object) -> dict:
    """Return a selected M23 profile whose only required facet is ``required``."""

    policy = {"profile_id": "m23", "required_facets": [required], "sample_rate_hz": [SAMPLE_RATE], "channels": [1], "sample_width_bits": [16], "rms_dbfs_min": -30, "rms_dbfs_max": -1, "peak_dbfs_max": -0.1, "clip_ratio_max": 1, "flat_top_max_ms": 100, "near_zero_abs_max": 0.001, "dropout_max_ms": 100, "snr_min_db": -10, "rolloff_min_hz": 0, "rolloff_max_hz": 4_000, "dc_max_dbfs": -10, "active_level_dbov_min": -50, "active_level_dbov_max": -1, "inapplicable_facets": {}}
    policy.update(values)
    return {"selected_profile": policy}


def _unit(rows: dict, metric_id: str, unit_id: str) -> dict:
    """Return one metric unit after checking that its target ID is present."""

    row = rows[metric_id]
    return next(unit for unit in row["units"] if unit["unit_id"] == unit_id)


@pytest.mark.parametrize(
    ("ledger", "probe", "policy", "clock_ns", "config", "status", "reason"),
    [
        ("F1 turn control分類違い", _probe(expected="resume"), _m21_policy(), 0, {}, "failed", "turn_control_classification_mismatch"),
        ("F2 stop/response/resume latencyがmax超過", _probe("response"), _m21_policy(), 0, {}, "failed", "latency_interval_exceeds_maximum"),
        ("F3 pause_holdがmin未満", _probe("pause_hold"), _m21_policy(), 0, {}, "failed", "hold_interval_below_minimum"),
        ("NM1 threshold profile未選択", _probe(), {"selected_profile": None}, 0, {}, "N/M", "M21_threshold_profile_unselected"),
        ("NM2 clock uncertainty上限超過", _probe(), _m21_policy(), 6_000_000, {}, "N/M", "clock_uncertainty_exceeds_policy"),
        ("NM3 uncertainty区間が閾値を跨ぐ", _probe(), _m21_policy(clock_uncertainty_max_ms=20), 15_000_000, {}, "N/M", "clock_uncertainty_crosses_latency_threshold"),
    ],
    ids=("f1", "f2", "f3", "nm1", "nm2", "nm3"),
)
def test_m21_unmarked_probe_rows(ledger: str, probe: dict, policy: dict, clock_ns: int, config: dict, status: str, reason: str, tmp_path: Path) -> None:
    """台帳B/M21 F1--F3, NM1--NM3: probe reason code and claim ID are fixed."""

    del ledger

    def mutate(audio: dict, events: list[dict], base_config: dict) -> None:
        del events
        audio["metric_plan"]["turn_probes"] = [deepcopy(probe)]
        audio["clock"]["measured_uncertainty_ns"] = clock_ns
        base_config["m21"] = deepcopy(policy)
        base_config.update(config)

    unit = _unit(_legacy_score(tmp_path, mutate=mutate), "M21", TARGET_ID)
    assert (unit["status"], unit["reason"]) == (status, reason)


def test_m21_unmarked_na_and_evidence_gap_rows(tmp_path: Path) -> None:
    """台帳B/M21 NA1, NM4: no probe and declared evidence gap remain explicit."""

    na = _legacy_score(tmp_path / "na")["M21"]
    assert (na["status"], na["reason"], na["units"]) == ("N/A", "no_applicable_units", [])

    def gap(audio: dict, events: list[dict], config: dict) -> None:
        del audio, events
        config["evidence_gaps"] = {"M21": ["turn_control_evidence_missing"]}

    unit = _unit(_legacy_score(tmp_path / "gap", mutate=gap), "M21", "m21-evidence-gap-1")
    assert (unit["status"], unit["reason"]) == ("N/M", "turn_control_evidence_missing")


@pytest.mark.parametrize(
    ("ledger", "sample_kind", "quality", "policy", "mutate_segment", "status", "reason"),
    [
        ("F1 decode_format", "tone", _quality(), _m23_policy("decode_format", sample_rate_hz=[16_000]), None, "failed", "required_signal_facet_failed"),
        ("F2 rms_peak", "tone", _quality(), _m23_policy("rms_peak", rms_dbfs_max=-100), None, "failed", "required_signal_facet_failed"),
        ("F3 clipping", "clipped", _quality(), _m23_policy("clipping", clip_ratio_max=0), None, "failed", "required_signal_facet_failed"),
        ("F4 flat_top", "clipped", _quality(), _m23_policy("flat_top", flat_top_max_ms=0), None, "failed", "required_signal_facet_failed"),
        ("F5 dropout", "silence", _quality(speech=[[0, 400]]), _m23_policy("dropout", dropout_max_ms=0), None, "failed", "required_signal_facet_failed"),
        ("F6 snr", "tone", _quality(), _m23_policy("snr", snr_min_db=10), None, "failed", "required_signal_facet_failed"),
        ("F7 bandwidth", "tone", _quality(), _m23_policy("bandwidth", rolloff_min_hz=10_000, rolloff_max_hz=12_000), None, "failed", "required_signal_facet_failed"),
        ("F8 dc", "dc", _quality(), _m23_policy("dc", dc_max_dbfs=-30), None, "failed", "required_signal_facet_failed"),
        ("F9 continuity", "tone", _quality(), _m23_policy("continuity"), lambda audio: audio["audio_segments"][0].update(complete=False), "failed", "required_signal_facet_failed"),
        ("F10 active_level", "tone", _quality(p56={"method": "ITU-T P.56 Method B", "conformance_verified": True, "active_level_dbov": -100}), _m23_policy("active_level"), None, "failed", "required_signal_facet_failed"),
        ("NA1 segmentなし/非applicable", "tone", _quality(applicable=False), _m23_policy("decode_format"), None, "N/A", "signal_quality_not_applicable"),
        ("NM1 profile未選択", "tone", _quality(), {"selected_profile": None}, None, "N/M", "M23_threshold_profile_unselected"),
        ("NM2 speech/noise range不足", "tone", _quality(speech=[], noise=[]), _m23_policy("snr"), None, "N/M", "required_signal_facet_not_measurable"),
        ("NM3 P.56なし/未検証", "tone", _quality(p56=None), _m23_policy("active_level"), None, "N/M", "required_signal_facet_not_measurable"),
    ],
    ids=("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "na1", "nm1", "nm2", "nm3"),
)
def test_m23_unmarked_signal_rows(ledger: str, sample_kind: str, quality: dict, policy: dict, mutate_segment, status: str, reason: str, tmp_path: Path) -> None:
    """台帳B/M23 F1--F10, NA1, NM1--NM3: required facet fixes result and quality ID."""

    del ledger

    def mutate(audio: dict, events: list[dict], config: dict) -> None:
        del events
        audio["metric_plan"]["signal_quality_segments"] = [deepcopy(quality)]
        config["m23"] = deepcopy(policy)
        if mutate_segment:
            mutate_segment(audio)

    unit = _unit(_legacy_score(tmp_path, sample_kind=sample_kind, mutate=mutate), "M23", QUALITY_ID)
    assert (unit["status"], unit["reason"]) == (status, reason)


def test_m23_unmarked_evidence_gap_row(tmp_path: Path) -> None:
    """台帳B/M23 NM4: declared evidence gap has a stable reason and synthetic ID."""

    def gap(audio: dict, events: list[dict], config: dict) -> None:
        del audio, events
        config["evidence_gaps"] = {"M23": ["signal_quality_evidence_missing"]}

    unit = _unit(_legacy_score(tmp_path, mutate=gap), "M23", "m23-evidence-gap-1")
    assert (unit["status"], unit["reason"]) == ("N/M", "signal_quality_evidence_missing")


@pytest.mark.parametrize("metric_id", ("M25", "M26"), ids=("m25", "m26"))
def test_unmarked_shape_rows_are_not_contract_invalid(metric_id: str) -> None:
    """台帳B/M25 NM1/X1 と M26 X1: malformed record is N/M, never CI/exception."""

    row = score_value_metrics(record={"event_log": None}, scenario={})["metrics"][metric_id]
    assert (row["status"], row["reason"], row["units"]) == ("N/M", "record.event_log must be an object array", [])
