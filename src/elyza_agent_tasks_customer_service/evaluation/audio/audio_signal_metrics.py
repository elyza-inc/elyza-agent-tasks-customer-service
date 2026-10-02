"""Deterministic M21 (response latency) and M23 (signal quality) scoring from persisted audio evidence.

Accepted inputs:

* ``audio_manifest`` is an object or a ``.json`` path with schema
  ``audio_evidence_manifest.v1``. Relative WAV paths are resolved below the
  manifest directory, or below ``evidence_root`` when an in-memory object is
  supplied.
* ``event_log`` is a list of event objects, an ``audio_events.v1`` object, a
  ``.json`` array/object path, or a ``.jsonl`` path containing one object per
  non-empty line.
* ``config`` is an object. M21 and M23 use only explicitly selected profiles;
  a missing or null selected profile makes the applicable metric ``N/M``.

Malformed JSON, non-object rows, absolute/traversing/symlink audio paths,
duplicate IDs, broken joins, inconsistent hashes, and inconsistent audio
metadata produce ``contract_invalid`` metric results. No threshold or
timestamp is coerced or filled automatically.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


RESULT_SCHEMA_VERSION = "audio_signal_metrics"
AUDIO_SCHEMA_VERSION = "audio_evidence_manifest.v1"
EVENT_SCHEMA_VERSION = "audio_events.v1"
METRIC_IDS = ("M21", "M23")
EVIDENCE_GAPS_CONFIG_KEY = "evidence_gaps"
BOUNDARIES = {"provider", "receiver", "playback"}
RESULT_STATUSES = {"passed", "failed", "N/A", "N/M", "contract_invalid"}
SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")
RIFF_HEADER = struct.Struct("<4sI4s")
RIFF_CHUNK_HEADER = struct.Struct("<4sI")
WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
DB_FLOOR = -300.0
NS_PER_SECOND = 1_000_000_000
MS_PER_SECOND = 1_000.0


class AudioMetricContractError(ValueError):
    """Raised when persisted audio evidence violates the scorer contract."""


def score_signal_metrics(
    *,
    audio_manifest: Mapping[str, Any] | str | Path,
    event_log: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    config: Mapping[str, Any],
    evidence_root: str | Path | None = None,
) -> dict[str, Any]:
    """Score M21 and M23 without combining them into a single score.

    See the module docstring for accepted input formats. Contract failures are
    represented in the returned vector rather than raised. Programming errors
    outside input validation are not swallowed.
    """

    try:
        loaded_audio, inferred_root = _load_object(audio_manifest, "audio_manifest")
        loaded_events = _load_events(event_log)
        if not isinstance(config, Mapping):
            raise AudioMetricContractError("config must be an object")
        root = _resolve_evidence_root(evidence_root, inferred_root)
        context = _validate_and_join(loaded_audio, loaded_events, root)
        results = {
            "M21": _score_m21(context, config.get("m21")),
            "M23": _score_m23(context, config.get("m23")),
        }
        _apply_evidence_gaps(results, config.get(EVIDENCE_GAPS_CONFIG_KEY))
    except AudioMetricContractError as exc:
        results = {metric_id: _contract_invalid(metric_id, str(exc)) for metric_id in METRIC_IDS}
        run_id = None
    else:
        run_id = loaded_audio["run_id"]

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "metrics": results,
    }


def inspect_wav(path: str | Path) -> dict[str, Any]:
    """Decode a RIFF/WAVE file and return metadata, hashes, and normalized samples.

    PCM integer (8/16/24/32-bit) and IEEE float (32/64-bit) WAV are accepted.
    RIFF/WAVE extensible PCM/float is also accepted. Compressed codecs, malformed
    chunks, empty audio, NaN/Inf float samples, and unsupported widths raise
    ``AudioMetricContractError``.
    """

    wav_path = Path(path)
    try:
        payload = wav_path.read_bytes()
    except OSError as exc:
        raise AudioMetricContractError(f"cannot read WAV {wav_path}: {exc}") from exc
    if len(payload) < RIFF_HEADER.size:
        raise AudioMetricContractError(f"WAV is truncated: {wav_path}")
    riff, declared_size, wave_id = RIFF_HEADER.unpack_from(payload)
    if riff != b"RIFF" or wave_id != b"WAVE":
        raise AudioMetricContractError(f"audio is not RIFF/WAVE: {wav_path}")
    if declared_size + 8 > len(payload):
        raise AudioMetricContractError(f"RIFF declared size exceeds file size: {wav_path}")

    fmt: bytes | None = None
    pcm_bytes: bytes | None = None
    offset = RIFF_HEADER.size
    while offset + RIFF_CHUNK_HEADER.size <= len(payload):
        chunk_id, chunk_size = RIFF_CHUNK_HEADER.unpack_from(payload, offset)
        offset += RIFF_CHUNK_HEADER.size
        chunk_end = offset + chunk_size
        if chunk_end > len(payload):
            raise AudioMetricContractError(f"WAV chunk is truncated: {wav_path}")
        chunk = payload[offset:chunk_end]
        if chunk_id == b"fmt " and fmt is None:
            fmt = chunk
        elif chunk_id == b"data" and pcm_bytes is None:
            pcm_bytes = chunk
        offset = chunk_end + (chunk_size % 2)

    if fmt is None or pcm_bytes is None or len(fmt) < 16:
        raise AudioMetricContractError(f"WAV requires fmt and data chunks: {wav_path}")
    format_tag, channels, sample_rate_hz, byte_rate, block_align, bits = struct.unpack_from(
        "<HHIIHH", fmt
    )
    if format_tag == WAVE_FORMAT_EXTENSIBLE:
        if len(fmt) < 40:
            raise AudioMetricContractError(f"WAV extensible fmt is truncated: {wav_path}")
        valid_bits, _channel_mask = struct.unpack_from("<HI", fmt, 18)
        subformat_tag = struct.unpack_from("<H", fmt, 24)[0]
        if valid_bits:
            bits = valid_bits
        format_tag = subformat_tag
    if format_tag not in {WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT}:
        raise AudioMetricContractError(f"unsupported WAV format tag {format_tag}: {wav_path}")
    if not isinstance(channels, int) or channels <= 0:
        raise AudioMetricContractError(f"WAV channels must be positive: {wav_path}")
    if sample_rate_hz <= 0 or bits <= 0 or bits % 8:
        raise AudioMetricContractError(f"WAV sample rate/width is invalid: {wav_path}")
    sample_width = bits // 8
    expected_align = channels * sample_width
    if block_align != expected_align or byte_rate != sample_rate_hz * block_align:
        raise AudioMetricContractError(f"WAV byte/block alignment is inconsistent: {wav_path}")
    if not pcm_bytes or len(pcm_bytes) % block_align:
        raise AudioMetricContractError(f"WAV data length is empty or misaligned: {wav_path}")

    samples = _decode_samples(pcm_bytes, format_tag, bits)
    if samples.size % channels:
        raise AudioMetricContractError(f"decoded WAV sample count is misaligned: {wav_path}")
    frames = samples.reshape((-1, channels))
    if not np.all(np.isfinite(frames)):
        raise AudioMetricContractError(f"WAV contains NaN or Inf samples: {wav_path}")
    frame_count = int(frames.shape[0])
    return {
        "format_tag": format_tag,
        "channels": channels,
        "sample_rate_hz": sample_rate_hz,
        "sample_width_bits": bits,
        "frame_count": frame_count,
        "duration_ns": round(frame_count * NS_PER_SECOND / sample_rate_hz),
        "wav_sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_pcm_sha256": hashlib.sha256(pcm_bytes).hexdigest(),
        "samples": frames,
    }


def _decode_samples(pcm: bytes, format_tag: int, bits: int) -> np.ndarray:
    if format_tag == WAVE_FORMAT_IEEE_FLOAT:
        if bits == 32:
            return np.frombuffer(pcm, dtype="<f4").astype(np.float64)
        if bits == 64:
            return np.frombuffer(pcm, dtype="<f8").astype(np.float64)
        raise AudioMetricContractError(f"unsupported IEEE float width: {bits}")
    if bits == 8:
        return (np.frombuffer(pcm, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if bits == 16:
        return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    if bits == 24:
        octets = np.frombuffer(pcm, dtype=np.uint8).reshape((-1, 3)).astype(np.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float64) / 8388608.0
    if bits == 32:
        return np.frombuffer(pcm, dtype="<i4").astype(np.float64) / 2147483648.0
    raise AudioMetricContractError(f"unsupported PCM width: {bits}")


def _load_object(
    value: Mapping[str, Any] | str | Path,
    label: str,
) -> tuple[dict[str, Any], Path | None]:
    """Load a mapping or object-root ``.json`` path.

    Non-JSON paths, malformed JSON, array/scalar roots, and non-string object
    keys raise ``AudioMetricContractError``.
    """

    inferred_root: Path | None = None
    if isinstance(value, Mapping):
        loaded: Any = dict(value)
    elif isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() != ".json":
            raise AudioMetricContractError(f"{label} path must end in .json: {path}")
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AudioMetricContractError(f"cannot load {label} {path}: {exc}") from exc
        inferred_root = path.resolve().parent
    else:
        raise AudioMetricContractError(f"{label} must be an object or .json path")
    if not isinstance(loaded, dict) or any(not isinstance(key, str) for key in loaded):
        raise AudioMetricContractError(f"{label} must have an object root with string keys")
    return loaded, inferred_root


def _load_events(
    value: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
) -> list[dict[str, Any]]:
    """Load event rows from a list, versioned object, ``.json``, or ``.jsonl``.

    JSON object roots must contain only ``schema_version`` and ``events``.
    Empty/non-object JSONL rows, malformed JSON, and other suffixes raise
    ``AudioMetricContractError``.
    """

    loaded: Any
    if isinstance(value, Mapping):
        loaded = dict(value)
    elif isinstance(value, (str, Path)):
        path = Path(value)
        try:
            if path.suffix.lower() == ".jsonl":
                rows: list[Any] = []
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise AudioMetricContractError(
                            f"event_log JSONL line {line_number} is malformed: {exc}"
                        ) from exc
                loaded = rows
            elif path.suffix.lower() == ".json":
                loaded = json.loads(path.read_text(encoding="utf-8"))
            else:
                raise AudioMetricContractError(f"event_log path must end in .json or .jsonl: {path}")
        except OSError as exc:
            raise AudioMetricContractError(f"cannot read event_log {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise AudioMetricContractError(f"event_log JSON is malformed: {exc}") from exc
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        loaded = list(value)
    else:
        raise AudioMetricContractError("event_log must be a list, object, .json, or .jsonl path")
    if isinstance(loaded, dict):
        if set(loaded) != {"schema_version", "events"}:
            raise AudioMetricContractError("event_log object fields must be schema_version and events")
        if loaded["schema_version"] != EVENT_SCHEMA_VERSION:
            raise AudioMetricContractError("event_log schema_version mismatch")
        loaded = loaded["events"]
    if not isinstance(loaded, list) or any(not isinstance(row, dict) for row in loaded):
        raise AudioMetricContractError("event_log must contain only event objects")
    return [dict(row) for row in loaded]


def _resolve_evidence_root(explicit: str | Path | None, inferred: Path | None) -> Path:
    if explicit is not None:
        root = Path(explicit).resolve()
    elif inferred is not None:
        root = inferred
    else:
        raise AudioMetricContractError(
            "evidence_root is required when audio_manifest is supplied in memory"
        )
    if not root.is_dir():
        raise AudioMetricContractError(f"evidence_root must be an existing directory: {root}")
    return root


def _validate_and_join(
    audio: dict[str, Any],
    events: list[dict[str, Any]],
    root: Path,
) -> dict[str, Any]:
    _require_exact_fields(
        audio,
        required={
            "schema_version",
            "run_id",
            "scenario_id",
            "route",
            "condition_hashes",
            "clock",
            "audio_segments",
            "observer_versions",
            "metric_plan",
        },
        label="audio_manifest",
    )
    if audio["schema_version"] != AUDIO_SCHEMA_VERSION:
        raise AudioMetricContractError("audio_manifest schema_version mismatch")
    _require_nonempty_string(audio["run_id"], "audio_manifest.run_id")
    _require_nonempty_string(audio["scenario_id"], "audio_manifest.scenario_id")
    _require_nonempty_string(audio["route"], "audio_manifest.route")
    hashes = audio["condition_hashes"]
    if not isinstance(hashes, dict) or set(hashes) != {"runtime", "scorer", "audio_policy"}:
        raise AudioMetricContractError("audio_manifest condition_hashes fields are invalid")
    for name, value in hashes.items():
        _validate_sha256(value, f"condition_hashes.{name}")

    clock = audio["clock"]
    _require_exact_fields(
        clock,
        required={"clock_id", "measured_uncertainty_ns"},
        label="audio_manifest.clock",
    )
    _require_nonempty_string(clock["clock_id"], "audio_manifest.clock.clock_id")
    _require_nonnegative_int(
        clock["measured_uncertainty_ns"],
        "audio_manifest.clock.measured_uncertainty_ns",
    )
    if not isinstance(audio["observer_versions"], dict):
        raise AudioMetricContractError("audio_manifest.observer_versions must be an object")

    raw_segments = audio["audio_segments"]
    if not isinstance(raw_segments, list):
        raise AudioMetricContractError("audio_manifest.audio_segments must be an array")
    segment_rows: dict[str, dict[str, Any]] = {}
    decoded: dict[str, dict[str, Any]] = {}
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, dict):
            raise AudioMetricContractError(f"audio_segments[{index}] must be an object")
        _validate_segment(segment, index)
        segment_id = segment["audio_segment_id"]
        if segment_id in segment_rows:
            raise AudioMetricContractError(f"duplicate audio_segment_id: {segment_id}")
        path = _contained_audio_path(root, segment["relative_path"])
        observation = inspect_wav(path)
        _verify_segment_audio(segment, observation)
        segment_rows[segment_id] = segment
        decoded[segment_id] = observation

    event_rows: dict[str, dict[str, Any]] = {}
    previous_seq: int | None = None
    for index, event in enumerate(events):
        required = {
            "event_id",
            "seq",
            "event_type",
            "clock_id",
            "monotonic_ns",
            "audio_segment_id",
            "sample_range",
        }
        _require_exact_fields(event, required=required, optional={"data"}, label=f"event_log[{index}]")
        event_id = _require_nonempty_string(event["event_id"], f"event_log[{index}].event_id")
        if event_id in event_rows:
            raise AudioMetricContractError(f"duplicate event_id: {event_id}")
        seq = _require_nonnegative_int(event["seq"], f"event_log[{index}].seq")
        if previous_seq is not None and seq <= previous_seq:
            raise AudioMetricContractError("event_log seq values must be strictly increasing")
        previous_seq = seq
        _require_nonempty_string(event["event_type"], f"event_log[{index}].event_type")
        if event["clock_id"] != clock["clock_id"]:
            raise AudioMetricContractError(f"event {event_id} uses a different clock_id")
        _require_nonnegative_int(event["monotonic_ns"], f"event_log[{index}].monotonic_ns")
        segment_id = event["audio_segment_id"]
        if segment_id is not None and segment_id not in segment_rows:
            raise AudioMetricContractError(f"event {event_id} references unknown audio segment")
        sample_range = event["sample_range"]
        _validate_range(sample_range, f"event_log[{index}].sample_range", allow_none=True)
        if not isinstance(event.get("data", {}), dict):
            raise AudioMetricContractError(f"event_log[{index}].data must be an object")
        event_rows[event_id] = event

    metric_plan = audio["metric_plan"]
    _require_exact_fields(
        metric_plan,
        required={"turn_probes", "signal_quality_segments"},
        label="audio_manifest.metric_plan",
    )
    for name in metric_plan:
        if not isinstance(metric_plan[name], list):
            raise AudioMetricContractError(f"metric_plan.{name} must be an array")
    return {
        "audio": audio,
        "events": event_rows,
        "segments": segment_rows,
        "decoded": decoded,
        "plan": metric_plan,
    }


def _validate_segment(segment: dict[str, Any], index: int) -> None:
    _require_exact_fields(
        segment,
        required={
            "audio_segment_id",
            "speaker",
            "direction",
            "boundary",
            "relative_path",
            "wav_sha256",
            "canonical_pcm_sha256",
            "sample_rate_hz",
            "channels",
            "sample_width_bits",
            "frame_count",
            "duration_ns",
            "sample_range",
            "monotonic_range_ns",
            "complete",
            "chunk_sequence_ids",
        },
        label=f"audio_segments[{index}]",
    )
    _require_nonempty_string(segment["audio_segment_id"], f"audio_segments[{index}].audio_segment_id")
    if segment["speaker"] not in {"user", "operator", "mixed"}:
        raise AudioMetricContractError(f"audio_segments[{index}].speaker is invalid")
    if segment["direction"] not in {"input", "output"}:
        raise AudioMetricContractError(f"audio_segments[{index}].direction is invalid")
    if segment["boundary"] not in BOUNDARIES:
        raise AudioMetricContractError(f"audio_segments[{index}].boundary is invalid")
    _require_nonempty_string(segment["relative_path"], f"audio_segments[{index}].relative_path")
    _validate_sha256(segment["wav_sha256"], f"audio_segments[{index}].wav_sha256")
    _validate_sha256(
        segment["canonical_pcm_sha256"],
        f"audio_segments[{index}].canonical_pcm_sha256",
    )
    for field in ("sample_rate_hz", "channels", "sample_width_bits", "frame_count", "duration_ns"):
        value = _require_nonnegative_int(segment[field], f"audio_segments[{index}].{field}")
        if value == 0:
            raise AudioMetricContractError(f"audio_segments[{index}].{field} must be positive")
    _validate_range(segment["sample_range"], f"audio_segments[{index}].sample_range")
    _validate_range(segment["monotonic_range_ns"], f"audio_segments[{index}].monotonic_range_ns")
    if segment["sample_range"][1] - segment["sample_range"][0] != segment["frame_count"]:
        raise AudioMetricContractError(f"audio_segments[{index}] sample_range/frame_count mismatch")
    if not isinstance(segment["complete"], bool):
        raise AudioMetricContractError(f"audio_segments[{index}].complete must be boolean")
    chunk_ids = segment["chunk_sequence_ids"]
    if (
        not isinstance(chunk_ids, list)
        or any(not isinstance(item, str) or not item for item in chunk_ids)
        or len(set(chunk_ids)) != len(chunk_ids)
    ):
        raise AudioMetricContractError(f"audio_segments[{index}].chunk_sequence_ids are invalid")


def _contained_audio_path(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AudioMetricContractError(f"audio relative_path escapes evidence root: {relative_path}")
    unresolved = root / candidate
    if unresolved.is_symlink():
        raise AudioMetricContractError(f"audio relative_path may not be a symlink: {relative_path}")
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(root):
        raise AudioMetricContractError(f"audio relative_path escapes evidence root: {relative_path}")
    if not resolved.is_file():
        raise AudioMetricContractError(f"audio relative_path is not a file: {relative_path}")
    return resolved


def _verify_segment_audio(segment: dict[str, Any], observation: dict[str, Any]) -> None:
    for field in ("sample_rate_hz", "channels", "sample_width_bits", "frame_count", "duration_ns"):
        if segment[field] != observation[field]:
            raise AudioMetricContractError(
                f"audio segment {segment['audio_segment_id']} declared {field} does not match WAV"
            )
    for field in ("wav_sha256", "canonical_pcm_sha256"):
        declared = _normalized_sha256(segment[field])
        if declared != observation[field]:
            raise AudioMetricContractError(
                f"audio segment {segment['audio_segment_id']} declared {field} does not match WAV"
            )


def _score_m21(context: dict[str, Any], raw_policy: Any) -> dict[str, Any]:
    probes = context["plan"]["turn_probes"]
    if not probes:
        return _metric_result("M21", [])
    policy = _selected_policy_or_none(raw_policy, "M21")
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict):
            raise AudioMetricContractError(f"metric_plan.turn_probes[{index}] must be an object")
        _require_exact_fields(
            probe,
            required={
                "probe_id",
                "applicable",
                "expected_control",
                "observed_control_event_id",
                "latency_kind",
                "start_event_id",
                "end_event_id",
            },
            label=f"metric_plan.turn_probes[{index}]",
        )
        probe_id = _require_nonempty_string(probe["probe_id"], f"turn_probes[{index}].probe_id")
        if probe_id in seen:
            raise AudioMetricContractError(f"duplicate turn probe_id: {probe_id}")
        seen.add(probe_id)
        if not isinstance(probe["applicable"], bool):
            raise AudioMetricContractError(f"turn probe {probe_id}.applicable must be boolean")
        if not probe["applicable"]:
            units.append(_unit(probe_id, "N/A", "probe_opportunity_not_present"))
            continue
        if policy is None:
            units.append(_unit(probe_id, "N/M", "M21_threshold_profile_unselected"))
            continue
        observed_event = _event(
            context, probe["observed_control_event_id"], f"turn probe {probe_id} observed control"
        )
        label = f"turn probe {probe_id} observed control"
        data = observed_event.get("data", {})
        if "control" not in data:
            raise AudioMetricContractError(f"{label} field path is missing: data.control")
        observed_control = data["control"]
        if observed_control is None:
            raise AudioMetricContractError(f"{label} field value may not be null")
        if not isinstance(observed_control, str):
            raise AudioMetricContractError(f"turn probe {probe_id} observed control must be string")
        expected = _require_nonempty_string(
            probe["expected_control"], f"turn probe {probe_id}.expected_control"
        )
        if observed_control != expected:
            units.append(
                _unit(
                    probe_id,
                    "failed",
                    "turn_control_classification_mismatch",
                    {"expected_control": expected, "observed_control": observed_control},
                )
            )
            continue
        latency_kind = probe["latency_kind"]
        if latency_kind is None:
            units.append(
                _unit(
                    probe_id,
                    "passed",
                    "turn_control_classification_matches",
                    {"expected_control": expected, "observed_control": observed_control},
                )
            )
            continue
        if latency_kind not in {"stop", "response", "resume", "pause_hold"}:
            raise AudioMetricContractError(f"turn probe {probe_id} latency_kind is invalid")
        threshold_key = f"{latency_kind}_latency_max_ms"
        if latency_kind == "pause_hold":
            threshold_key = "pause_hold_min_ms"
        threshold = _require_finite_number(policy.get(threshold_key), f"M21 policy.{threshold_key}")
        if threshold < 0:
            raise AudioMetricContractError(f"M21 policy.{threshold_key} must be nonnegative")
        start = _event(context, probe["start_event_id"], f"turn probe {probe_id} start")
        end = _event(context, probe["end_event_id"], f"turn probe {probe_id} end")
        latency_ms = (end["monotonic_ns"] - start["monotonic_ns"]) / 1_000_000
        if latency_ms < 0:
            raise AudioMetricContractError(f"turn probe {probe_id} latency is negative")
        uncertainty_ms = context["audio"]["clock"]["measured_uncertainty_ns"] / 1_000_000
        uncertainty_max = _require_finite_number(
            policy.get("clock_uncertainty_max_ms"),
            "M21 policy.clock_uncertainty_max_ms",
        )
        if uncertainty_ms > uncertainty_max:
            units.append(_unit(probe_id, "N/M", "clock_uncertainty_exceeds_policy"))
            continue
        lower = max(0.0, latency_ms - uncertainty_ms)
        upper = latency_ms + uncertainty_ms
        if latency_kind == "pause_hold":
            status, reason = _threshold_interval_min(lower, upper, threshold)
        else:
            status, reason = _threshold_interval_max(lower, upper, threshold)
        units.append(
            _unit(
                probe_id,
                status,
                reason,
                {
                    "latency_kind": latency_kind,
                    "latency_ms": latency_ms,
                    "uncertainty_ms": uncertainty_ms,
                    "threshold_ms": threshold,
                },
            )
        )
    return _metric_result("M21", units)


def _score_m23(context: dict[str, Any], raw_policy: Any) -> dict[str, Any]:
    plans = context["plan"]["signal_quality_segments"]
    if not plans:
        return _metric_result("M23", [])
    policy = _selected_policy_or_none(raw_policy, "M23")
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict):
            raise AudioMetricContractError(
                f"metric_plan.signal_quality_segments[{index}] must be an object"
            )
        _require_exact_fields(
            plan,
            required={
                "quality_id",
                "applicable",
                "audio_segment_id",
                "speech_sample_ranges",
                "noise_sample_ranges",
                "p56_observation",
            },
            label=f"metric_plan.signal_quality_segments[{index}]",
        )
        quality_id = _require_nonempty_string(
            plan["quality_id"], f"signal_quality_segments[{index}].quality_id"
        )
        if quality_id in seen:
            raise AudioMetricContractError(f"duplicate signal quality_id: {quality_id}")
        seen.add(quality_id)
        if not isinstance(plan["applicable"], bool):
            raise AudioMetricContractError(f"signal quality {quality_id}.applicable must be boolean")
        if not plan["applicable"]:
            units.append(_unit(quality_id, "N/A", "signal_quality_not_applicable"))
            continue
        segment_id = plan["audio_segment_id"]
        if segment_id not in context["segments"]:
            raise AudioMetricContractError(f"signal quality {quality_id} references unknown segment")
        if policy is None:
            units.append(_unit(quality_id, "N/M", "M23_threshold_profile_unselected"))
            continue
        facets, observations = _signal_facets(context, plan, policy)
        required_facets = policy.get("required_facets")
        if (
            not isinstance(required_facets, list)
            or not required_facets
            or any(not isinstance(item, str) for item in required_facets)
            or len(set(required_facets)) != len(required_facets)
        ):
            raise AudioMetricContractError("M23 policy.required_facets must be unique strings")
        unknown = set(required_facets) - set(facets)
        if unknown:
            raise AudioMetricContractError(f"M23 policy has unknown required facets: {sorted(unknown)}")
        required_statuses = [facets[name]["status"] for name in required_facets]
        if any(status == "failed" for status in required_statuses):
            status = "failed"
            reason = "required_signal_facet_failed"
        elif any(status == "N/M" for status in required_statuses):
            status = "N/M"
            reason = "required_signal_facet_not_measurable"
        else:
            status = "passed"
            reason = "all_required_signal_facets_passed"
        units.append(
            _unit(
                quality_id,
                status,
                reason,
                {"facets": facets, "observations": observations},
            )
        )
    return _metric_result("M23", units)


def _signal_facets(
    context: dict[str, Any],
    plan: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    segment = context["segments"][plan["audio_segment_id"]]
    decoded = context["decoded"][plan["audio_segment_id"]]
    samples = decoded["samples"].mean(axis=1)
    sample_rate = decoded["sample_rate_hz"]
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    rms_dbfs = _db(rms)
    peak_dbfs = _db(peak)
    clip_level = 1.0
    if decoded["format_tag"] == WAVE_FORMAT_PCM:
        clip_level = 1.0 - 2.0 ** (1 - decoded["sample_width_bits"])
    clipped = np.abs(samples) >= clip_level
    clip_ratio = float(np.mean(clipped))
    flat_top_ms = _flat_top_max_samples(samples, clipped) * MS_PER_SECOND / sample_rate
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size))) ** 2
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    rolloff_hz = _spectral_rolloff(frequencies, spectrum, 0.95)
    dc_dbfs = _db(abs(float(np.mean(samples))))

    facets: dict[str, dict[str, Any]] = {}
    expected_rates = policy.get("sample_rate_hz")
    expected_channels = policy.get("channels")
    expected_widths = policy.get("sample_width_bits")
    if (
        not isinstance(expected_rates, list)
        or any(not isinstance(item, int) or isinstance(item, bool) for item in expected_rates)
        or not isinstance(expected_channels, list)
        or any(not isinstance(item, int) or isinstance(item, bool) for item in expected_channels)
        or not isinstance(expected_widths, list)
        or any(not isinstance(item, int) or isinstance(item, bool) for item in expected_widths)
    ):
        raise AudioMetricContractError(
            "M23 policy sample_rate_hz/channels/sample_width_bits must be integer arrays"
        )
    decode_ok = (
        decoded["sample_rate_hz"] in expected_rates
        and decoded["channels"] in expected_channels
        and decoded["sample_width_bits"] in expected_widths
        and decoded["frame_count"] == segment["frame_count"]
    )
    facets["decode_format"] = _facet(decode_ok, "format_matches_channel_profile")

    rms_min = _require_finite_number(policy.get("rms_dbfs_min"), "M23 policy.rms_dbfs_min")
    rms_max = _require_finite_number(policy.get("rms_dbfs_max"), "M23 policy.rms_dbfs_max")
    peak_max = _require_finite_number(policy.get("peak_dbfs_max"), "M23 policy.peak_dbfs_max")
    facets["rms_peak"] = _facet(
        rms_min <= rms_dbfs <= rms_max and peak_dbfs <= peak_max,
        "RMS_and_peak_within_profile",
    )
    clip_max = _require_ratio(policy.get("clip_ratio_max"), "M23 policy.clip_ratio_max")
    facets["clipping"] = _facet(clip_ratio <= clip_max, "clip_ratio_within_profile")
    flat_top_max = _require_nonnegative_number(
        policy.get("flat_top_max_ms"), "M23 policy.flat_top_max_ms"
    )
    facets["flat_top"] = _facet(
        flat_top_ms <= flat_top_max,
        "same_sign_flat_top_within_profile",
    )

    speech_ranges = _validated_local_ranges(
        plan["speech_sample_ranges"], segment, "speech_sample_ranges"
    )
    noise_ranges = _validated_local_ranges(
        plan["noise_sample_ranges"], segment, "noise_sample_ranges"
    )
    if speech_ranges:
        near_zero = _require_nonnegative_number(
            policy.get("near_zero_abs_max"), "M23 policy.near_zero_abs_max"
        )
        dropout_max = _require_nonnegative_number(
            policy.get("dropout_max_ms"), "M23 policy.dropout_max_ms"
        )
        speech_samples = _concatenate_ranges(samples, speech_ranges)
        dropout_ms = (
            _longest_true_run(np.abs(speech_samples) <= near_zero)
            * MS_PER_SECOND
            / sample_rate
        )
        facets["dropout"] = _facet(dropout_ms <= dropout_max, "speech_dropout_within_profile")
    else:
        dropout_ms = None
        facets["dropout"] = _facet_nm("expected_speech_ranges_missing")

    if speech_ranges and noise_ranges:
        speech_samples = _concatenate_ranges(samples, speech_ranges)
        noise_samples = _concatenate_ranges(samples, noise_ranges)
        speech_rms = float(np.sqrt(np.mean(np.square(speech_samples))))
        noise_rms = float(np.sqrt(np.mean(np.square(noise_samples))))
        if noise_rms == 0.0:
            snr_db = math.inf
        else:
            snr_db = 20.0 * math.log10(speech_rms / noise_rms) if speech_rms else DB_FLOOR
        snr_min = _require_finite_number(policy.get("snr_min_db"), "M23 policy.snr_min_db")
        facets["snr"] = _facet(snr_db >= snr_min, "reference_or_noise_span_SNR_within_profile")
    else:
        snr_db = None
        facets["snr"] = _facet_nm("noise_only_or_expected_speech_ranges_missing")

    rolloff_min = _require_nonnegative_number(
        policy.get("rolloff_min_hz"), "M23 policy.rolloff_min_hz"
    )
    raw_rolloff_max = policy.get("rolloff_max_hz")
    rolloff_max = (
        _require_nonnegative_number(raw_rolloff_max, "M23 policy.rolloff_max_hz")
        if raw_rolloff_max is not None
        else math.inf
    )
    if rolloff_max < rolloff_min:
        raise AudioMetricContractError("M23 policy rolloff range is inverted")
    facets["bandwidth"] = _facet(
        rolloff_min <= rolloff_hz <= rolloff_max,
        "spectral_rolloff_within_channel_profile",
    )
    dc_max = _require_finite_number(policy.get("dc_max_dbfs"), "M23 policy.dc_max_dbfs")
    facets["dc"] = _facet(dc_dbfs <= dc_max, "DC_level_within_profile")
    facets["continuity"] = _facet(
        segment["complete"] and _chunks_delivered(context, segment, segment["boundary"]),
        "chunk_sequence_is_complete",
    )

    p56 = plan["p56_observation"]
    if p56 is None:
        facets["active_level"] = _facet_nm("verified_P56_observation_missing")
        p56_dbov = None
    else:
        _require_exact_fields(
            p56,
            required={"method", "conformance_verified", "active_level_dbov"},
            label=f"signal quality {plan['quality_id']}.p56_observation",
        )
        if (
            p56["method"] != "ITU-T P.56 Method B"
            or p56["conformance_verified"] is not True
        ):
            facets["active_level"] = _facet_nm("P56_observation_not_conformance_verified")
            p56_dbov = None
        else:
            p56_dbov = _require_finite_number(
                p56["active_level_dbov"], "p56_observation.active_level_dbov"
            )
            p56_min = _require_finite_number(
                policy.get("active_level_dbov_min"),
                "M23 policy.active_level_dbov_min",
            )
            p56_max = _require_finite_number(
                policy.get("active_level_dbov_max"),
                "M23 policy.active_level_dbov_max",
            )
            facets["active_level"] = _facet(
                p56_min <= p56_dbov <= p56_max,
                "verified_P56_active_level_within_profile",
            )

    inapplicable = policy.get("inapplicable_facets", {})
    if not isinstance(inapplicable, Mapping):
        raise AudioMetricContractError("M23 policy.inapplicable_facets must be an object")
    unknown_inapplicable = set(inapplicable) - set(facets)
    if unknown_inapplicable:
        raise AudioMetricContractError(
            f"M23 policy has unknown inapplicable facets: {sorted(unknown_inapplicable)}"
        )
    for facet_name, reason in inapplicable.items():
        if not isinstance(reason, str) or not reason:
            raise AudioMetricContractError(
                f"M23 policy.inapplicable_facets.{facet_name} must be a non-empty string"
            )
        facets[facet_name] = _facet_na(reason)

    observations = {
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "crest_factor_db": peak_dbfs - rms_dbfs,
        "clip_ratio": clip_ratio,
        "flat_top_max_ms": flat_top_ms,
        "dropout_max_ms": dropout_ms,
        "snr_db": snr_db,
        "spectral_rolloff_95_hz": rolloff_hz,
        "dc_dbfs": dc_dbfs,
        "p56_active_level_dbov": p56_dbov,
    }
    return facets, observations


def _selected_policy_or_none(raw: Any, metric_id: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise AudioMetricContractError(f"{metric_id} config must be an object or null")
    if set(raw) != {"selected_profile"}:
        raise AudioMetricContractError(f"{metric_id} config only accepts selected_profile")
    selected = raw["selected_profile"]
    if selected is None:
        return None
    if not isinstance(selected, dict):
        raise AudioMetricContractError(f"{metric_id} selected_profile must be an object or null")
    _require_nonempty_string(selected.get("profile_id"), f"{metric_id} policy.profile_id")
    return selected


def _event(context: dict[str, Any], event_id: Any, label: str) -> dict[str, Any]:
    if not isinstance(event_id, str) or not event_id:
        raise AudioMetricContractError(f"{label} must be a non-empty event ID")
    event = context["events"].get(event_id)
    if event is None:
        raise AudioMetricContractError(f"{label} references missing event {event_id}")
    return event


def _chunks_delivered(context: dict[str, Any], segment: dict[str, Any], boundary: str) -> bool:
    required_event_type = {
        "provider": (
            "audio_chunk_played_or_virtual_played"
            if segment["direction"] == "output"
            else "audio_chunk_received"
        ),
        "receiver": "audio_chunk_received",
        "playback": "audio_chunk_played_or_virtual_played",
    }[boundary]
    seen: set[str] = set()
    for event in context["events"].values():
        if event["audio_segment_id"] != segment["audio_segment_id"]:
            continue
        data = event.get("data", {})
        chunk_id = data.get("chunk_sequence_id")
        if event["event_type"] == "audio_chunk_dropped" and chunk_id in segment["chunk_sequence_ids"]:
            return False
        if event["event_type"] == required_event_type and chunk_id in segment["chunk_sequence_ids"]:
            seen.add(chunk_id)
    return seen == set(segment["chunk_sequence_ids"])


def _validated_local_ranges(
    ranges: Any,
    segment: dict[str, Any],
    label: str,
) -> list[tuple[int, int]]:
    if not isinstance(ranges, list):
        raise AudioMetricContractError(f"{label} must be an array")
    local: list[tuple[int, int]] = []
    for index, item in enumerate(ranges):
        _validate_range(item, f"{label}[{index}]")
        _require_range_within(item, segment["sample_range"], f"{label}[{index}]")
        local.append(
            (
                item[0] - segment["sample_range"][0],
                item[1] - segment["sample_range"][0],
            )
        )
    return local


def _concatenate_ranges(samples: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([samples[start:end] for start, end in ranges])


def _flat_top_max_samples(samples: np.ndarray, clipped: np.ndarray) -> int:
    if samples.size == 0:
        return 0
    maximum = 0
    current = 0
    previous = None
    for sample, is_clipped in zip(samples, clipped, strict=True):
        if is_clipped and previous is not None and sample == previous:
            current += 1
        elif is_clipped:
            current = 1
        else:
            current = 0
        maximum = max(maximum, current)
        previous = sample
    return maximum


def _longest_true_run(mask: np.ndarray) -> int:
    maximum = 0
    current = 0
    for item in mask:
        if bool(item):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _spectral_rolloff(
    frequencies: np.ndarray,
    powers: np.ndarray,
    fraction: float,
) -> float:
    total = float(np.sum(powers))
    if total <= 0.0:
        return 0.0
    index = int(np.searchsorted(np.cumsum(powers), total * fraction))
    index = min(index, frequencies.size - 1)
    return float(frequencies[index])


def _threshold_interval_max(lower: float, upper: float, threshold: float) -> tuple[str, str]:
    if upper <= threshold:
        return "passed", "latency_interval_within_maximum"
    if lower > threshold:
        return "failed", "latency_interval_exceeds_maximum"
    return "N/M", "clock_uncertainty_crosses_latency_threshold"


def _threshold_interval_min(lower: float, upper: float, threshold: float) -> tuple[str, str]:
    if lower >= threshold:
        return "passed", "hold_interval_meets_minimum"
    if upper < threshold:
        return "failed", "hold_interval_below_minimum"
    return "N/M", "clock_uncertainty_crosses_latency_threshold"


def _metric_result(metric_id: str, units: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [unit["status"] for unit in units]
    if not units or all(status == "N/A" for status in statuses):
        status = "N/A"
        reason = "no_applicable_units"
    elif any(status == "failed" for status in statuses):
        status = "failed"
        reason = "one_or_more_units_failed"
    elif any(status == "contract_invalid" for status in statuses):
        status = "contract_invalid"
        reason = "one_or_more_units_contract_invalid"
    elif any(status == "N/M" for status in statuses):
        status = "N/M"
        reason = "one_or_more_applicable_units_not_measurable"
    else:
        status = "passed"
        reason = "all_applicable_measurable_units_passed"
    coverage = {
        "total": len(units),
        "applicable": sum(unit["status"] != "N/A" for unit in units),
        "measurable": sum(unit["status"] in {"passed", "failed"} for unit in units),
        "passed": sum(unit["status"] == "passed" for unit in units),
        "failed": sum(unit["status"] == "failed" for unit in units),
        "N/A": sum(unit["status"] == "N/A" for unit in units),
        "N/M": sum(unit["status"] == "N/M" for unit in units),
    }
    return {
        "metric_id": metric_id,
        "status": status,
        "passed": status == "passed" if status in {"passed", "failed"} else None,
        "reason": reason,
        "coverage": coverage,
        "units": units,
    }


def _apply_evidence_gaps(
    results: dict[str, dict[str, Any]],
    raw_gaps: Any,
) -> None:
    """Turn declared no-unit evidence gaps into explicit N/M units.

    ``raw_gaps`` may be null or an object whose keys are M21/M23 and values
    are non-empty string arrays. Unknown metrics, malformed arrays, or gaps
    declared alongside applicable units raise ``AudioMetricContractError``.
    """

    if raw_gaps is None:
        return
    if not isinstance(raw_gaps, Mapping):
        raise AudioMetricContractError("config.evidence_gaps must be an object")
    unknown = set(raw_gaps) - set(METRIC_IDS)
    if unknown:
        raise AudioMetricContractError(
            f"config.evidence_gaps has unknown metrics: {sorted(unknown)}"
        )
    for metric_id, reasons in raw_gaps.items():
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise AudioMetricContractError(
                f"config.evidence_gaps.{metric_id} must be a non-empty string array"
            )
        result = results[metric_id]
        if result["units"]:
            raise AudioMetricContractError(
                f"config.evidence_gaps.{metric_id} cannot coexist with metric units"
            )
        units = [
            _unit(
                f"{metric_id.lower()}-evidence-gap-{index}",
                "N/M",
                reason,
            )
            for index, reason in enumerate(reasons, start=1)
        ]
        results[metric_id] = _metric_result(metric_id, units)


def _contract_invalid(metric_id: str, reason: str) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "status": "contract_invalid",
        "passed": None,
        "reason": reason,
        "coverage": {
            "total": 0,
            "applicable": 0,
            "measurable": 0,
            "passed": 0,
            "failed": 0,
            "N/A": 0,
            "N/M": 0,
        },
        "units": [],
    }


def _unit(
    unit_id: str,
    status: str,
    reason: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in RESULT_STATUSES:
        raise RuntimeError(f"invalid internal metric status: {status}")
    return {
        "unit_id": unit_id,
        "status": status,
        "passed": status == "passed" if status in {"passed", "failed"} else None,
        "reason": reason,
        "diagnostics": diagnostics or {},
    }


def _facet(passed: bool, reason: str) -> dict[str, Any]:
    return {
        "applicable": True,
        "measurable": True,
        "status": "passed" if passed else "failed",
        "reason": reason,
    }


def _facet_nm(reason: str) -> dict[str, Any]:
    return {
        "applicable": True,
        "measurable": False,
        "status": "N/M",
        "reason": reason,
    }


def _facet_na(reason: str) -> dict[str, Any]:
    return {
        "applicable": False,
        "measurable": False,
        "status": "N/A",
        "reason": reason,
    }


def _require_exact_fields(
    value: Any,
    *,
    required: set[str],
    label: str,
    optional: set[str] | None = None,
) -> None:
    if not isinstance(value, dict):
        raise AudioMetricContractError(f"{label} must be an object")
    allowed = required | (optional or set())
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing or unknown:
        raise AudioMetricContractError(
            f"{label} fields invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AudioMetricContractError(f"{label} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AudioMetricContractError(f"{label} must be a nonnegative integer")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioMetricContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AudioMetricContractError(f"{label} must be a finite number")
    return result


def _require_nonnegative_number(value: Any, label: str) -> float:
    result = _require_finite_number(value, label)
    if result < 0:
        raise AudioMetricContractError(f"{label} must be nonnegative")
    return result


def _require_ratio(value: Any, label: str) -> float:
    result = _require_finite_number(value, label)
    if not 0.0 <= result <= 1.0:
        raise AudioMetricContractError(f"{label} must be between 0 and 1")
    return result


def _validate_range(value: Any, label: str, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] < 0
        or value[1] <= value[0]
    ):
        raise AudioMetricContractError(f"{label} must be [nonnegative_start, greater_end]")


def _require_range_within(inner: list[int], outer: list[int], label: str) -> None:
    if inner[0] < outer[0] or inner[1] > outer[1]:
        raise AudioMetricContractError(f"{label} is outside its audio segment")


def _validate_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AudioMetricContractError(f"{label} must be a lowercase SHA-256 digest")


def _normalized_sha256(value: str) -> str:
    match = SHA256_RE.fullmatch(value)
    if match is None:
        raise AudioMetricContractError("invalid SHA-256 digest")
    return match.group(1)


def _db(amplitude: float) -> float:
    if amplitude <= 0.0:
        return DB_FLOOR
    return 20.0 * math.log10(amplitude)
