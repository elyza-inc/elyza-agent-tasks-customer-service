"""Build conservative M21/M23 input artifacts from one persisted run.

``build_audio_evidence_artifacts`` accepts one existing run directory and
already-decoded object rows from a runtime-produced event log. Rows may be
empty. The builder writes three object-root JSON files below the run directory:
``audio_evidence_manifest.json``, ``audio_events.json``, and
``audio_metric_config.json``. Unknown row fields are retained only in a source
hash; timestamps, metric opportunities, and audio boundaries are never
inferred. Invalid roots raise ``ValueError``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.audio.audio_signal_metrics import inspect_wav
from elyza_agent_tasks_customer_service.evaluation.audio.omni_audio_common import write_json


BUILDER_VERSION = "audio_evidence_builder"
AUDIO_MANIFEST_FILE = "audio_evidence_manifest.json"
AUDIO_EVENTS_FILE = "audio_events.json"
AUDIO_CONFIG_FILE = "audio_metric_config.json"
EVIDENCE_GAP_REASON_BY_METRIC = {
    "M21": "monotonic_turn_control_boundaries_missing",
    "M23": "eligible_signal_segment_boundary_or_threshold_profile_missing",
}
PLAN_KEY_BY_METRIC = {
    "M21": "turn_probes",
    "M23": "signal_quality_segments",
}
PLAN_ENTRY_FIELDS = {
    "turn_probes": frozenset(
        (
            "probe_id", "applicable", "expected_control",
            "observed_control_event_id", "latency_kind", "start_event_id",
            "end_event_id",
        )
    ),
    "signal_quality_segments": frozenset(
        (
            "quality_id", "applicable", "audio_segment_id",
            "speech_sample_ranges", "noise_sample_ranges", "p56_observation",
        )
    ),
}
PLAN_ENTRY_ID_FIELDS = {
    "turn_probes": "probe_id",
    "signal_quality_segments": "quality_id",
}


def build_audio_evidence_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    scenario_id: str,
    route: str,
    source_rows: list[dict[str, Any]],
    scenario_contract: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write conservative audio scorer inputs and return relative references.

    ``run_dir`` must be an existing directory. ``run_id``, ``scenario_id``,
    and ``route`` must be non-empty strings. ``source_rows`` must be an array
    of objects decoded from a runtime-produced event log. This version does
    not accept implicit wall-clock-to-monotonic conversion.
    ``scenario_contract`` is an ``audio_measurement_contract`` (see
    ``audio_contract_materializer``) or ``None`` for an empty plan. Malformed
    inputs raise ``ValueError``.
    """

    if not isinstance(run_dir, Path) or not run_dir.is_dir():
        raise ValueError(f"run_dir must be an existing directory: {run_dir}")
    for label, value in (
        ("run_id", run_id),
        ("scenario_id", scenario_id),
        ("route", route),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")
    if not isinstance(source_rows, list) or any(
        not isinstance(row, dict) for row in source_rows
    ):
        raise ValueError("source_rows must be an object array")

    source_hash = _sha256_json(source_rows)
    builder_hash = _sha256_bytes(Path(__file__).read_bytes())
    segments, events, segment_clock_ids = extract_measured_audio_evidence(
        run_dir=run_dir,
        rows=source_rows,
    )
    plan = _scenario_plan_or_empty(scenario_contract)
    if _plan_entries_are_well_formed(plan):
        plan = _filter_unjoinable_plan(
            plan=plan,
            segment_ids={row["audio_segment_id"] for row in segments},
            event_ids={row["event_id"] for row in events},
            delivered_segment_ids={
                row["audio_segment_id"]
                for row in events
                if row["event_type"]
                in {"audio_chunk_received", "audio_chunk_played_or_virtual_played"}
                and row["audio_segment_id"] is not None
            },
        )
    evidence_gaps = {
        metric_id: [EVIDENCE_GAP_REASON_BY_METRIC[metric_id]]
        for metric_id, plan_key in PLAN_KEY_BY_METRIC.items()
        if not plan[plan_key]
    }
    config = {
        "schema_version": "audio_metric_config.v1",
        "m21": {"selected_profile": None},
        "m23": {"selected_profile": None},
        "evidence_gaps": evidence_gaps,
    }
    clock = _clock_receipt(
        source_rows,
        events=events,
        segment_clock_ids=segment_clock_ids,
    )
    audio_manifest = {
        "schema_version": "audio_evidence_manifest.v1",
        "run_id": run_id,
        "scenario_id": scenario_id,
        "route": route,
        "condition_hashes": {
            "runtime": source_hash,
            "scorer": builder_hash,
            "audio_policy": _sha256_json(config),
        },
        "clock": clock,
        "audio_segments": segments,
        "observer_versions": {
            "builder": {
                "name": BUILDER_VERSION,
                "version": BUILDER_VERSION,
                "hash": builder_hash,
            },
            "source_rows_hash": source_hash,
        },
        "metric_plan": plan,
    }
    audio_events = {
        "schema_version": "audio_events.v1",
        "events": events,
    }
    payloads = {
        AUDIO_MANIFEST_FILE: audio_manifest,
        AUDIO_EVENTS_FILE: audio_events,
        AUDIO_CONFIG_FILE: config,
    }
    for file_name, payload in payloads.items():
        write_json(run_dir / file_name, payload)
    return {
        "audio_manifest": AUDIO_MANIFEST_FILE,
        "event_log": AUDIO_EVENTS_FILE,
        "config": AUDIO_CONFIG_FILE,
        "evidence_root": ".",
    }


def extract_measured_audio_evidence(
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Return measured segments, sorted events, and segment clock IDs from runtime rows."""

    segments: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    segment_clock_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        raw_segment = metadata.get("audio_metric_segment")
        segment_id: str | None = None
        if isinstance(raw_segment, dict):
            segment = _materialize_segment(run_dir, raw_segment)
            if segment is not None:
                segment_clock_ids.add(raw_segment["clock_id"])
                segment_id = segment["audio_segment_id"]
                if segment_id in segments and segments[segment_id] != segment:
                    raise ValueError(f"conflicting measured audio segment: {segment_id}")
                segments[segment_id] = segment
        elif isinstance(metadata.get("audio_segment_id"), str):
            segment_id = metadata["audio_segment_id"]
        monotonic_ns = row.get("monotonic_ns")
        clock_id = row.get("clock_id")
        if isinstance(monotonic_ns, int) and monotonic_ns >= 0 and isinstance(
            clock_id, str
        ):
            events.append(
                {
                    "event_id": f"runtime:{index}",
                    "seq": len(events) + 1,
                    "event_type": str(row.get("event_type") or "runtime_event"),
                    "clock_id": clock_id,
                    "monotonic_ns": monotonic_ns,
                    "audio_segment_id": segment_id,
                    "sample_range": None,
                    "data": {
                        **metadata,
                        "actor": row.get("actor"),
                        "content": row.get("content"),
                        **(
                            {"tool": row.get("tool"), "arguments": row.get("arguments")}
                            if row.get("tool") is not None
                            else {}
                        ),
                        **(
                            {"result": row.get("result")}
                            if row.get("result") is not None
                            else {}
                        ),
                    },
                }
            )
        embedded_events = metadata.get("audio_metric_events")
        if isinstance(embedded_events, list):
            for embedded_index, embedded in enumerate(embedded_events, start=1):
                if not isinstance(embedded, dict):
                    raise ValueError("audio_metric_events must be an object array")
                events.append(
                    _materialize_embedded_event(
                        embedded,
                        event_id=f"runtime:{index}:audio:{embedded_index}",
                        seq=len(events) + 1,
                    )
                )
    materialized_events = [
        row
        for row in events
        if row["audio_segment_id"] is None or row["audio_segment_id"] in segments
    ]
    materialized_events.sort(key=lambda event: event["monotonic_ns"])
    for event_seq, event in enumerate(materialized_events, start=1):
        event["seq"] = event_seq
    return (
        sorted(segments.values(), key=lambda row: row["audio_segment_id"]),
        materialized_events,
        segment_clock_ids,
    )


def _materialize_embedded_event(
    value: dict[str, Any],
    *,
    event_id: str,
    seq: int,
) -> dict[str, Any]:
    required = {
        "event_type",
        "clock_id",
        "monotonic_ns",
        "audio_segment_id",
        "chunk_sequence_id",
    }
    if set(value) - {"control"} != required:
        raise ValueError("audio_metric_events event fields are invalid")
    if not isinstance(value["monotonic_ns"], int) or value["monotonic_ns"] < 0:
        raise ValueError("audio_metric_events monotonic_ns is invalid")
    return {
        "event_id": event_id,
        "seq": seq,
        "event_type": value["event_type"],
        "clock_id": value["clock_id"],
        "monotonic_ns": value["monotonic_ns"],
        "audio_segment_id": value["audio_segment_id"],
        "sample_range": None,
        "data": {
            "chunk_sequence_id": value["chunk_sequence_id"],
            **({"control": value["control"]} if "control" in value else {}),
        },
    }


def _materialize_segment(
    run_dir: Path,
    value: dict[str, Any],
) -> dict[str, Any] | None:
    required = {
        "audio_segment_id",
        "speaker",
        "direction",
        "boundary",
        "audio_path",
        "clock_id",
        "monotonic_range_ns",
        "complete",
        "chunk_sequence_ids",
    }
    if set(value) != required:
        raise ValueError("audio_metric_segment fields are invalid")
    if not isinstance(value["clock_id"], str) or not value["clock_id"]:
        raise ValueError("audio_metric_segment.clock_id is invalid")
    audio_path = Path(str(value["audio_path"])).resolve()
    try:
        relative_path = audio_path.relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return None
    if not audio_path.is_file():
        return None
    observed = inspect_wav(audio_path)
    monotonic_range = value["monotonic_range_ns"]
    if (
        not isinstance(monotonic_range, list)
        or len(monotonic_range) != 2
        or any(not isinstance(item, int) or item < 0 for item in monotonic_range)
        or monotonic_range[1] < monotonic_range[0]
    ):
        raise ValueError("audio_metric_segment.monotonic_range_ns is invalid")
    return {
        "audio_segment_id": str(value["audio_segment_id"]),
        "speaker": value["speaker"],
        "direction": value["direction"],
        "boundary": value["boundary"],
        "relative_path": relative_path,
        "wav_sha256": observed["wav_sha256"],
        "canonical_pcm_sha256": observed["canonical_pcm_sha256"],
        "sample_rate_hz": observed["sample_rate_hz"],
        "channels": observed["channels"],
        "sample_width_bits": observed["sample_width_bits"],
        "frame_count": observed["frame_count"],
        "duration_ns": observed["duration_ns"],
        "sample_range": [0, observed["frame_count"]],
        "monotonic_range_ns": monotonic_range,
        "complete": value["complete"],
        "chunk_sequence_ids": value["chunk_sequence_ids"],
    }


def _scenario_plan_or_empty(
    value: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    empty: dict[str, list[dict[str, Any]]] = {key: [] for key in PLAN_ENTRY_FIELDS}
    if value is None:
        return empty
    if not isinstance(value, dict) or set(value) != {"schema_version", "metric_plan"}:
        raise ValueError("audio_measurement_contract fields are invalid")
    if value["schema_version"] != "audio_measurement_contract":
        raise ValueError("audio measurement contract version is invalid")
    plan = value["metric_plan"]
    if not isinstance(plan, dict) or set(plan) != set(empty):
        raise ValueError("audio measurement metric_plan fields are invalid")
    if any(not isinstance(plan[key], list) for key in empty):
        raise ValueError("audio measurement metric_plan values must be arrays")
    if any(
        not isinstance(row, dict)
        for key in empty
        for row in plan[key]
    ):
        raise ValueError("audio measurement metric_plan entries must be objects")
    return {key: list(plan[key]) for key in empty}


def _filter_unjoinable_plan(
    *,
    plan: dict[str, list[dict[str, Any]]],
    segment_ids: set[str],
    event_ids: set[str],
    delivered_segment_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "turn_probes": [
            row
            for row in plan["turn_probes"]
            if row.get("observed_control_event_id") in event_ids
            and row.get("start_event_id") in event_ids
            and row.get("end_event_id") in event_ids
        ],
        "signal_quality_segments": [
            row
            for row in plan["signal_quality_segments"]
            if row.get("audio_segment_id") in segment_ids
            and row.get("audio_segment_id") in delivered_segment_ids
        ],
    }


def _plan_entries_are_well_formed(
    plan: dict[str, list[dict[str, Any]]],
) -> bool:
    """Return whether plan rows have the scorer's required structural shape."""

    for plan_key, rows in plan.items():
        expected_fields = PLAN_ENTRY_FIELDS[plan_key]
        id_field = PLAN_ENTRY_ID_FIELDS[plan_key]
        identifiers: list[str] = []
        for row in rows:
            if set(row) != expected_fields:
                return False
            identifier = row.get(id_field)
            if not isinstance(identifier, str) or not identifier:
                return False
            identifiers.append(identifier)
            if not _plan_entry_field_types_are_valid(plan_key, row):
                return False
        if len(set(identifiers)) != len(identifiers):
            return False
    return True


def _plan_entry_field_types_are_valid(
    plan_key: str,
    row: dict[str, Any],
) -> bool:
    if plan_key == "turn_probes":
        string_fields = {
            "expected_control", "observed_control_event_id", "start_event_id",
            "end_event_id",
        }
        latency_kind = row["latency_kind"]
        return (
            all(isinstance(row[field], str) and row[field] for field in string_fields)
            and isinstance(row["applicable"], bool)
            and (latency_kind is None or isinstance(latency_kind, str))
        )
    return (
        isinstance(row["applicable"], bool)
        and isinstance(row["audio_segment_id"], str)
        and bool(row["audio_segment_id"])
        and isinstance(row["speech_sample_ranges"], list)
        and isinstance(row["noise_sample_ranges"], list)
        and (row["p56_observation"] is None or isinstance(row["p56_observation"], dict))
    )


def _clock_receipt(
    rows: list[dict[str, Any]],
    *,
    events: list[dict[str, Any]],
    segment_clock_ids: set[str],
) -> dict[str, Any]:
    clock_rows = [
        row
        for row in rows
        if isinstance(row.get("clock_id"), str)
        and isinstance(row.get("monotonic_ns"), int)
    ]
    clock_ids = {row["clock_id"] for row in clock_rows}
    clock_ids.update(
        event["clock_id"]
        for event in events
        if isinstance(event.get("clock_id"), str)
    )
    clock_ids.update(segment_clock_ids)
    if not clock_ids:
        return {"clock_id": "no_monotonic_audio_clock_evidence", "measured_uncertainty_ns": 0}
    if len(clock_ids) != 1:
        raise ValueError(f"runtime audio evidence mixes clocks: {sorted(clock_ids)}")
    # Runtime rows carry no clock uncertainty measurement, so M21 uses 0.
    return {"clock_id": next(iter(clock_ids)), "measured_uncertainty_ns": 0}


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def transcribe_audio_profile(*, wav_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    """Transcribe one WAV with one asr_profiles entry via the shared ASR client."""

    from elyza_agent_tasks_customer_service.evaluation.core.asr_runtime import AsrProfileConfig, OpenAICompatibleAsrClient

    config = AsrProfileConfig(
        profile=str(profile["profile"]),
        endpoint=str(profile["endpoint"]),
        model=str(profile["model"]),
        file_field_name=str(profile.get("file_field_name", "file")),
        request_fields={str(k): str(v) for k, v in (profile.get("request_fields") or {}).items()},
        api_key_env=profile.get("api_key_env"),
        timeout_sec=float(profile.get("timeout_sec", 600.0)),
    )
    result = OpenAICompatibleAsrClient(config).transcribe(wav_path)
    return {"transcript": result.transcript, "profile": result.profile, "model": result.model}
