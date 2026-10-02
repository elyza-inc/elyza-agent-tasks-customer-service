"""Materialize the M21/M23 audio measurement plan from persisted run evidence.

Turn probes and signal-quality segments are built only from measured
segment/event IDs; malformed inputs raise ``ValueError``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from elyza_agent_tasks_customer_service.evaluation.audio.omni_audio_common import write_json


CONTRACT_FILE = "audio_measurement_contract.materialized.json"


def materialize_audio_measurement_contract(
    *,
    audio_events: Mapping[str, Any] | str | Path,
    audio_evidence_manifest: Mapping[str, Any] | str | Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a runtime-ID-bound ``audio_measurement_contract`` object for M21/M23."""

    events_root = _load_object(audio_events, "audio_events")
    audio = _load_object(audio_evidence_manifest, "audio_evidence_manifest")
    events = events_root.get("events")
    segments = audio.get("audio_segments")
    if not isinstance(events, list) or any(not isinstance(row, dict) for row in events):
        raise ValueError("audio_events.events must be an object array")
    if not isinstance(segments, list) or any(not isinstance(row, dict) for row in segments):
        raise ValueError("audio_evidence_manifest.audio_segments must be an object array")
    candidate_segments = {
        row["audio_segment_id"]: row
        for row in segments
        if isinstance(row.get("audio_segment_id"), str)
    }
    customer_segments = {
        segment_id
        for segment_id, row in candidate_segments.items()
        if row.get("speaker") == "user" and row.get("direction") == "input"
    }
    operator_segments = {
        segment_id: row
        for segment_id, row in candidate_segments.items()
        if row.get("speaker") == "operator"
        and row.get("direction") == "output"
        and row.get("boundary") in {"provider", "receiver", "playback"}
    }
    delivered = {
        row.get("audio_segment_id")
        for row in events
        if row.get("event_type") in {
            "audio_chunk_received",
            "audio_chunk_played_or_virtual_played",
        }
    }
    quality = [
        {
            "quality_id": f"materialized-quality-{index}",
            "applicable": True,
            "audio_segment_id": segment_id,
            "speech_sample_ranges": [list(operator_segments[segment_id]["sample_range"])],
            "noise_sample_ranges": [],
            "p56_observation": None,
        }
        for index, segment_id in enumerate(
            sorted(set(operator_segments) & delivered if delivered else set(operator_segments)),
            start=1,
        )
    ]
    probes = _materialized_turn_probes(
        events,
        customer_segment_ids=customer_segments,
        operator_segment_ids=set(operator_segments),
    )
    contract = {
        "schema_version": "audio_measurement_contract",
        "metric_plan": {
            "turn_probes": probes,
            "signal_quality_segments": quality,
        },
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / CONTRACT_FILE, contract)
    return contract


def _load_object(value: Mapping[str, Any] | str | Path, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        loaded: Any = dict(value)
    else:
        loaded = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must have an object root")
    return loaded


def _materialized_turn_probes(
    events: list[dict[str, Any]],
    *,
    customer_segment_ids: set[str],
    operator_segment_ids: set[str],
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    ordered = sorted(
        enumerate(events),
        key=lambda item: (
            item[1].get("seq") if isinstance(item[1].get("seq"), int) else item[0],
            item[0],
        ),
    )
    latest_customer_receipt: dict[str, Any] | None = None
    for position, (_index, event) in enumerate(ordered):
        segment_id = event.get("audio_segment_id")
        if (
            event.get("event_type") == "audio_chunk_received"
            and segment_id in customer_segment_ids
        ):
            latest_customer_receipt = event
            continue
        if (
            event.get("event_type") != "response_started"
            or segment_id not in operator_segment_ids
            or latest_customer_receipt is None
        ):
            continue
        end = next(
            (
                candidate
                for _candidate_index, candidate in ordered[position + 1:]
                if candidate.get("event_type")
                == "audio_chunk_played_or_virtual_played"
                and candidate.get("audio_segment_id") == segment_id
            ),
            None,
        )
        if end is None:
            continue
        probes.append(
            {
                "probe_id": f"materialized-response-{event['event_id']}",
                "applicable": True,
                "expected_control": "yield_and_respond",
                "observed_control_event_id": event["event_id"],
                "latency_kind": "response",
                "start_event_id": latest_customer_receipt["event_id"],
                "end_event_id": end["event_id"],
            }
        )
    return probes
