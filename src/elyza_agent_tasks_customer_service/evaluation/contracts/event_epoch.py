"""Call/post-call event boundary validation and M11 input projection."""

from __future__ import annotations

from typing import Any

from elyza_agent_tasks_customer_service.evaluation.contracts.artifact_receipts import sha256_json


CALL_EPOCH = "call"
POST_CALL_EPOCH = "post_call"
VALID_EPOCHS = {CALL_EPOCH, POST_CALL_EPOCH}


def validate_event_epoch_order(events: Any) -> dict[str, Any]:
    """Validate ordered object events using call then post-call epochs.

    Events without an explicit epoch are treated as call events.
    Unknown epochs, non-object rows, non-increasing sequence numbers, or a call
    event after the first post-call event raise ``ValueError``.
    """

    if not isinstance(events, list):
        raise ValueError("event log must be an array")
    previous_seq: int | None = None
    post_call_started = False
    call_events: list[dict[str, Any]] = []
    post_call_events: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"event log row[{index}] must be an object")
        epoch = event.get("epoch", CALL_EPOCH)
        if epoch not in VALID_EPOCHS:
            raise ValueError(f"event log row[{index}] has unknown epoch: {epoch}")
        seq = event.get("seq")
        if seq is not None:
            if not isinstance(seq, int) or isinstance(seq, bool):
                raise ValueError(f"event log row[{index}].seq must be an integer")
            if previous_seq is not None and seq <= previous_seq:
                raise ValueError("event log sequence must be strictly increasing")
            previous_seq = seq
        if epoch == POST_CALL_EPOCH:
            post_call_started = True
            post_call_events.append(event)
        else:
            if post_call_started:
                raise ValueError("call event appears after post-call epoch started")
            call_events.append(event)
    return {
        "call_events": call_events,
        "post_call_events": post_call_events,
        "call_event_log_hash": sha256_json(call_events),
        "post_call_event_log_hash": sha256_json(post_call_events),
    }


