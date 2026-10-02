"""Assign stable scenario aliases to concrete runtime events.

The helpers accept one object-root scenario and concrete runtime event fields.
Only deterministic matches are emitted. Repeated runtime Tool retries align
from the deadline side, and paraphrased messages align by actor and order
between Tool anchors. M05 deadline, required, and protected-action refs
instead pin the first successful runtime call of their Tool, and M05 consent pins the last customer
message before that protected action. Unknown or ambiguous events return an empty alias array.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


IDEAL_EVENT_ALIAS_PREFIX = "ideal_conversation:"
IDEAL_TOOL_RESULT_ALIAS_SUFFIX = ":tool_result"
CANONICAL_ACTOR_BY_RUNTIME_ACTOR = {
    "agent": "assistant",
    "assistant": "assistant",
    "customer": "user",
    "operator": "assistant",
    "user": "user",
}
INTERACTION_VALUE_SEPARATOR_PATTERN = re.compile(r"[\s　、。,.，．:：\-ー－_/]")


def align_referenced_ideal_event_aliases(
    scenario: dict[str, Any],
    call_events: list[dict[str, Any]],
    referenced_event_ids: set[str],
) -> list[dict[str, Any]]:
    """Return call events with unambiguous referenced ideal aliases attached.

    ``scenario`` accepts an object with an optional object-array
    ``_ideal_conversation``. ``call_events`` must be an object array whose
    optional ``event_aliases`` values are string arrays. Referenced Tool turns
    align by Tool name and occurrence order from the final occurrence.
    Referenced utterances align by actor and ordered position within the
    surrounding Tool interval. Consolidated or omitted trailing messages do
    not discard an otherwise ordered prefix. Malformed event arrays raise
    ``ValueError``; absent, ambiguous, or non-ideal refs are left unresolved.
    """

    if not isinstance(scenario, dict):
        raise ValueError("scenario must be an object")
    if not isinstance(call_events, list) or any(
        not isinstance(event, dict) for event in call_events
    ):
        raise ValueError("call_events must be an object array")
    if not isinstance(referenced_event_ids, set) or any(
        not isinstance(ref, str) for ref in referenced_event_ids
    ):
        raise ValueError("referenced_event_ids must be a string set")
    events = deepcopy(call_events)
    conversation = scenario.get("_ideal_conversation")
    if not isinstance(conversation, list):
        return events
    ideal_refs = _referenced_ideal_indices(
        referenced_event_ids,
        conversation_length=len(conversation),
    )
    ideal_result_refs = _referenced_ideal_tool_result_indices(
        referenced_event_ids,
        conversation_length=len(conversation),
    )
    if not ideal_refs and not ideal_result_refs:
        return events
    aliases_by_event = _validated_aliases_by_event(events)
    alias_owners = {
        alias: event_index
        for event_index, aliases in aliases_by_event.items()
        for alias in aliases
    }
    _attach_m05_first_call_aliases(
        scenario,
        conversation,
        events,
        aliases_by_event,
        alias_owners,
        referenced_event_ids,
    )
    tool_anchors, ideal_tool_indices = _ideal_tool_anchors(conversation, events)
    tool_result_by_call_event = _tool_result_events_by_call_event(events)
    for ideal_index, event_index in tool_anchors.items():
        ref = f"{IDEAL_EVENT_ALIAS_PREFIX}{ideal_index}"
        if ideal_index in ideal_refs:
            _attach_alias(
                events,
                aliases_by_event,
                alias_owners,
                event_index=event_index,
                alias=ref,
            )
        result_ref = f"{ref}{IDEAL_TOOL_RESULT_ALIAS_SUFFIX}"
        result_event_index = tool_result_by_call_event.get(event_index)
        if ideal_index in ideal_result_refs and result_event_index is not None:
            _attach_alias(
                events,
                aliases_by_event,
                alias_owners,
                event_index=result_event_index,
                alias=result_ref,
            )
    message_groups: dict[tuple[int | None, int | None, str], list[int]] = {}
    for ideal_index, turn in enumerate(conversation):
        if not isinstance(turn, dict) or not isinstance(turn.get("utterance"), str):
            continue
        actor = CANONICAL_ACTOR_BY_RUNTIME_ACTOR.get(
            str(turn.get("speaker") or ""),
            str(turn.get("speaker") or ""),
        )
        prior_tool = max(
            (index for index in ideal_tool_indices if index < ideal_index),
            default=None,
        )
        next_tool = min(
            (index for index in ideal_tool_indices if index > ideal_index),
            default=None,
        )
        if prior_tool is not None and prior_tool not in tool_anchors:
            continue
        message_groups.setdefault((prior_tool, next_tool, actor), []).append(
            ideal_index
        )
    for (prior_tool, next_tool, actor), ideal_indices in message_groups.items():
        if not any(index in ideal_refs for index in ideal_indices):
            continue
        runtime_start = tool_anchors.get(prior_tool, -1)
        runtime_end = tool_anchors.get(next_tool, len(events))
        candidates = [
            index
            for index, event in enumerate(events)
            if runtime_start < index < runtime_end
            and event.get("event_type") == "message"
            and CANONICAL_ACTOR_BY_RUNTIME_ACTOR.get(
                str(event.get("actor") or ""),
                str(event.get("actor") or ""),
            )
            == actor
        ]
        for ideal_index, event_index in zip(
            sorted(ideal_indices), candidates
        ):
            if ideal_index not in ideal_refs:
                continue
            _attach_alias(
                events,
                aliases_by_event,
                alias_owners,
                event_index=event_index,
                alias=f"{IDEAL_EVENT_ALIAS_PREFIX}{ideal_index}",
            )
    _attach_declared_interaction_aliases(
        scenario,
        conversation,
        events,
        aliases_by_event,
        alias_owners,
    )
    return events


def _attach_m05_first_call_aliases(
    scenario: dict[str, Any],
    conversation: list[Any],
    events: list[dict[str, Any]],
    aliases_by_event: dict[int, list[str]],
    alias_owners: dict[str, int],
    referenced_event_ids: set[str],
) -> None:
    """Pin M05 deadline, required, protected action, and consent refs first.

    The package adapter declares each of these Tool refs at the first gold
    occurrence of its Tool, so each pins the first runtime call of that Tool,
    the same rule ``score_m05_obligations`` uses to observe the trigger. Consent is the last customer message before
    that protected action. Refs whose gold turn is not a Tool call, or whose
    Tool is never called, stay unresolved.
    """

    contract = scenario.get("m05_obligation_contract")
    obligations = contract.get("obligations") if isinstance(contract, dict) else None
    if not isinstance(obligations, list):
        return
    for obligation in obligations:
        if not isinstance(obligation, dict):
            continue
        required = obligation.get("required_event_ids")
        tool_refs = [obligation.get("deadline_event_id"), obligation.get("protected_action_event_id")]
        tool_refs += required if isinstance(required, list) else []
        for ref in tool_refs:
            if ref not in referenced_event_ids:
                continue
            event_index = _first_runtime_call_of_ideal_tool(conversation, events, ref)
            if event_index is not None:
                _attach_alias(events, aliases_by_event, alias_owners, event_index=event_index, alias=ref)
        consent_ref = obligation.get("consent_event_id")
        action_index = alias_owners.get(obligation.get("protected_action_event_id"))
        if consent_ref not in referenced_event_ids or action_index is None:
            continue
        consent_index = next(
            (
                index
                for index in range(action_index - 1, -1, -1)
                if events[index].get("event_type") == "message"
                and CANONICAL_ACTOR_BY_RUNTIME_ACTOR.get(str(events[index].get("actor") or "")) == "user"
            ),
            None,
        )
        if consent_index is not None:
            _attach_alias(events, aliases_by_event, alias_owners, event_index=consent_index, alias=consent_ref)


def _first_runtime_call_of_ideal_tool(
    conversation: list[Any], events: list[dict[str, Any]], ref: Any
) -> int | None:
    indices = _referenced_ideal_indices({ref}, conversation_length=len(conversation)) if isinstance(ref, str) else {}
    if not indices:
        return None
    turn = conversation[next(iter(indices))]
    tool_call = turn.get("tool_call") if isinstance(turn, dict) else None
    name = tool_call.get("name") if isinstance(tool_call, dict) else None
    if not isinstance(name, str) or not name:
        return None
    calls = [
        index
        for index, event in enumerate(events)
        if event.get("event_type") == "tool_call" and event.get("tool") == name
    ]
    # A failed or not-executed call (error, persona interrupt) did not perform the
    # operation, so the first successful call bounds M05; with none, the first call.
    return next((index for index in calls if _call_succeeded(events, index)), calls[0] if calls else None)


def _call_succeeded(events: list[dict[str, Any]], call_index: int) -> bool:
    call = events[call_index]
    for event in events[call_index + 1 :]:
        if event.get("event_type") != "tool_result" or event.get("tool") != call.get("tool"):
            continue
        result = event.get("result")
        return isinstance(result, dict) and result.get("ok") is True
    return False


def _attach_declared_interaction_aliases(
    scenario: dict[str, Any],
    conversation: list[Any],
    events: list[dict[str, Any]],
    aliases_by_event: dict[int, list[str]],
    alias_owners: dict[str, int],
) -> None:
    """Attach exact value/utterance aliases declared by interaction questions."""

    contract = scenario.get("interaction_closed_questions")
    instances = contract.get("instances") if isinstance(contract, dict) else None
    if not isinstance(instances, list):
        return
    for instance in instances:
        parameters = instance.get("parameters") if isinstance(instance, dict) else None
        if not isinstance(parameters, dict):
            continue
        target_ref = parameters.get("user_turn_ref") or parameters.get(
            "trigger_turn_ref"
        )
        if not isinstance(target_ref, str) or not target_ref.startswith(
            IDEAL_EVENT_ALIAS_PREFIX
        ):
            continue
        suffix = target_ref.removeprefix(IDEAL_EVENT_ALIAS_PREFIX)
        if not suffix.isdigit() or int(suffix) >= len(conversation):
            continue
        expected = parameters.get("value")
        if not isinstance(expected, str) or not expected:
            turn = conversation[int(suffix)]
            expected = turn.get("utterance") if isinstance(turn, dict) else None
        if not isinstance(expected, str) or not expected:
            continue
        normalized_expected = _normalize_interaction_text(expected)
        candidates = [
            index
            for index, event in enumerate(events)
            if event.get("event_type") == "message"
            and CANONICAL_ACTOR_BY_RUNTIME_ACTOR.get(
                str(event.get("actor") or ""),
                str(event.get("actor") or ""),
            )
            == "user"
            and normalized_expected
            in _normalize_interaction_text(str(event.get("content") or ""))
        ]
        if len(candidates) == 1:
            _attach_alias(
                events,
                aliases_by_event,
                alias_owners,
                event_index=candidates[0],
                alias=target_ref,
            )


def _normalize_interaction_text(value: str) -> str:
    return INTERACTION_VALUE_SEPARATOR_PATTERN.sub("", value).lower()


def _referenced_ideal_indices(
    refs: set[str], *, conversation_length: int
) -> dict[int, str]:
    result: dict[int, str] = {}
    for ref in refs:
        if not ref.startswith(IDEAL_EVENT_ALIAS_PREFIX):
            continue
        suffix = ref.removeprefix(IDEAL_EVENT_ALIAS_PREFIX)
        if not suffix.isdigit():
            continue
        index = int(suffix)
        if index < conversation_length:
            result[index] = ref
    return result


def _referenced_ideal_tool_result_indices(
    refs: set[str], *, conversation_length: int
) -> dict[int, str]:
    result: dict[int, str] = {}
    for ref in refs:
        if not ref.startswith(IDEAL_EVENT_ALIAS_PREFIX) or not ref.endswith(
            IDEAL_TOOL_RESULT_ALIAS_SUFFIX
        ):
            continue
        suffix = ref.removeprefix(IDEAL_EVENT_ALIAS_PREFIX).removesuffix(
            IDEAL_TOOL_RESULT_ALIAS_SUFFIX
        )
        if not suffix.isdigit():
            continue
        index = int(suffix)
        if index < conversation_length:
            result[index] = ref
    return result


def _validated_aliases_by_event(
    events: list[dict[str, Any]],
) -> dict[int, list[str]]:
    aliases_by_event: dict[int, list[str]] = {}
    for index, event in enumerate(events):
        aliases = event.get("event_aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias for alias in aliases
        ):
            raise ValueError("call event_aliases must be a string array")
        aliases_by_event[index] = list(aliases)
    return aliases_by_event


def _ideal_tool_anchors(
    conversation: list[Any], events: list[dict[str, Any]]
) -> tuple[dict[int, int], list[int]]:
    ideal_by_name: dict[str, list[int]] = {}
    for index, turn in enumerate(conversation):
        tool_call = turn.get("tool_call") if isinstance(turn, dict) else None
        name = tool_call.get("name") if isinstance(tool_call, dict) else None
        if isinstance(name, str) and name:
            ideal_by_name.setdefault(name, []).append(index)
    runtime_by_name: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        name = event.get("tool")
        if event.get("event_type") == "tool_call" and isinstance(name, str) and name:
            runtime_by_name.setdefault(name, []).append(index)
    anchors: dict[int, int] = {}
    all_ideal_indices: list[int] = []
    for name, ideal_indices in ideal_by_name.items():
        all_ideal_indices.extend(ideal_indices)
        runtime_indices = runtime_by_name.get(name, [])
        if len(runtime_indices) < len(ideal_indices):
            continue
        for ideal_index, runtime_index in zip(
            reversed(ideal_indices), reversed(runtime_indices)
        ):
            anchors[ideal_index] = runtime_index
    return anchors, sorted(all_ideal_indices)


def _tool_result_events_by_call_event(
    events: list[dict[str, Any]],
) -> dict[int, int]:
    pending: dict[str, list[int]] = {}
    result: dict[int, int] = {}
    for index, event in enumerate(events):
        tool_name = event.get("tool")
        if not isinstance(tool_name, str) or not tool_name:
            continue
        if event.get("event_type") == "tool_call":
            pending.setdefault(tool_name, []).append(index)
            continue
        if event.get("event_type") != "tool_result":
            continue
        calls = pending.get(tool_name)
        if calls:
            result[calls.pop(0)] = index
    return result


def _attach_alias(
    events: list[dict[str, Any]],
    aliases_by_event: dict[int, list[str]],
    alias_owners: dict[str, int],
    *,
    event_index: int,
    alias: str,
) -> None:
    owner = alias_owners.get(alias)
    if owner is not None and owner != event_index:
        return
    if alias in aliases_by_event[event_index]:
        return
    aliases_by_event[event_index].append(alias)
    events[event_index]["event_aliases"] = aliases_by_event[event_index]
    alias_owners[alias] = event_index
