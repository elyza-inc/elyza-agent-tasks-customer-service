"""Apply deterministic, business-safe variations to a perfect record."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.scoring.conversation_log_metrics import (
    _depends_on_preceding_step,
    _scalar_values,
)


TRANSFORM_KINDS = ("commute_swap", "paraphrase", "insert_confirmation")
SOP_TOOLS = {"search_sop_categories", "search_sops_in_category", "get_sop"}
PARAPHRASES = (
    ("お伺いしてもよろしいでしょうか", "お聞きしても問題ないでしょうか"),
    ("よろしいでしょうか", "問題ないでしょうか"),
    ("教えていただけますか", "お聞かせいただけますか"),
    ("でございます", "です"),
    ("いたしました", "しました"),
    ("いたします", "します"),
)


def transform(record: dict[str, Any], kind: str) -> dict[str, Any]:
    """Return a transformed copy of one object-root perfect record.

    ``kind`` must be ``commute_swap``, ``paraphrase``, or
    ``insert_confirmation``. The record must contain an object-array
    ``event_log``. Unsupported kinds, malformed logs, and variations with no
    valid deterministic candidate raise ``ValueError``.
    """

    if kind not in TRANSFORM_KINDS:
        raise ValueError(f"unsupported transform kind: {kind}")
    candidate = deepcopy(record)
    events = candidate.get("event_log")
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError("record.event_log must be an object array")
    if kind == "commute_swap":
        candidate["event_log"] = _commute_swap(events)
    elif kind == "paraphrase":
        _paraphrase(events)
    else:
        _insert_confirmation(candidate, events)
    _resequence(candidate["event_log"])
    return candidate


def _commute_swap(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = _steps(events)
    for index, (former, later) in enumerate(zip(steps, steps[1:])):
        if _successful_read(former) and _successful_read(later):
            if not _depends_on_preceding_step(steps, index):
                return _swap(steps, index)
    for index, (former, later) in enumerate(zip(steps, steps[1:])):
        if former[-1].get("event_type") != "message" or not _successful_read(later):
            continue
        if not _depends_on_preceding_step(steps, index):
            return _swap(steps, index)
    raise ValueError("no data-independent adjacent steps")


def _steps(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    steps = []
    index = 0
    while index < len(events):
        event = events[index]
        paired = (
            event.get("event_type") == "tool_call"
            and index + 1 < len(events)
            and events[index + 1].get("event_type") == "tool_result"
            and event.get("tool") == events[index + 1].get("tool")
        )
        if paired:
            steps.append(events[index : index + 2])
            index += 2
        else:
            steps.append([event])
            index += 1
    return steps


def _successful_read(step: list[dict[str, Any]]) -> bool:
    if len(step) != 2 or step[0].get("tool") in SOP_TOOLS:
        return False
    result = step[1].get("result")
    return isinstance(result, dict) and result.get("ok") is True and not result.get("changes")


def _swap(steps: list[list[dict[str, Any]]], index: int) -> list[dict[str, Any]]:
    steps[index], steps[index + 1] = steps[index + 1], steps[index]
    return [event for step in steps for event in step]


def _paraphrase(events: list[dict[str, Any]]) -> None:
    changed = False
    for event in events:
        if event.get("event_type") != "message" or event.get("actor") != "operator":
            continue
        content = event.get("content")
        if not isinstance(content, str):
            continue
        rewritten = content
        for original, replacement in PARAPHRASES:
            rewritten = rewritten.replace(original, replacement)
        event["content"] = rewritten
        changed |= rewritten != content
    if not changed:
        raise ValueError("no paraphrase template matched")


def _insert_confirmation(record: dict[str, Any], events: list[dict[str, Any]]) -> None:
    user_utterances = [
        str(event.get("content", ""))
        for event in events
        if event.get("event_type") == "message" and event.get("actor") == "user"
    ]
    user_text = " ".join(user_utterances)
    values = [
        scalar
        for event in events
        if event.get("event_type") == "tool_call" and event.get("tool") not in SOP_TOOLS
        for scalar in _scalar_values(event.get("arguments", {}))
        if str(scalar) in user_text
        and str(scalar) not in user_utterances
    ]
    if not values:
        raise ValueError("no spoken important value found")
    value = max(values, key=lambda item: (len(str(item)), str(item)))
    use_index = next(
        index
        for index, event in enumerate(events)
        if event.get("event_type") == "tool_call"
        and value in _scalar_values(event.get("arguments", {}))
    )
    insertion_index = 0
    for index, event in enumerate(events[:use_index]):
        if event.get("event_type") == "message" and event.get("actor") == "user":
            insertion_index = index + 1
    template = events[insertion_index - 1]
    events.insert(
        insertion_index,
        {
            "event_id": f"event:{record['run_id']}:confirmation",
            "event_ref": f"event:{record['run_id']}:confirmation",
            "event_aliases": [],
            "episode_id": template["episode_id"],
            "seq": 0,
            "epoch": "call",
            "actor": "operator",
            "event_type": "message",
            "content": f"重要な確認値を「{value}」として承りました。この内容で問題ないでしょうか。",
        },
    )


def _resequence(events: list[dict[str, Any]]) -> None:
    for seq, event in enumerate(events, 1):
        event["seq"] = seq
