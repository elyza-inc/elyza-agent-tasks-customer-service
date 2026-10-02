"""Deterministic M20, M22, M25, and M26 scoring.

``score_value_metrics`` accepts decoded object mappings. ``record`` must
contain an object-array ``event_log``. ``scenario`` is the assembled package
and may contain ``contracts.m20`` and ``contracts.m05``. Malformed shapes
return N/M rows; no LLM is called.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from collections.abc import Mapping, Sequence
from typing import Any


RESULT_SCHEMA_VERSION = "audio_value_metrics"
METRIC_IDS = ("M20", "M22", "M25", "M26")
DIGIT_RE = re.compile(r"\d+")
KANA_RE = re.compile(r"[ァ-ヶー]{2,}")
TIME_WORDS = str.maketrans("〇一二三四五六七八九", "0123456789")


def score_value_metrics(
    *,
    record: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Score decoded persisted artifacts; invalid or missing evidence is N/M."""

    try:
        events = _events(record)
        slots = _slots(scenario)
        utterances = _utterances(events)
    except (TypeError, ValueError) as exc:
        return _result(record, {metric_id: _nm(metric_id, str(exc)) for metric_id in METRIC_IDS})

    spoken = [slot for slot in slots if _customer_spoken(slot, utterances)]
    m20 = _score_m20(spoken, events, utterances)
    m20["contract_source"] = _m20_contract_source(scenario)
    return _result(
        record,
        {
            "M20": m20,
            "M22": _score_m22(scenario, utterances, events),
            "M25": _score_m25(spoken, utterances),
            "M26": _score_m26(spoken, events),
        },
    )


def _m20_contract_source(scenario: Mapping[str, Any]) -> str:
    contracts = scenario.get("contracts")
    m20 = contracts.get("m20") if isinstance(contracts, Mapping) else None
    if isinstance(m20, Mapping) and isinstance(m20.get("important_values"), list):
        return "scenario.contracts.m20.important_values"
    return "fallback"


def _events(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = record.get("event_log")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("record.event_log must be an object array")
    return sorted(rows, key=lambda row: row.get("seq", -1))


def _slots(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    contracts = scenario.get("contracts")
    m20 = contracts.get("m20") if isinstance(contracts, Mapping) else None
    raw = m20.get("important_values") if isinstance(m20, Mapping) else None
    return _with_gold_argument_names([_slot(row) for row in raw], scenario) if isinstance(raw, list) else []


def _with_gold_argument_names(slots: list[dict[str, Any]], scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Name each slot's Tool arguments from gold: the arguments that carry its value."""

    calls = scenario.get("gold_tool_calls")
    if not isinstance(calls, list):
        return slots
    for slot in slots:
        slot["argument_names"] = {
            name
            for call in calls
            for name, value in ((call.get("arguments") or {}).items() if isinstance(call, Mapping) else ())
            if _normalize(value) == slot["normalized"]
        }
    return slots


def _slot(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("important value slots must be objects")
    name = row.get("slot_id", row.get("name"))
    if not isinstance(name, str) or not name or "value" not in row:
        raise ValueError("important value slot requires name and value")
    value = row["value"]
    return {
        "slot_id": name,
        "value": value,
        "normalized": _normalize(value),
        "kind": _kind(row.get("value_type"), value),
    }


def _utterances(events: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for event in events:
        if event.get("event_type") != "message" or event.get("actor") not in {"user", "customer", "operator"}:
            continue
        metadata = event.get("metadata")
        asr = metadata.get("diagnostic_user_audio_asr", {}).get("asr") if isinstance(metadata, Mapping) else None
        profiles = asr.get("profiles") if isinstance(asr, Mapping) else None
        transcripts = [
            row.get("transcript")
            for row in profiles or []
            if isinstance(row, Mapping)
            and isinstance(row.get("transcript"), str)
            and row["transcript"].strip()
        ]
        result.append(
            {
                "seq": event.get("seq"),
                "actor": event.get("actor"),
                "source": str(event.get("content") or ""),
                "profiles": transcripts,
            }
        )
    return result


def _customer_spoken(slot: Mapping[str, Any], utterances: list[dict[str, Any]]) -> bool:
    return any(
        row["actor"] in {"user", "customer"}
        and _contains(slot, _normalize(row["source"]))
        for row in utterances
    )


def _score_m20(
    slots: list[dict[str, Any]],
    events: list[Mapping[str, Any]],
    utterances: list[dict[str, Any]],
) -> dict[str, Any]:
    units = []
    customer_texts = [_normalize(row["source"]) for row in utterances if row["actor"] in {"user", "customer"}]
    for slot in slots:
        if slot["kind"] == "text" and not slot["normalized"].isascii():
            # Free Japanese text (agreement wording etc.) has no exact spoken form to check.
            continue
        evidence = _tool_evidence(slot, events, customer_texts) + _operator_evidence(slot, utterances)
        if not evidence and "argument_names" in slot and not slot["argument_names"]:
            # Gold never passes this value to a Tool and no readback was heard: nothing to judge.
            continue
        passed = bool(evidence) and all(item["correct"] for item in evidence)
        units.append(_unit(slot["slot_id"], passed, evidence_count=len(evidence)))
    return _metric("M20", units, empty_status="N/A")


def _score_m25(slots: list[dict[str, Any]], utterances: list[dict[str, Any]]) -> dict[str, Any]:
    units = []
    for slot in slots:
        user_rows = [
            row for row in utterances
            if row["actor"] in {"user", "customer"} and _contains(slot, _normalize(row["source"]))
        ]
        for user in user_rows:
            next_user = min(
                (row["seq"] for row in utterances if row["actor"] in {"user", "customer"} and row["seq"] > user["seq"]),
                default=float("inf"),
            )
            candidates = [
                evidence
                for row in utterances
                if row["actor"] == "operator" and user["seq"] < row["seq"] < next_user
                for evidence in _typed_profile_evidence(slot, row["profiles"], row.get("source"))
            ]
            if candidates:
                units.append(_unit(f"{slot['slot_id']}@{user['seq']}", all(item["correct"] for item in candidates)))
    result = _metric("M25", units, empty_status="N/A")
    result["accuracy"] = (
        sum(unit["passed"] for unit in units) / len(units) if units else None
    )
    result["observation"] = {
        "opportunity_count": len(units),
        "target_value_count": len(slots),
        "rate": len(units) / len(slots) if slots else None,
    }
    result["observation_rate"] = result["observation"]["rate"]
    return result


def _score_m26(slots: list[dict[str, Any]], events: list[Mapping[str, Any]]) -> dict[str, Any]:
    calls = [event for event in events if event.get("event_type") == "tool_call"]
    if any(not isinstance(call.get("argument_provenance"), Mapping) for call in calls):
        return _nm("M26", "argument_provenance_missing")
    units = []
    for call in calls:
        arguments = call.get("arguments")
        provenance = call["argument_provenance"]
        if not isinstance(arguments, Mapping):
            continue
        for name, value in arguments.items():
            slot = next((item for item in slots if _same_typed_value(item, value)), None)
            if slot is None:
                continue
            source = provenance.get(name)
            if not isinstance(source, Mapping):
                return _nm("M26", "argument_provenance_missing")
            units.append(_unit(f"{call.get('event_id', call.get('seq'))}:{name}", source.get("source_kind") != "unknown"))
    result = _metric("M26", units, empty_status="N/A")
    result["grounded_use_rate"] = (
        sum(unit["passed"] for unit in units) / len(units) if units else None
    )
    return result


def _score_m22(
    scenario: Mapping[str, Any],
    utterances: list[dict[str, Any]],
    events: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    contracts = scenario.get("contracts")
    contract = contracts.get("m05") if isinstance(contracts, Mapping) else None
    if contract is None:
        return _metric("M22", [], empty_status="N/A")
    if not isinstance(contract, Mapping) or not isinstance(contract.get("disclosure"), str):
        return _nm("M22", "contracts.m05.disclosure_invalid")
    deadline_tool = contract.get("deadline_tool_id")
    if events is not None and isinstance(deadline_tool, str) and not any(
        event.get("event_type") == "tool_call" and event.get("tool") == deadline_tool for event in events
    ):
        # Same as M05: the disclosure is owed only once the call reaches its deadline Tool.
        return _metric("M22", [], empty_status="N/A")
    operator_profiles = [row["profiles"] for row in utterances if row["actor"] == "operator"]
    keywords = _meaning_core_keywords(contract["disclosure"])
    if not keywords:
        # With no number or action word there is nothing to hear; failing it would
        # score the disclosure wording, not the operator.
        return _nm("M22", "disclosure_has_no_meaning_core_keywords")
    passed = all(
        any(_agreed_contains(keyword, profiles) for profiles in operator_profiles)
        for keyword in keywords
    )
    unit_id = str(contract.get("deadline_tool_id") or "m05-disclosure")
    return _metric("M22", [_unit(unit_id, passed, keywords=keywords)], empty_status="N/A")


def _meaning_core_keywords(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).translate(TIME_WORDS)
    values = [
        _normalize(token)
        for token in re.findall(r"\d{1,2}:\d{2}|\d[\d,]*(?:時|円|%|泊|日|分|名|回)?", normalized)
    ]
    for keyword in ("保持", "返金", "取消", "取り消", "キャンセル", "変更", "登録", "移管", "連絡"):
        if keyword in normalized:
            values.append(keyword)
    return list(dict.fromkeys(values))


def _tool_evidence(
    slot: Mapping[str, Any],
    events: list[Mapping[str, Any]],
    customer_texts: Sequence[str] = (),
) -> list[dict[str, Any]]:
    evidence = []
    for event in events:
        if event.get("event_type") != "tool_call" or not isinstance(event.get("arguments"), Mapping):
            continue
        provenance = event.get("argument_provenance")
        for name, value in event["arguments"].items():
            source = provenance.get(name) if isinstance(provenance, Mapping) else None
            linked = isinstance(source, Mapping) and source.get("source_kind") in {"customer_utterance", "unknown"}
            if not _argument_names_slot(name, slot) or not _same_shape(slot, value):
                continue
            correct = _same_typed_value(slot, value)
            # A correct value counts whatever provenance guessed (a "3" also appears in results);
            # a wrong one is the operator's only when it was not copied from a Tool result.
            if correct or linked:
                if not correct and any(_normalize(value) in text for text in customer_texts):
                    # The customer said this value (e.g. a name corrected later); using it is faithful.
                    continue
                evidence.append({"kind": "tool_argument", "correct": correct})
    return evidence


def _operator_evidence(slot: Mapping[str, Any], utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for row in utterances
        if row["actor"] == "operator"
        for item in _typed_profile_evidence(slot, row["profiles"], row.get("source"))
    ]


def _argument_names_slot(name: str, slot: Mapping[str, Any]) -> bool:
    """An argument is evidence for a slot only when its name refers to the same kind of value."""

    slot_id = str(slot.get("slot_id") or "")
    if name == slot_id:
        return True
    if "argument_names" in slot:
        # Gold shows which arguments carry this value; name substrings are only a fallback.
        return name in slot["argument_names"]
    if slot["kind"] == "kana":
        return "kana" in name
    if "phone" in slot_id:
        return "phone" in name
    return False


def _typed_profile_evidence(
    slot: Mapping[str, Any], profiles: list[str], operator_text: str | None = None
) -> list[dict[str, Any]]:
    if len(profiles) < 2:
        return []
    token_sets = [_tokens(slot["kind"], text) for text in profiles]
    agreed = set.intersection(*token_sets)
    if slot["kind"] == "digits":
        # Only numbers that read like the value are attempts at it; another number of the
        # same length (a tracking number, a different size) is not evidence either way.
        agreed = {
            token for token in agreed
            if len(token) == len(slot["normalized"])
            and SequenceMatcher(None, token, slot["normalized"]).ratio() >= 0.7
        }
        if operator_text is not None and slot["normalized"] in _normalize(operator_text):
            # Both ASR profiles are whisper-1: when the operator's own text says the right
            # number, a different number in the transcripts may be a shared mishearing.
            agreed = {token for token in agreed if token == slot["normalized"]}
    if slot["kind"] == "kana":
        # Only katakana words that read like the value are attempts at it; other
        # katakana ("カタカナ", "メッセージ") is not evidence either way.
        agreed = {
            token for token in agreed
            if SequenceMatcher(None, token, slot["normalized"]).ratio() >= 0.5
            # ASR writes a surname in kanji, leaving only part of the name in katakana.
            and (token == slot["normalized"] or (len(token) == len(slot["normalized"]) and token not in slot["normalized"]))
        }
    return [{"kind": "audio_asr", "correct": token == slot["normalized"]} for token in agreed]


def _tokens(kind: str, text: str) -> set[str]:
    normalized = _normalize(text)
    if kind == "digits":
        return {token for token in DIGIT_RE.findall(normalized) if len(token) >= 2}
    if kind == "kana":
        # Join katakana words split by spaces or 中黒 (「イシカワ アヤノ」) before extracting.
        joined = re.sub(r"(?<=[ァ-ヶー])[\s・･]+(?=[ァ-ヶー])", "", unicodedata.normalize("NFKC", text))
        return set(KANA_RE.findall(joined))
    return set()


def _agreed_contains(keyword: str, profiles: list[str]) -> bool:
    needle = _normalize(keyword)
    return len(profiles) >= 2 and all(needle in _normalize(text) for text in profiles)


def _contains(slot: Mapping[str, Any], text: str) -> bool:
    return slot["normalized"] in text


def _same_shape(slot: Mapping[str, Any], value: Any) -> bool:
    normalized = _normalize(value)
    if slot["kind"] == "digits":
        return normalized.isdigit() and len(normalized) == len(slot["normalized"])
    if slot["kind"] == "kana":
        return bool(normalized) and all("ァ" <= char <= "ヶ" or char == "ー" for char in normalized)
    return normalized == slot["normalized"]


def _same_typed_value(slot: Mapping[str, Any], value: Any) -> bool:
    return _normalize(value) == slot["normalized"]


def _kind(declared: Any, value: Any) -> str:
    normalized = _normalize(value)
    if declared in {"phone", "phone_number", "time", "date", "number", "integer"} or normalized.isdigit():
        return "digits"
    if declared in {"name", "kana_name", "name_kana"} or (normalized and all("ァ" <= char <= "ヶ" or char == "ー" for char in normalized)):
        return "kana"
    return "text"


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).translate(TIME_WORDS).upper()
    return "".join(char for char in text if char.isalnum() or "ァ" <= char <= "ヶ" or char == "ー")


def _unit(unit_id: str, passed: bool, **diagnostics: Any) -> dict[str, Any]:
    return {"unit_id": unit_id, "status": "passed" if passed else "failed", "passed": passed, "diagnostics": diagnostics}


def _metric(metric_id: str, units: list[dict[str, Any]], *, empty_status: str) -> dict[str, Any]:
    if not units:
        return {"metric_id": metric_id, "status": empty_status, "passed": None, "units": []}
    passed = all(unit["passed"] for unit in units)
    return {"metric_id": metric_id, "status": "passed" if passed else "failed", "passed": passed, "units": units}


def _nm(metric_id: str, reason: str) -> dict[str, Any]:
    return {"metric_id": metric_id, "status": "N/M", "passed": None, "reason": reason, "units": []}


def _result(record: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": RESULT_SCHEMA_VERSION, "run_id": record.get("run_id"), "metrics": dict(metrics)}
