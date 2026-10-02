"""Build package-declared gold records and return deterministic judge answers."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.scoring.ticket_fact_builder import (
    build_ticket_fact,
)


ROOT = Path(__file__).resolve().parents[2]
JUDGE_CONFIG = {
    "schema_version": "conversation_log_judge_config",
    "judge": {
        "member_id": "perfect-score-test-judge",
        "family": "fixture",
        "provider": "fixture",
        "endpoint": "https://judge.invalid",
        "transport": "chat_completions",
        "resolved_model_revision": "perfect-judge-2026-08-25",
        "revision_policy": "fixed",
        "api_key_env": "PERFECT_SCORE_TEST_API_KEY",
        "sampling": {"temperature": 0, "seed": 1, "max_tokens": 4096},
    },
    "max_input_bytes": 10_000_000,
    "minimum_accuracy": 1.0,
}
PERSONA_FIELDS = (
    "customer_pressure",
    "customer_misconception",
    "customer_correction",
    "customer_emotion",
)
TICKET_FIELDS = (
    "handling_type",
    "target_ids",
    "performed_actions",
    "evidence_refs",
)
HANDLING_TYPES = {
    "change": "change_procedure",
    "guidance": "guidance_inquiry_answer",
    "callback": "callback_commitment",
    "escalation": "escalation",
    "refusal": "refusal",
}


def load_source_package(domain: str, scenario_id: str) -> dict[str, Any]:
    """Join object-root task and solution JSON for one source package.

    Both inputs must be JSON objects with no overlapping non-metadata root
    fields. Missing files, malformed JSON, non-object roots, and overlap raise.
    """

    paths = (
        ROOT / "data" / "tasks" / domain / f"{scenario_id}.json",
        ROOT / "data" / "solutions" / domain / f"{scenario_id}.json",
    )
    parts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(not isinstance(part, dict) for part in parts):
        raise ValueError("task and solution JSON roots must be objects")
    overlap = set(parts[0]) & set(parts[1]) - {"metadata"}
    if overlap:
        raise ValueError(f"task and solution fields overlap: {sorted(overlap)}")
    package = {**parts[0], **parts[1]}
    package.pop("metadata", None)
    return package


def _required_sops(package: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {
        sop["id"]: sop
        for category in package["sop_catalog"]["categories"].values()
        for sop in category["sops"]
    }
    return [deepcopy(by_id[sop_id]) for sop_id in package["use_sops"]]


def _tool_result(turn: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": "error" not in turn}
    if "error" in turn:
        result["error"] = turn["error"]
    if "returned_rows" in turn:
        result["rows"] = [deepcopy(row["values"]) for row in turn["returned_rows"]]
    if "world_changes" in turn:
        result["changes"] = deepcopy(turn["world_changes"])
    return result


def project_record(record: dict[str, Any], package: dict[str, Any]) -> None:
    conversation = []
    tool_calls = []
    pending: dict[str, list[dict[str, Any]]] = {}
    for event in record["event_log"]:
        if event["event_type"] == "message":
            row = {
                "actor": "customer" if event["actor"] == "user" else event["actor"],
                "content": event["content"],
                "event_id": event["event_id"],
                "event_ref": event["event_ref"],
                "seq": event["seq"],
                "epoch": "call",
            }
            if "persona_fired" in event:
                row["persona_fired"] = event["persona_fired"]
            conversation.append(row)
        elif event["event_type"] == "tool_call":
            pending.setdefault(event["tool"], []).append(event)
        elif event["event_type"] == "tool_result":
            call = pending[event["tool"]].pop(0)
            row = {
                "actor": "operator",
                "tool_id": call["tool"],
                "arguments": deepcopy(call["arguments"]),
                "result": deepcopy(event["result"]),
                "event_id": call["event_id"],
                "event_ref": call["event_ref"],
                "seq": call["seq"],
                "result_event_id": event["event_id"],
                "result_event_ref": event["event_ref"],
                "result_seq": event["seq"],
                "epoch": "call",
            }
            conversation.append(row)
            tool_calls.append(deepcopy(row))
    record["conversation"] = conversation
    record["tool_calls"] = tool_calls
    fact = build_ticket_fact(record["event_log"], run_outcome=record["run_outcome"])
    ticket = {field: deepcopy(fact[field]) for field in TICKET_FIELDS}
    ticket.update(
        {
            "refusal_reason": None,
            "promises": [],
            "answer_summary": " ".join(
                turn["text"]
                for turn in package["gold_dialogue"]["turns"]
                if turn["kind"] == "utterance" and turn["speaker"] == "operator"
            ),
        }
    )
    record["operator_ticket_artifact"] = {
        "status": "submitted",
        "reason": "schema_valid",
        "ticket": ticket,
    }


def perfect_record(package: dict[str, Any]) -> dict[str, Any]:
    """Build a runtime record from one object-root package's gold declarations.

    ``package`` must contain the v3 gold dialogue, trace, world, contracts,
    persona, and SOP catalog shapes. Missing or malformed fields raise their
    normal mapping/list errors; no undeclared evidence is invented.
    """

    run_id = f"perfect-{package['scenario_id']}"
    events: list[dict[str, Any]] = []

    def emit(actor: str, event_type: str, **fields: Any) -> None:
        seq = len(events) + 1
        ref = f"event:{run_id}:{seq}"
        events.append(
            {
                "event_id": ref,
                "event_ref": ref,
                "event_aliases": [],
                "episode_id": run_id,
                "seq": seq,
                "epoch": "call",
                "actor": actor,
                "event_type": event_type,
                **deepcopy(fields),
            }
        )

    turns = deepcopy(package["gold_dialogue"]["turns"])
    persona = next(
        (
            (name, package["persona"][name])
            for name in PERSONA_FIELDS
            if package["persona"].get(name)
        ),
        None,
    )
    if persona is not None:
        trigger_name, trigger = persona
        trigger_turn = next(
            turn
            for turn in turns[1:]
            if turn["kind"] == "utterance" and turn["speaker"] == "user"
        )
        trigger_turn["text"] = f"{trigger['utterance']} {trigger_turn['text']}"
        trigger_turn["persona_fired"] = trigger_name

    def emit_turn(turn: dict[str, Any]) -> None:
        actor = {"operator": "operator", "user": "user", "tool": "tool"}[
            turn["speaker"]
        ]
        if turn["kind"] == "utterance":
            fields = {"content": turn["text"]}
            if "persona_fired" in turn:
                fields["persona_fired"] = turn["persona_fired"]
            emit(actor, "message", **fields)
        elif turn["kind"] == "tool_call":
            emit(actor, "tool_call", tool=turn["tool_id"], arguments=turn["arguments"])
        else:
            emit(actor, "tool_result", tool=turn["tool_id"], result=_tool_result(turn))

    emit_turn(turns[0])
    sops = _required_sops(package)
    emit(
        "operator",
        "tool_call",
        tool="search_sops_in_category",
        arguments={"category_id": "gold", "query": package["scenario_id"]},
    )
    emit(
        "tool",
        "tool_result",
        tool="search_sops_in_category",
        result={
            "ok": True,
            "candidates": [
                {"sop_id": sop["id"], "title": sop["title"]} for sop in sops
            ],
        },
    )
    for sop in sops:
        emit("operator", "tool_call", tool="get_sop", arguments={"sop_id": sop["id"]})
        emit("tool", "tool_result", tool="get_sop", result={"ok": True, "sop": sop})
    for turn in turns[1:]:
        emit_turn(turn)

    record = {
        "run_id": run_id,
        "scenario_id": package["scenario_id"],
        "variant_id": "hard",
        "status": "success",
        "initial_world": deepcopy(package["world"]),
        "final_world": deepcopy(package["final_world"]),
        "event_log": events,
        "hard_fails": [],
        "run_outcome": {
            "schema_version": "run_outcome",
            "handling_type": HANDLING_TYPES[
                package["contracts"]["completion"]["handling_type"]
            ],
            "completed": True,
            "reason": "package_gold_trace_completed",
        },
    }
    project_record(record, package)
    return record


def zero_record(package: dict[str, Any]) -> dict[str, Any]:
    """Build an empty, unexecuted record from one object-root v3 package.

    ``package`` must contain object ``world`` and string ``scenario_id`` fields.
    Missing or malformed values raise their normal mapping/key errors.
    """

    scenario_id = package["scenario_id"]
    return {
        "run_id": f"zero-{scenario_id}",
        "scenario_id": scenario_id,
        "variant_id": "baseline",
        "status": "incomplete",
        "initial_world": deepcopy(package["world"]),
        "final_world": deepcopy(package["world"]),
        "event_log": [],
        "conversation": [],
        "tool_calls": [],
        "hard_fails": [],
        "operator_ticket_artifact": None,
        "run_outcome": {
            "schema_version": "run_outcome",
            "handling_type": "incomplete",
            "completed": False,
            "reason": "no_conversation_or_tool_execution",
        },
    }


def perfect_judge_transport(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    """Return schema-shaped yes answers with rule-valid record evidence."""

    del url, headers, timeout
    data = json.loads(payload["messages"][-1]["content"].removeprefix("DATA_JSON="))
    schema_name = payload["response_format"]["json_schema"]["name"]
    if schema_name == "m05_consent_equivalence":
        judgements = [
            {"question_id": pair["question_id"], "answer": "yes", "reason": "gold consent"}
            for pair in data["pairs"]
        ]
    else:
        sources = data["conversation"] + data["tool_calls"]
        judgements = []
        for task in data["tasks"]:
            rules = task["evidence_rules"]
            candidates = sources
            if any("operator_utterance" in rule or rule == "operator_consent_before_deadline" for rule in rules):
                candidates = [
                    source
                    for source in sources
                    if source["event"].get("actor") == "operator"
                    and isinstance(source["event"].get("content"), str)
                ]
            if "operator_utterance_before_deadline" in rules or "operator_consent_before_deadline" in rules:
                candidates = [
                    source
                    for source in candidates
                    if source["event"]["seq"]
                    < task["relevant_context"]["deadline_seq"]
                ]
            if "operator_utterance_after_trigger" in rules:
                candidates = [
                    source
                    for source in candidates
                    if source["event"]["seq"]
                    > task["relevant_context"]["trigger_seq"]
                ]
            answer = "no" if ":decoy:" in task["question_id"] else "yes"
            judgements.append(
                {
                    "question_id": task["question_id"],
                    "answer": answer,
                    "evidence": [
                        {"ref": candidates[0]["ref"], "description": "gold record evidence"}
                    ] if answer == "yes" else [],
                }
            )
    return {
        "model": payload["model"],
        "choices": [{"message": {"content": json.dumps({"judgements": judgements})}}],
    }
