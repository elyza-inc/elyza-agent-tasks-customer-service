"""Judge M05 (semantic disclosure/consent), M15, M16, and M19 from persisted conversation evidence.

``load_mapping`` accepts one object-root JSON, YAML, or YML file. Scoring
accepts a standalone package mapping, its runtime ``record.json`` mapping,
a validated fixed-model config, and a writable cache directory.
Conversation rows and tool-call rows must be complete object arrays; missing,
malformed, oversized, truncated, or schema-invalid inputs raise ``ValueError``.
Judge rows that violate evidence rules are retained as ``invalid`` and scored
incorrect. No default answer or summary is accepted.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Callable

import yaml
from jsonschema import Draft202012Validator


RUNTIME_ROOT = Path(__file__).resolve().parent

from elyza_agent_tasks_customer_service.evaluation.contracts.artifact_receipts import (
    canonical_json_bytes,
    find_content_addressed_receipt_by_request,
    replay_content_addressed_bundle,
    sha256_json,
    write_content_addressed_bundle,
)
from elyza_agent_tasks_customer_service.evaluation.llm.llm_io import extract_chat_message, post_json
from elyza_agent_tasks_customer_service.evaluation.contracts.semantic_judge_contract import (
    judge_sampling_parameters,
    validate_judge_member,
)
from elyza_agent_tasks_customer_service.evaluation.contracts.run_outcome import _event_succeeded as _tool_call_succeeded
from elyza_agent_tasks_customer_service.evaluation.scoring.ticket_conversation_questions import build_ticket_conversation_questions
from elyza_agent_tasks_customer_service.evaluation.engine.package_adapter import m05_judge_deadline_tool
from elyza_agent_tasks_customer_service.evaluation.scoring.conversation_log_metrics import PACKAGE_SOP_TOOLS


SCORER_VERSION = "conversation_log_scoring"
CONFIG_VERSION = "conversation_log_judge_config"
INPUT_VERSION = "conversation_log_judge_input_v1"  # ジャッジのリクエスト本文に入る。変えるとキャッシュが使えなくなる
JUDGE_NAMESPACE = "conversation_log_judge"
JUDGE_CALL_KIND = "conversation_log_judgement"
PROMPT_PATH = RUNTIME_ROOT.parent / "prompt_templates" / "conversation_log_judge.md"
REQUEST_TIMEOUT_SECONDS = 900.0
TARGET_METRICS = ("M05", "M15", "M16", "M19")
M16_CRITERIA = ("M16-M1", "M16-M2", "M16-M4", "M16-M5")
M16_QUESTIONS = {
    "M16-M1": "会話に、調べる間お待ちいただくよう顧客へ伝えたオペレーター発話がありますか",
    "M16-M2": "会話に、内容をまとめて顧客へ確認したオペレーター発話がありますか",
    "M16-M4": "会話に、重要値を顧客へ復唱したオペレーター発話がありますか",
    "M16-M5": "会話に、手続きの実行前に顧客へ同意を確認し、顧客から同意の返事を得たオペレーター発話がありますか",
}
M19_QUESTIONS = {
    "CQ1": "困難な要求や発言を受けたことを踏まえて応答したか。",
    "CQ2": "手順を曲げず、断りまたは規定を説明したか。",
    "CQ3": "顧客の事情・感情・訂正・誤解に応じた説明または対応を行ったか。",
}
M05_DISCLOSURE_QUESTION = (
    "契約の開示内容「{disclosure}」を、対象の変更操作より前に顧客へ伝えた"
    "オペレーター発話が通話中にありますか。言い回しの違いは問わず、"
    "内容が同じであれば伝えたとみなします。"
)
M05_CONSENT_QUESTION = (
    "オペレーターが「{consent}」について顧客に同意を求め、顧客が同意する返事をした発話が、"
    "対象の変更操作より前にありますか。言い回しの違いは問わず、同意の対象が同じであれば"
    "同意を得たとみなします。顧客が自分から話しただけの発話は同意の返事とみなしません。"
)
EDGE_FIELDS = (
    "customer_pressure",
    "customer_misconception",
    "customer_correction",
    "customer_emotion",
)


Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def load_mapping(path: Path) -> dict[str, Any]:
    """Load one object-root ``.json``, ``.yaml``, or ``.yml`` file.

    Other suffixes, malformed documents, arrays, scalars, and empty files
    raise ``ValueError`` instead of being coerced.
    """

    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"mapping file must be JSON or YAML: {path}")
    if not isinstance(value, dict):
        raise ValueError(f"mapping file root must be an object: {path}")
    return value


def load_judge_config(path: Path) -> dict[str, Any]:
    """Load an exact fixed-model judge config from JSON or YAML.

    The root must contain only ``schema_version``, ``judge``,
    ``max_input_bytes``, and ``minimum_accuracy``. Temperature may be numeric
    or null (explicitly omitted), seed must be an integer, and the model
    revision must be fixed. Invalid or incomplete configs raise ``ValueError``.
    """

    return _validate_config_value(load_mapping(path))


def score_conversation(
    *,
    package: dict[str, Any],
    record: dict[str, Any],
    config: dict[str, Any],
    cache_dir: Path,
    transport: Transport = post_json,
) -> dict[str, Any]:
    """Score M05/M15/M16/M19 from one complete runtime record.

    ``package`` is the standalone package. ``record`` must contain complete,
    possibly empty ``conversation``, ordered ``tool_calls``, and ``event_log``
    arrays plus an optional operator ticket. Judge evidence-rule violations are
    retained as invalid judgements; cache, schema, transport, and evidence-
    reference failures still raise. No failure is converted to ``N/M`` or a
    default answer.
    """

    config = _validate_config_value(config)
    evidence = _evidence(record)
    tasks, applicability = _tasks(package, record, evidence)
    if not tasks:
        return {
            "schema_version": SCORER_VERSION,
            "scenario_id": package.get("scenario_id"),
            "input_hash": sha256_json({"package": _judge_contract(package), "record": record}),
            "metric_results": _score_tasks(tasks, {"by_id": {}}, applicability),
            "judge": {
                "model": config["judge"]["resolved_model_revision"],
                "prompt_hash": sha256_json(PROMPT_PATH.read_text(encoding="utf-8")),
                "sampling": deepcopy(config["judge"]["sampling"]),
                "request_hash": None,
                "cache_reused": False,
                "receipt": None,
                "reason": "all_metrics_explicitly_not_applicable",
                "api_call_count": 0,
                "classification": "valid",
                "invalid_judgement_count": 0,
            },
        }
    requests = {}
    decisions = {"by_id": {}}
    for metric_id in TARGET_METRICS:
        metric_tasks = [task for task in tasks if task["metric_id"] == metric_id]
        if not metric_tasks:
            requests[metric_id] = _empty_judge_request()
            continue
        result = _run_judge(
            tasks=metric_tasks,
            evidence=evidence,
            config=config,
            cache_dir=cache_dir,
            transport=transport,
        )
        decisions["by_id"].update(result["by_id"])
        requests[metric_id] = {
            key: deepcopy(result[key])
            for key in (
                "request_hash",
                "cache_reused",
                "receipt",
                "classification",
                "invalid_judgement_count",
            )
        }
        requests[metric_id]["api_call_count"] = 0 if result["cache_reused"] else 1
    rows = _score_tasks(tasks, decisions, applicability)
    invalid_count = sum(row["invalid_judgement_count"] for row in requests.values())
    return {
        "schema_version": SCORER_VERSION,
        "scenario_id": package.get("scenario_id"),
        "input_hash": sha256_json({"package": _judge_contract(package), "record": record}),
        "metric_results": rows,
        "judge": {
            "model": config["judge"]["resolved_model_revision"],
            "prompt_hash": sha256_json(PROMPT_PATH.read_text(encoding="utf-8")),
            "sampling": deepcopy(config["judge"]["sampling"]),
            "requests": requests,
            "api_call_count": sum(row["api_call_count"] for row in requests.values()),
            "classification": "invalid" if invalid_count else "valid",
            "invalid_judgement_count": invalid_count,
        },
    }


def _empty_judge_request() -> dict[str, Any]:
    return {
        "request_hash": None,
        "cache_reused": False,
        "receipt": None,
        "api_call_count": 0,
        "classification": "valid",
        "invalid_judgement_count": 0,
        "reason": "metric_explicitly_not_applicable",
    }


def judge_response_schema(question_ids: list[str], evidence_refs: list[str]) -> dict[str, Any]:
    """Return the exact response schema for non-empty question and ref lists."""

    if not question_ids or len(set(question_ids)) != len(question_ids):
        raise ValueError("judge question IDs must be non-empty and unique")
    if not evidence_refs or len(set(evidence_refs)) != len(evidence_refs):
        raise ValueError("judge evidence refs must be non-empty and unique")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["judgements"],
        "properties": {
            "judgements": {
                "type": "array",
                "minItems": len(question_ids),
                "maxItems": len(question_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["question_id", "answer", "evidence"],
                    "properties": {
                        "question_id": {"enum": question_ids},
                        "answer": {"enum": ["yes", "no"]},
                        "evidence": {
                            "type": "array",
                            "minItems": 0,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["ref", "description"],
                                "properties": {
                                    "ref": {"enum": evidence_refs},
                                    "description": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                    },
                },
            }
        },
    }


def _tasks(
    package: dict[str, Any],
    record: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(package, dict) or not isinstance(package.get("scenario_id"), str):
        raise ValueError("package.scenario_id must be a string")
    contracts = package.get("contracts")
    if not isinstance(contracts, dict):
        raise ValueError("package.contracts must be an object")
    tasks: list[dict[str, Any]] = []
    applicability: dict[str, Any] = {metric: True for metric in TARGET_METRICS}

    m05 = contracts.get("m05")
    if m05 is None:
        applicability["M05"] = "N/A"
    else:
        if not isinstance(m05, dict):
            raise ValueError("package.contracts.m05 must be an object or null")
        disclosure = m05.get("disclosure")
        deadline_tool_id = m05.get("deadline_tool_id")
        if not isinstance(disclosure, str) or not disclosure.strip():
            raise ValueError("package.contracts.m05.disclosure must be a non-empty string")
        if not isinstance(deadline_tool_id, str) or not deadline_tool_id:
            raise ValueError("package.contracts.m05.deadline_tool_id must be a non-empty string")
        deadlines = [
            row
            for row in evidence["tool_calls"]
            if row["event"].get("tool_id") == deadline_tool_id
        ]
        deadline = next(
            (row for row in deadlines if _tool_call_succeeded(row["event"])),
            None,
        )
        if deadline is None and deadlines:
            deadline = deadlines[0]
        if deadline is None:
            applicability["M05"] = "N/A"
        else:
            # The declared deadline Tool triggers the obligation; the bound is the
            # same one package_adapter uses (next gold Tool, or the whole call).
            deadline_seq = _event_seq(deadline)
            bound_tool = m05_judge_deadline_tool(package, package["scenario_id"])
            if bound_tool != deadline_tool_id:
                later = [
                    row
                    for row in evidence["tool_calls"]
                    if bound_tool is not None
                    and row["event"].get("tool_id") == bound_tool
                    and _event_seq(row) > deadline_seq
                ]
                if later:
                    deadline, deadline_seq = later[0], _event_seq(later[0])
                else:
                    deadline_seq = 1 + max(
                        _event_seq(row) for row in evidence["conversation"] + evidence["tool_calls"]
                    )
            tasks.append(
                _task(
                    f"{package['scenario_id']}:M05:DISCLOSURE",
                    "M05",
                    "DISCLOSURE",
                    M05_DISCLOSURE_QUESTION.format(disclosure=disclosure),
                    "yes",
                    {
                        "disclosure": disclosure,
                        "deadline_tool_id": deadline_tool_id,
                        "deadline_ref": deadline["ref"],
                        "deadline_seq": deadline_seq,
                    },
                    ["operator_utterance_before_deadline"],
                )
            )
            # Position alone accepted any customer message before the protected
            # action as consent; ask whether the declared consent was obtained.
            required_consent = (contracts.get("completion") or {}).get("required_consent")
            protected_tool = m05.get("protected_action_tool_id")
            actions = [
                row
                for row in evidence["tool_calls"]
                if protected_tool is not None and row["event"].get("tool_id") == protected_tool
            ]
            action = next(
                (row for row in actions if _tool_call_succeeded(row["event"])),
                actions[0] if actions else None,
            )
            if isinstance(required_consent, str) and required_consent.strip() and action is not None:
                tasks.append(
                    _task(
                        f"{package['scenario_id']}:M05:CONSENT",
                        "M05",
                        "CONSENT",
                        M05_CONSENT_QUESTION.format(consent=required_consent),
                        "yes",
                        {
                            "required_consent": required_consent,
                            "protected_action_tool_id": protected_tool,
                            "deadline_ref": action["ref"],
                            "deadline_seq": _event_seq(action),
                        },
                        # The customer's reply is part of consent, so it may be cited too.
                        ["operator_consent_before_deadline"],
                    )
                )

    artifact = record.get("operator_ticket_artifact")
    ticket = artifact.get("ticket") if isinstance(artifact, dict) else None
    if not isinstance(ticket, dict):
        applicability["M15"] = "N/M"
    else:
        contract = package.get("post_call_ticket_contract")
        catalog = contract.get("enum_catalog") if isinstance(contract, dict) else None
        descriptions = catalog.get("code_descriptions") if isinstance(catalog, dict) else None
        operator_refs = [
            row["ref"]
            for row in evidence["conversation"]
            if _source_kind(row) == "operator_utterance"
        ]
        domain = package.get("domain")
        domain_id = domain.get("domain_id") if isinstance(domain, dict) else None
        completion = contracts.get("completion")
        if m05 is not None and applicability["M05"] == "N/A" and isinstance(completion, dict):
            # The call never reached the M05 deadline Tool, so the disclosure and
            # consent were never owed; asking the ticket about them would fail it for nothing.
            completion = {**completion, "required_disclosures": [], "required_consent": None}
        compiled = build_ticket_conversation_questions(
            ticket,
            candidate_span_refs=operator_refs,
            code_descriptions=descriptions,
            scenario_id=package["scenario_id"],
            domain_id=domain_id,
            completion=completion,
        )
        if not compiled["questions"]:
            applicability["M15"] = "N/A"
        for question in compiled["questions"]:
            tasks.append(
                _task(
                    f"{package['scenario_id']}:M15:{question['question_id']}",
                    "M15",
                    question["template_id"],
                    question["question_text"],
                    question["expected_answer"],
                    {"ticket_source_refs": deepcopy(question["source_refs"])},
                    # A consent needs the customer's reply as well, like M16-M5.
                    ["operator_utterance_required"]
                    if question["template_id"] == "required_consent"
                    else ["operator_utterance"],
                )
            )

    important_values = _important_values(contracts)
    # The judge read "the first investigating Tool" as SOP search, identity check or
    # a lookup depending on the call; name the call: the first non-SOP Tool call.
    investigation = next(
        (row for row in evidence["tool_calls"] if row["event"].get("tool_id") not in PACKAGE_SOP_TOOLS),
        None,
    )
    m16_m1_context: dict[str, Any] = {"before": "the first operator tool call used to investigate the request"}
    if investigation is not None:
        m16_m1_context.update(
            investigation_tool_ref=investigation["ref"],
            investigation_tool_id=investigation["event"].get("tool_id"),
        )
    tasks.extend(
        (
            _task(
                f"{package['scenario_id']}:M16:M1",
                "M16",
                "M16-M1",
                M16_QUESTIONS["M16-M1"],
                "yes",
                m16_m1_context,
                ["conversation_or_tool"],
            ),
            _task(
                f"{package['scenario_id']}:M16:M2",
                "M16",
                "M16-M2",
                M16_QUESTIONS["M16-M2"],
                "yes",
                {},
                ["operator_utterance_required"],
            ),
        )
    )
    readback_values = (
        []
        if important_values is None
        else [
            row
            for row in important_values
            if row["readback_role"] == "readback"
        ]
    )
    consent_values = (
        []
        if important_values is None
        else [
            row
            for row in important_values
            if row["readback_role"] == "consent"
        ]
    )
    if readback_values:
        tasks.append(
            _task(
                f"{package['scenario_id']}:M16:M4",
                "M16",
                "M16-M4",
                M16_QUESTIONS["M16-M4"],
                "yes",
                {"important_values": readback_values},
                ["operator_utterance_required"],
            )
        )
    if consent_values:
        tasks.append(
            _task(
                f"{package['scenario_id']}:M16:M5",
                "M16",
                "M16-M5",
                M16_QUESTIONS["M16-M5"],
                "yes",
                {"consent_values": consent_values},
                ["operator_utterance_required"],
            )
        )

    edge = _edge(package)
    trigger = _trigger_source(edge, evidence)
    if edge is None or trigger is None:
        applicability["M19"] = "N/A"
    else:
        trigger_seq = _event_seq(trigger)
        for kind, question in M19_QUESTIONS.items():
            # A correction only restates a value: there is nothing to refuse, so CQ2 does not apply.
            if kind == "CQ2" and edge["field"] == "customer_correction":
                continue
            tasks.append(
                _task(
                    f"{package['scenario_id']}:M19:{kind}",
                    "M19",
                    kind,
                    question,
                    "yes",
                    {
                        "trigger": edge,
                        "trigger_ref": trigger["ref"],
                        "trigger_seq": trigger_seq,
                    },
                    ["operator_utterance_after_trigger"],
                )
            )
    return tasks, applicability


def _task(
    question_id: str,
    metric_id: str,
    kind: str,
    question: str,
    expected_answer: str,
    relevant_context: dict[str, Any],
    evidence_rules: list[str],
) -> dict[str, Any]:
    if expected_answer not in {"yes", "no"}:
        raise ValueError(f"{question_id}: expected answer must be yes or no")
    return {
        "question_id": question_id,
        "metric_id": metric_id,
        "kind": kind,
        "question": question,
        "expected_answer": expected_answer,
        "relevant_context": relevant_context,
        "evidence_rules": evidence_rules,
    }


def _run_judge(
    *,
    tasks: list[dict[str, Any]],
    evidence: dict[str, Any],
    config: dict[str, Any],
    cache_dir: Path,
    transport: Transport,
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("no applicable judge tasks")
    refs = [row["ref"] for row in evidence["conversation"] + evidence["tool_calls"]]
    if not refs:
        return {
            "by_id": {
                task["question_id"]: {
                    "status": "valid",
                    "answer": "no",
                    "evidence": [],
                    "contract_violations": [],
                }
                for task in tasks
            },
            "request_hash": None,
            "cache_reused": False,
            "receipt": None,
            "classification": "valid",
            "invalid_judgement_count": 0,
        }
    schema = judge_response_schema([task["question_id"] for task in tasks], refs)
    data = {
        "schema_version": INPUT_VERSION,
        "tasks": [{key: deepcopy(task[key]) for key in ("question_id", "metric_id", "kind", "question", "relevant_context", "evidence_rules")} for task in tasks],
        "conversation": evidence["conversation"],
        "tool_calls": evidence["tool_calls"],
    }
    input_size = len(canonical_json_bytes(data))
    if input_size > config["max_input_bytes"]:
        raise ValueError(f"judge input requires {input_size} bytes; truncation is forbidden")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    member = config["judge"]
    sampling = member["sampling"]
    request_payload = {
        "model": member["resolved_model_revision"],
        **judge_sampling_parameters(member),
        "max_completion_tokens": sampling["max_tokens"],
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "DATA_JSON=" + json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "conversation_log_judgements", "strict": True, "schema": schema}},
    }
    cached = find_content_addressed_receipt_by_request(root=cache_dir, namespace=JUDGE_NAMESPACE, request_payload=request_payload)
    if cached is not None:
        bundle = replay_content_addressed_bundle(root=cache_dir, receipt=cached)
        parsed = _validate_response(bundle["parsed"], tasks, evidence, schema)
        invalid_count = sum(row["status"] == "invalid" for row in parsed.values())
        return {"by_id": parsed, "request_hash": cached["bundle_identity"]["request_hash"], "cache_reused": True, "receipt": cached, "classification": "invalid" if invalid_count else "valid", "invalid_judgement_count": invalid_count}
    raw: dict[str, Any]
    parsed_value: dict[str, Any] | None = None
    failure: Exception | None = None
    try:
        api_key = os.environ.get(member["api_key_env"])
        if not api_key:
            raise ValueError(f"judge API key environment variable is missing: {member['api_key_env']}")
        raw = transport(f"{member['endpoint'].rstrip('/')}/v1/chat/completions", request_payload, {"Authorization": f"Bearer {api_key}"}, REQUEST_TIMEOUT_SECONDS)
        if raw.get("model") != member["resolved_model_revision"]:
            raise ValueError("judge resolved model revision mismatch")
        message = extract_chat_message(raw)
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("judge response content must be a JSON string")
        parsed_value = json.loads(content)
        parsed = _validate_response(parsed_value, tasks, evidence, schema)
    except Exception as exc:  # noqa: BLE001 - persisted then re-raised; no substitute result.
        failure = exc
        if "raw" not in locals():
            raw = {"error_type": type(exc).__name__, "message": str(exc)}
    if failure is not None:
        invalid_count = 0
    else:
        invalid_count = sum(row["status"] == "invalid" for row in parsed.values())
    metadata = {
        "classification": "invalid" if failure is not None or invalid_count else "valid",
        "call_kind": JUDGE_CALL_KIND,
        "model": member["resolved_model_revision"],
        "prompt_hash": sha256_json(prompt),
        "schema_hash": sha256_json(schema),
        "source_input_hash": sha256_json(data),
        "sampling": deepcopy(sampling),
        "input_bytes": input_size,
    }
    receipt = write_content_addressed_bundle(root=cache_dir, namespace=JUDGE_NAMESPACE, request_payload=request_payload, raw_response=raw, parsed=parsed_value, metadata=metadata)
    if failure is not None:
        raise ValueError(f"conversation-log judge failed: {failure}") from failure
    return {"by_id": parsed, "request_hash": receipt["bundle_identity"]["request_hash"], "cache_reused": False, "receipt": receipt, "classification": metadata["classification"], "invalid_judgement_count": invalid_count}


def _validate_response(
    value: Any,
    tasks: list[dict[str, Any]],
    evidence: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=str)
    if errors:
        raise ValueError(f"judge response violates schema: {errors[0].message}")
    expected_ids = [task["question_id"] for task in tasks]
    rows = value["judgements"]
    if [row["question_id"] for row in rows] != expected_ids:
        raise ValueError("judge response question order or coverage is invalid")
    sources = {row["ref"]: row for row in evidence["conversation"] + evidence["tool_calls"]}
    tasks_by_id = {task["question_id"]: task for task in tasks}
    decisions = {}
    for row in rows:
        refs = [item["ref"] for item in row["evidence"]]
        task = tasks_by_id[row["question_id"]]
        violations = _validate_evidence_rules(task, row["answer"], refs, sources)
        decision = deepcopy(row)
        decision["status"] = "invalid" if violations else "valid"
        decision["contract_violations"] = violations
        decisions[row["question_id"]] = decision
    return decisions


def _validate_evidence_rules(
    task: dict[str, Any],
    answer: str,
    refs: list[str],
    sources: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    kinds = {_source_kind(sources[ref]) for ref in refs}
    violations = []
    for rule in task["evidence_rules"]:
        if rule == "conversation_or_tool" and answer == "yes" and not kinds & {
            "conversation",
            "operator_utterance",
            "tool",
        }:
            violations.append({"rule": rule, "reason": "conversation/tool evidence is required"})
        if rule == "operator_utterance":
            if (answer == "yes" and not refs) or any(
                _source_kind(sources[ref]) != "operator_utterance" for ref in refs
            ):
                violations.append({"rule": rule, "reason": "only operator utterances may support this judgement"})
        if rule == "operator_utterance_required" and answer == "yes" and "operator_utterance" not in kinds:
            violations.append({"rule": rule, "reason": "operator utterance evidence is required"})
        if rule == "operator_or_tool" and answer == "yes" and not kinds & {"operator_utterance", "tool"}:
            violations.append({"rule": rule, "reason": "operator/tool response evidence is required"})
        if rule in {
            "operator_utterance_before_deadline",
            "operator_consent_before_deadline",
            "operator_utterance_after_trigger",
        } and answer == "yes":
            if rule == "operator_utterance_after_trigger":
                boundary_name = "trigger_seq"
            else:
                boundary_name = "deadline_seq"
            boundary = task["relevant_context"].get(boundary_name)
            sequences = [
                _event_seq(sources[ref])
                for ref in refs
                if _source_kind(sources[ref]) == "operator_utterance"
            ]
            if rule == "operator_utterance_before_deadline":
                if any(
                    _source_kind(sources[ref]) != "operator_utterance"
                    for ref in refs
                ):
                    violations.append(
                        {
                            "rule": rule,
                            "reason": "only operator utterances may support this judgement",
                        }
                    )
                in_window = any(sequence < boundary for sequence in sequences)
            elif rule == "operator_consent_before_deadline":
                in_window = any(sequence < boundary for sequence in sequences)
            else:
                in_window = any(sequence > boundary for sequence in sequences)
            if not in_window:
                violations.append(
                    {
                        "rule": rule,
                        "reason": "operator utterance evidence in the required event window is required",
                    }
                )
    return violations


def _score_tasks(
    tasks: list[dict[str, Any]],
    decisions: dict[str, Any],
    applicability: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = decisions["by_id"]
    rows: list[dict[str, Any]] = []
    for metric_id in TARGET_METRICS:
        metric_tasks = [task for task in tasks if task["metric_id"] == metric_id]
        if metric_id == "M16":
            rows.append(_score_m16(metric_tasks, by_id))
            continue
        if applicability[metric_id] is not True:
            status = applicability[metric_id]
            rows.append({"metric_id": metric_id, "status": status, "score": None, "reason": "explicitly_not_applicable" if status == "N/A" else "empty_applicable_contract", "judgements": []})
            continue
        judgement_rows = [_judgement_row(task, by_id[task["question_id"]]) for task in metric_tasks]
        score = sum(row["correct"] for row in judgement_rows) / len(judgement_rows)
        row = {"metric_id": metric_id, "status": "pass" if score == 1.0 else "fail", "score": score, "reason": "all_judgements_match" if score == 1.0 else "one_or_more_judgements_mismatch", "judgements": judgement_rows}
        rows.append(row)
    return rows


def _score_m16(
    tasks: list[dict[str, Any]], decisions: dict[str, Any]
) -> dict[str, Any]:
    observations = []
    for criterion_id in M16_CRITERIA:
        task = next((row for row in tasks if row["kind"] == criterion_id), None)
        if task is None:
            observations.append(
                {
                    "criterion_id": criterion_id,
                    "status": "N/M",
                    "passed": None,
                    "values": {"reason": "explicitly_not_applicable"},
                }
            )
            continue
        judgement = _judgement_row(task, decisions[task["question_id"]])
        observations.append(
            {
                "criterion_id": criterion_id,
                "status": "pass" if judgement["correct"] else "fail",
                "passed": judgement["correct"],
                "values": judgement,
            }
        )
    fired = [row["passed"] for row in observations if isinstance(row["passed"], bool)]
    return {
        "metric_id": "M16",
        "status": "measured",
        "machine_observations": observations,
        "closed_question_results": [],
        "fired_item_pass_rate": sum(fired) / len(fired),
        "reason": "independent_judge_rows_with_fired_item_pass_rate",
    }


def _judgement_row(task: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": task["question_id"],
        "kind": task["kind"],
        "status": decision["status"],
        "answer": decision["answer"],
        "expected_answer": task["expected_answer"],
        "correct": decision["status"] == "valid" and decision["answer"] == task["expected_answer"],
        "evidence": deepcopy(decision["evidence"]),
        "contract_violations": deepcopy(decision["contract_violations"]),
    }


def _evidence(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    conversation = record.get("conversation")
    tool_calls = record.get("tool_calls")
    event_log = record.get("event_log")
    for name, value in (("conversation", conversation), ("tool_calls", tool_calls), ("event_log", event_log)):
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise ValueError(f"record.{name} must be a complete object array")
    labeled_conversation = [{"ref": f"conversation[{index:04d}]", "event": deepcopy(event)} for index, event in enumerate(conversation)]
    labeled_tools = [{"ref": f"tool_calls[{index:04d}]", "event": deepcopy(event)} for index, event in enumerate(tool_calls)]
    labeled_events = [{"ref": f"event_log[{index:04d}]", "event": deepcopy(event)} for index, event in enumerate(event_log)]
    return {
        "conversation": labeled_conversation,
        "tool_calls": labeled_tools,
        "event_log": labeled_events,
    }


def _important_values(contracts: dict[str, Any]) -> list[dict[str, Any]] | None:
    if "m20" not in contracts:
        raise ValueError("package.contracts.m20 is required")
    contract = contracts["m20"]
    if contract is None:
        return None
    if not isinstance(contract, dict) or not isinstance(contract.get("important_values"), list):
        raise ValueError("package.contracts.m20.important_values must be an array")
    values = contract["important_values"]
    if not values:
        return None
    required = {
        "slot_id",
        "value",
        "value_type",
        "question_step_id",
        "readback_role",
    }
    identifier_keys = {"slot_id", "value_type", "question_step_id"}
    valid_readback_roles = {"readback", "consent", "statement"}
    for index, row in enumerate(values):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"package.contracts.m20.important_values[{index}] fields are invalid")
        if not all(
            isinstance(row[key], str) and row[key]
            for key in identifier_keys
        ):
            raise ValueError(f"package.contracts.m20.important_values[{index}] identifiers are invalid")
        if row["readback_role"] not in valid_readback_roles:
            raise ValueError(
                f"package.contracts.m20.important_values[{index}].readback_role is invalid"
            )
    return deepcopy(values)


def _edge(package: dict[str, Any]) -> dict[str, Any] | None:
    persona = package.get("persona")
    if not isinstance(persona, dict):
        raise ValueError("package.persona must be an object")
    active = [(field, persona.get(field)) for field in EDGE_FIELDS if persona.get(field) is not None]
    if len(active) > 1:
        raise ValueError("package.persona has multiple difficult-customer events")
    if not active:
        return None
    field, value = active[0]
    if not isinstance(value, dict) or not isinstance(value.get("utterance"), str) or not value["utterance"]:
        raise ValueError(f"package.persona.{field}.utterance must be a non-empty string")
    return {"field": field, **deepcopy(value)}


def _trigger_source(
    edge: dict[str, Any] | None, evidence: dict[str, Any]
) -> dict[str, Any] | None:
    if edge is None:
        return None
    for row in evidence["event_log"]:
        event = row["event"]
        fired = event.get("persona_fired")
        if event.get("actor") in {"user", "customer"} and (
            fired == edge["field"] or isinstance(fired, list) and edge["field"] in fired
        ):
            return row
    return None


def _event_seq(source: dict[str, Any]) -> int:
    event = source.get("event")
    seq = event.get("seq") if isinstance(event, dict) else None
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise ValueError(f"{source.get('ref')}: integer event seq is required")
    return seq


def _source_kind(source: dict[str, Any]) -> str:
    ref = source["ref"]
    event = source.get("event")
    if ref.startswith("tool_calls[") or isinstance(event, dict) and isinstance(event.get("tool_id"), str):
        return "tool"
    if isinstance(event, dict) and event.get("actor") == "operator" and isinstance(event.get("content"), str):
        return "operator_utterance"
    return "conversation"


def _judge_contract(package: dict[str, Any]) -> dict[str, Any]:
    contracts = package.get("contracts")
    selected_contracts = None
    if isinstance(contracts, dict):
        selected_contracts = {
            key: deepcopy(contracts.get(key)) for key in ("m05", "m20")
        }
    return {
        "scenario_id": package.get("scenario_id"),
        "contracts": selected_contracts,
        "persona": deepcopy(package.get("persona")),
    }


def _validate_config_value(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("conversation-log judge config must be an object")
    required = {"schema_version", "judge", "max_input_bytes", "minimum_accuracy"}
    if set(config) != required or config.get("schema_version") != CONFIG_VERSION:
        raise ValueError("conversation-log judge config fields or version are invalid")
    seed = validate_judge_member(config["judge"], label="conversation-log judge")["sampling"]["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("conversation-log judge seed must be an integer")
    maximum = config["max_input_bytes"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("max_input_bytes must be a positive integer")
    threshold = config["minimum_accuracy"]
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0 <= threshold <= 1:
        raise ValueError("minimum_accuracy must be between zero and one")
    return config
