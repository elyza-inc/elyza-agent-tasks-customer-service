"""Judge only written/spoken M05 consent declarations by semantic equality."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from elyza_agent_tasks_customer_service.evaluation.contracts.artifact_receipts import (
    canonical_json_bytes,
    find_content_addressed_receipt_by_request,
    replay_content_addressed_bundle,
    sha256_json,
    write_content_addressed_bundle,
)
from elyza_agent_tasks_customer_service.evaluation.contracts.run_outcome import _event_succeeded as _tool_call_succeeded
from elyza_agent_tasks_customer_service.evaluation.llm.llm_io import extract_chat_message, post_json
from elyza_agent_tasks_customer_service.evaluation.contracts.semantic_judge_contract import judge_sampling_parameters, validate_judge_member


RUNTIME_ROOT = Path(__file__).resolve().parent
PROMPT_PATH = RUNTIME_ROOT.parent / "prompt_templates" / "m05_consent_equivalence.md"
QUESTION_TEXT = "この2つの文は、同じ内容に同意しているか。"
INPUT_VERSION = "m05_consent_equivalence_input_v1"  # ジャッジのリクエスト本文に入る。変えるとキャッシュが使えなくなる
SCORER_VERSION = "m05_consent_judge"
JUDGE_NAMESPACE = "m05_consent_judge"
JUDGE_CALL_KIND = "m05_consent_equivalence"
REQUEST_TIMEOUT_SECONDS = 900.0
AGREEMENT_FIELD_SUFFIX = "_agreement"


Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def semantic_consent_declarations(package: dict[str, Any]) -> list[dict[str, str]]:
    """Return M05 declarations that require semantic written/spoken matching.

    ``package`` must be a package object. Only declared input rows whose
    names end in ``_agreement`` and whose non-empty string ``spoken`` differs
    from string ``value`` are returned. All other declared inputs remain
    outside this judge. Malformed package containers raise ``ValueError``.
    """

    if not isinstance(package, dict):
        raise ValueError("package must be an object")
    contracts = package.get("contracts")
    inputs = package.get("inputs")
    if not isinstance(contracts, dict) or not isinstance(inputs, dict):
        raise ValueError("package.contracts and package.inputs must be objects")
    completion = contracts.get("completion")
    declared = inputs.get("declared")
    if not isinstance(completion, dict) or not isinstance(declared, list):
        raise ValueError("package contracts.completion and inputs.declared are invalid")
    if contracts.get("m05") is None or not isinstance(
        completion.get("required_consent"), str
    ):
        return []
    if any(not isinstance(row, dict) for row in declared):
        raise ValueError("package.inputs.declared must be an object array")
    return [
        {
            "field": row["name"],
            "value": row["value"],
            "spoken": row["spoken"],
        }
        for row in declared
        if isinstance(row.get("name"), str)
        and row["name"].endswith(AGREEMENT_FIELD_SUFFIX)
        and isinstance(row.get("value"), str)
        and row["value"]
        and isinstance(row.get("spoken"), str)
        and row["spoken"]
        and row["value"] != row["spoken"]
    ]


def build_consent_pairs(
    package: dict[str, Any], record: dict[str, Any]
) -> tuple[list[dict[str, str]], str | None]:
    """Build declaration/customer pairs before the protected M05 action.

    ``record.conversation`` must be a complete object array. A present
    semantic declaration requires an exact ``contracts.m05.deadline_tool_id``
    Tool row. Missing deadlines or pre-deadline customer utterances return an
    empty pair list plus an explicit not-measurable reason; values are never
    substituted.
    """

    declarations = semantic_consent_declarations(package)
    if not declarations:
        return [], "semantic_consent_not_applicable"
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    conversation = record.get("conversation")
    if not isinstance(conversation, list) or any(
        not isinstance(row, dict) for row in conversation
    ):
        raise ValueError("record.conversation must be a complete object array")
    deadline_tool = package["contracts"]["m05"].get("deadline_tool_id")
    if not isinstance(deadline_tool, str) or not deadline_tool:
        raise ValueError("package.contracts.m05.deadline_tool_id must be a string")
    deadline_indexes = [
        index
        for index, row in enumerate(conversation)
        if row.get("tool_id") == deadline_tool
    ]
    deadline_index = next(
        (index for index in deadline_indexes if _tool_call_succeeded(conversation[index])),
        None,
    )
    if deadline_index is None and deadline_indexes:
        deadline_index = deadline_indexes[0]
    if deadline_index is None:
        return [], "m05_protected_action_not_observed"
    utterances = [
        (index, row["content"])
        for index, row in enumerate(conversation[:deadline_index])
        if row.get("actor") == "customer"
        and isinstance(row.get("content"), str)
        and row["content"].strip()
    ]
    if not utterances:
        return [], "pre_action_customer_utterance_missing"
    scenario_id = package.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("package.scenario_id must be a string")
    return [
        {
            "question_id": f"{scenario_id}:M05:{row['field']}:{index:04d}",
            "question": QUESTION_TEXT,
            "declared_value": row["value"],
            "customer_utterance": utterance,
            "evidence_ref": f"conversation[{index:04d}]",
        }
        for row in declarations
        for index, utterance in utterances
    ], None


def judge_m05_consent(
    *,
    package: dict[str, Any],
    record: dict[str, Any],
    config: dict[str, Any],
    cache_dir: Path,
    transport: Transport = post_json,
) -> dict[str, Any]:
    """Return M05 semantic consent status without substituting judge failures."""

    pairs, unavailable_reason = build_consent_pairs(package, record)
    if not pairs:
        status = "N/A" if unavailable_reason == "semantic_consent_not_applicable" else "N/M"
        return _result(status=status, reason=unavailable_reason, judgements=[], judge=None)
    judged = judge_consent_pairs(
        pairs=pairs,
        config=config,
        cache_dir=cache_dir,
        transport=transport,
    )
    if judged["status"] == "N/M":
        return judged
    passed = any(row["answer"] == "yes" for row in judged["judgements"])
    judged["status"] = "pass" if passed else "fail"
    judged["reason"] = (
        "semantically_matching_consent_observed"
        if passed
        else "semantically_matching_consent_not_observed"
    )
    return judged


def judge_consent_pairs(
    *,
    pairs: list[dict[str, str]],
    config: dict[str, Any],
    cache_dir: Path,
    transport: Transport = post_json,
) -> dict[str, Any]:
    """Judge non-empty consent pairs and persist the input-addressed result."""

    _validate_pairs(pairs)
    member = _judge_member(config)
    schema = _response_schema([row["question_id"] for row in pairs])
    data = {"schema_version": INPUT_VERSION, "pairs": deepcopy(pairs)}
    maximum = config.get("max_input_bytes")
    input_bytes = len(canonical_json_bytes(data))
    if isinstance(maximum, bool) or not isinstance(maximum, int) or input_bytes > maximum:
        raise ValueError("M05 consent judge input exceeds max_input_bytes")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    sampling = member["sampling"]
    request_payload = {
        "model": member["resolved_model_revision"],
        **judge_sampling_parameters(member),
        "max_completion_tokens": sampling["max_tokens"],
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "DATA_JSON="
                + json.dumps(
                    data,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "m05_consent_equivalence",
                "strict": True,
                "schema": schema,
            },
        },
    }
    cached = find_content_addressed_receipt_by_request(
        root=cache_dir,
        namespace=JUDGE_NAMESPACE,
        request_payload=request_payload,
    )
    if cached is not None:
        bundle = replay_content_addressed_bundle(root=cache_dir, receipt=cached)
        return _from_bundle(
            parsed=bundle["parsed"],
            receipt=cached,
            pairs=pairs,
            schema=schema,
            cache_reused=True,
        )
    parsed_value: dict[str, Any] | None = None
    failure: Exception | None = None
    raw: Any = None
    try:
        api_key = os.environ.get(member["api_key_env"])
        if not api_key:
            raise ValueError(
                f"judge API key environment variable is missing: {member['api_key_env']}"
            )
        raw = transport(
            f"{member['endpoint'].rstrip('/')}/v1/chat/completions",
            request_payload,
            {"Authorization": f"Bearer {api_key}"},
            REQUEST_TIMEOUT_SECONDS,
        )
        if raw.get("model") != member["resolved_model_revision"]:
            raise ValueError("judge resolved model revision mismatch")
        content = extract_chat_message(raw).get("content")
        if not isinstance(content, str):
            raise ValueError("judge response content must be a JSON string")
        parsed_value = json.loads(content)
        _validate_response(parsed_value, pairs, schema)
    except Exception as exc:  # noqa: BLE001 - invalid result is persisted as N/M.
        failure = exc
        if raw is None:
            raw = {"error_type": type(exc).__name__, "message": str(exc)}
    metadata = {
        "classification": "valid" if failure is None else "invalid",
        "call_kind": JUDGE_CALL_KIND,
        "model": member["resolved_model_revision"],
        "prompt_hash": sha256_json(prompt),
        "schema_hash": sha256_json(schema),
        "source_input_hash": sha256_json(data),
        "sampling": deepcopy(sampling),
        "input_bytes": input_bytes,
    }
    receipt = write_content_addressed_bundle(
        root=cache_dir,
        namespace=JUDGE_NAMESPACE,
        request_payload=request_payload,
        raw_response=raw,
        parsed=parsed_value,
        metadata=metadata,
    )
    if failure is not None:
        return _result(
            status="N/M",
            reason="semantic_consent_judge_unavailable",
            judgements=[],
            judge=_judge_summary(receipt, cache_reused=False),
        )
    return _from_bundle(
        parsed=parsed_value,
        receipt=receipt,
        pairs=pairs,
        schema=schema,
        cache_reused=False,
    )


def _from_bundle(
    *,
    parsed: Any,
    receipt: dict[str, Any],
    pairs: list[dict[str, str]],
    schema: dict[str, Any],
    cache_reused: bool,
) -> dict[str, Any]:
    if receipt["metadata"].get("classification") != "valid":
        return _result(
            status="N/M",
            reason="semantic_consent_judge_unavailable",
            judgements=[],
            judge=_judge_summary(receipt, cache_reused=cache_reused),
        )
    rows = _validate_response(parsed, pairs, schema)
    return _result(
        status="measured",
        reason="semantic_consent_judged",
        judgements=rows,
        judge=_judge_summary(receipt, cache_reused=cache_reused),
    )


def _judge_summary(receipt: dict[str, Any], *, cache_reused: bool) -> dict[str, Any]:
    return {
        "model": receipt["metadata"]["model"],
        "sampling": deepcopy(receipt["metadata"]["sampling"]),
        "input_hash": receipt["metadata"]["source_input_hash"],
        "request_hash": receipt["bundle_identity"]["request_hash"],
        "cache_reused": cache_reused,
        "receipt": receipt,
    }


def _judge_member(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("judge config must be an object")
    member = validate_judge_member(config.get("judge"), label="M05 consent judge")
    if isinstance(member["sampling"]["seed"], bool) or not isinstance(
        member["sampling"]["seed"], int
    ):
        raise ValueError("M05 consent judge seed must be an integer")
    return member


def _response_schema(question_ids: list[str]) -> dict[str, Any]:
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
                    "required": ["question_id", "answer", "reason"],
                    "properties": {
                        "question_id": {"enum": question_ids},
                        "answer": {"enum": ["yes", "no"]},
                        "reason": {"type": "string", "minLength": 1},
                    },
                },
            }
        },
    }


def _validate_response(
    value: Any,
    pairs: list[dict[str, str]],
    schema: dict[str, Any],
) -> list[dict[str, str]]:
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=str)
    if errors:
        raise ValueError(f"M05 consent judge response violates schema: {errors[0].message}")
    expected = [row["question_id"] for row in pairs]
    rows = value["judgements"]
    if [row["question_id"] for row in rows] != expected:
        raise ValueError("M05 consent judge response order or coverage is invalid")
    return deepcopy(rows)


def _validate_pairs(pairs: Any) -> None:
    required = {
        "question_id",
        "question",
        "declared_value",
        "customer_utterance",
        "evidence_ref",
    }
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("M05 consent pairs must be a non-empty array")
    for index, row in enumerate(pairs):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"M05 consent pairs[{index}] fields are invalid")
        if not all(isinstance(row[key], str) and row[key] for key in required):
            raise ValueError(f"M05 consent pairs[{index}] values are invalid")
        if row["question"] != QUESTION_TEXT:
            raise ValueError(f"M05 consent pairs[{index}].question is invalid")
    ids = [row["question_id"] for row in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("M05 consent pair question IDs must be unique")


def _result(
    *,
    status: str,
    reason: str | None,
    judgements: list[dict[str, str]],
    judge: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCORER_VERSION,
        "metric_id": "M05",
        "status": status,
        "reason": reason,
        "judgements": judgements,
        "judge": judge,
    }
