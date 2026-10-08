"""Run chat evaluation directly from one package mapping."""

from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
import json
from pathlib import Path
import time
import unicodedata
from typing import Any, Callable, Iterable
from uuid import uuid4


from elyza_agent_tasks_customer_service.evaluation.engine.fault_injection import FaultInjector
from elyza_agent_tasks_customer_service.evaluation.engine.package_build import TICKET_CATALOG_ACTION_CODES, TICKET_COMPARISON_FIELDS
from elyza_agent_tasks_customer_service.evaluation.contracts.type_contracts import (
    JSON_SCHEMA_TYPES,
    _normalized_date,
    _normalized_match_value,
    normalize_typed_value,
    value_matches_type,
)
from elyza_agent_tasks_customer_service.evaluation.llm.llm_io import (
    accumulate_chat_usage,
    apply_chat_provider_profile,
    chat_completions_url,
    chat_with_context_retry,
    empty_chat_usage,
    extract_chat_message,
    post_chat_via_responses,
    post_chat_with_prompt_cache,
    post_json,
)
from elyza_agent_tasks_customer_service.evaluation.contracts.run_outcome import HANDLING_TYPE_ALIASES, build_run_outcome_from_facts
from elyza_agent_tasks_customer_service.evaluation.contracts.variants import normalize_variant


SOP_CATEGORY_SEARCH_TOOL = "search_sop_categories"
SOP_SEARCH_TOOL = "search_sops_in_category"
SOP_DETAIL_TOOL = "get_sop"
PROVENANCE_RESULT_METADATA_FIELDS = frozenset(("ok", "error", "message"))
SOP_TOOL_ARGUMENTS = {
    SOP_CATEGORY_SEARCH_TOOL: {"query"},
    SOP_SEARCH_TOOL: {"category_id", "query"},
    SOP_DETAIL_TOOL: {"sop_id"},
}
DEFAULT_SOP_SEARCH_TOP_K = 4
PACKAGE_REQUIRED_FIELDS = (
    "business",
    "inputs",
    "domain",
    "world_schema",
    "world",
    "tools",
    "sop_catalog",
    "persona",
)
MAX_COMPLETION_TOKENS = 4096
USER_MAX_COMPLETION_TOKENS = 1024
TICKET_MAX_COMPLETION_TOKENS = 16384
MAX_SCHEMA_REPAIRS = 2
MAX_TICKET_REPAIRS = 2
TICKET_REPAIR_OUTPUT_CHARS = 2000
RUN_SCHEMA_VERSION = "package_run"
TICKET_RESPONSE_FORMAT_NAME = "operator_post_call_ticket"
TICKET_CONVERSATION_FIELDS = (
    "refusal_reason",
    "promises",
    "answer_summary",
)
OPERATOR_TICKET_FIELDS = TICKET_COMPARISON_FIELDS + TICKET_CONVERSATION_FIELDS
CUSTOMER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "customer_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "minLength": 1},
                "end_conversation": {"type": "boolean"},
                "persona_fired": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["message", "end_conversation", "persona_fired"],
            "additionalProperties": False,
        },
    },
}


def _customer_response_format(unfired_names: Iterable[str]) -> dict[str, Any]:
    response_format = deepcopy(CUSTOMER_RESPONSE_FORMAT)
    persona_items = response_format["json_schema"]["schema"]["properties"]["persona_fired"]["items"]
    persona_items["enum"] = sorted(unfired_names)
    return response_format


class PackageError(ValueError):
    """A package cannot be executed without inventing a missing value."""


def _stop(scenario_id: str, field: str, message: str) -> None:
    raise PackageError(f"{scenario_id}: {field}: {message}")


def _mapping(value: Any, scenario_id: str, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _stop(scenario_id, field, "mapping is required")
    return value


def _required(mapping: dict[str, Any], key: str, scenario_id: str, field: str = "package") -> Any:
    if key not in mapping:
        _stop(scenario_id, f"{field}.{key}", "value is required")
    return mapping[key]


def _string(mapping: dict[str, Any], key: str, scenario_id: str, field: str) -> str:
    value = _required(mapping, key, scenario_id, field)
    if not isinstance(value, str) or not value.strip():
        _stop(scenario_id, f"{field}.{key}", "non-empty string is required")
    return value


def _invalid_tool_calls_reason(raw_calls: Any) -> str | None:
    """Return why an operator ``tool_calls`` value cannot run, or ``None`` when every call is well formed."""

    if not isinstance(raw_calls, list):
        return "operator_response.tool_calls: list is required"
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict) or not isinstance(raw_call.get("function"), dict):
            return "operator_response.tool_call.function: mapping is required"
        for owner, key, field in (
            (raw_call["function"], "name", "operator_response.tool_call.function.name"),
            (raw_call, "id", "operator_response.tool_call.id"),
        ):
            value = owner.get(key)
            if not isinstance(value, str) or not value.strip():
                return f"{field}: non-empty string is required"
    return None


def _public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(row["values"]) for row in rows]


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_response_json(content: str, *, label: str) -> tuple[dict[str, Any], bool]:
    """Parse an object from exact JSON or fallback response text.

    Exact JSON is tried first. Only after that fails, JSON inside a fenced
    ``json`` block or the first decodable object beginning with ``{`` is accepted.
    Missing, malformed, array, and scalar values raise ``ValueError``.
    """

    try:
        candidate = json.loads(content)
        fallback_parse = False
    except json.JSONDecodeError as original_error:
        stripped = content.strip()
        fence_start = stripped.find("```json")
        if fence_start >= 0:
            start = fence_start + len("```json")
            fence_end = stripped.find("```", start)
            fallback_text = stripped[start:fence_end] if fence_end >= 0 else ""
            decoder = json.loads
        else:
            start = stripped.find("{")
            fallback_text = stripped[start:] if start >= 0 else ""
            decoder = lambda value: json.JSONDecoder().raw_decode(value)[0]
        try:
            candidate = decoder(fallback_text.lstrip())
        except json.JSONDecodeError:
            raise ValueError(f"{label} is not valid JSON: {content}") from original_error
        fallback_parse = True
    if not isinstance(candidate, dict):
        raise ValueError(f"{label} must be a JSON object: {content}")
    return candidate, fallback_parse


def _matching_value_path(
    candidate: Any,
    expected: Any,
    *,
    column: str,
    value_type: str | None,
    contains_text: bool,
    path: str,
) -> str | None:
    if isinstance(candidate, dict):
        for key, value in candidate.items():
            matched = _matching_value_path(
                value,
                expected,
                column=column,
                value_type=value_type,
                contains_text=contains_text,
                path=f"{path}.{key}",
            )
            if matched is not None:
                return matched
        return None
    if isinstance(candidate, list):
        for index, value in enumerate(candidate):
            matched = _matching_value_path(
                value,
                expected,
                column=column,
                value_type=value_type,
                contains_text=contains_text,
                path=f"{path}[{index}]",
            )
            if matched is not None:
                return matched
        return None
    if value_type is not None:
        try:
            candidate = normalize_typed_value(candidate, value_type)
            expected = normalize_typed_value(expected, value_type)
        except ValueError:
            return None
    normalized_candidate = _normalized_match_value(candidate, column, value_type)
    normalized_expected = _normalized_match_value(expected, column, value_type)
    if (
        contains_text
        and isinstance(normalized_candidate, str)
        and isinstance(normalized_expected, str)
        and normalized_expected
    ):
        return path if normalized_expected in normalized_candidate else None
    return path if normalized_candidate == normalized_expected else None


def _event(
    events: list[dict[str, Any]],
    *,
    run_id: str,
    epoch: str,
    actor: str,
    event_type: str,
    **fields: Any,
) -> dict[str, Any]:
    seq = len(events) + 1
    event_id = f"event:{run_id}:{seq}"
    row = {
        "event_id": event_id,
        "event_ref": event_id,
        "episode_id": run_id,
        "seq": seq,
        "epoch": epoch,
        "actor": actor,
        "event_type": event_type,
        **deepcopy(fields),
    }
    events.append(row)
    return row


def _validate_ticket_contract(value: Any, scenario_id: str) -> dict[str, Any]:
    contract = _mapping(value, scenario_id, "post_call_ticket_contract")
    if set(contract) != {"schema_version", "enum_catalog", "comparison_fields"}:
        _stop(scenario_id, "post_call_ticket_contract", "fields are invalid")
    if contract["schema_version"] != "post_call_ticket_contract":
        _stop(scenario_id, "post_call_ticket_contract.schema_version", "unsupported value")
    if contract["comparison_fields"] != list(TICKET_COMPARISON_FIELDS):
        _stop(scenario_id, "post_call_ticket_contract.comparison_fields", "must match the four ticket fields")
    catalog = _mapping(contract["enum_catalog"], scenario_id, "post_call_ticket_contract.enum_catalog")
    required = {"handling_types", "action_codes", "id_types", "evidence_kinds"}
    if set(catalog) != required:
        _stop(scenario_id, "post_call_ticket_contract.enum_catalog", "fields are invalid")
    for field in required:
        values = catalog[field]
        if not isinstance(values, list) or len(values) != len(set(values)) or any(
            not isinstance(item, str) or not item for item in values
        ):
            _stop(scenario_id, f"post_call_ticket_contract.enum_catalog.{field}", "unique string array is required")
    if any(not catalog[field] for field in required):
        _stop(scenario_id, "post_call_ticket_contract.enum_catalog", "required catalogs cannot be empty")
    return contract


def _ticket_response_format(contract: dict[str, Any]) -> dict[str, Any]:
    catalog = contract["enum_catalog"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": TICKET_RESPONSE_FORMAT_NAME,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "handling_type": {"type": "string", "enum": catalog["handling_types"]},
                    "target_ids": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id_type": {"type": "string", "enum": catalog["id_types"]},
                                "value": {"type": "string", "minLength": 1},
                            },
                            "required": ["id_type", "value"],
                            "additionalProperties": False,
                        },
                    },
                    "performed_actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action_code": {"type": "string", "enum": catalog["action_codes"]},
                                "target_ref": {"type": ["string", "null"]},
                                "outcome": {"type": "string", "enum": ["succeeded", "failed"]},
                                "receipt_refs": {"type": "array", "items": {"type": "string", "minLength": 1}},
                            },
                            "required": ["action_code", "target_ref", "outcome", "receipt_refs"],
                            "additionalProperties": False,
                        },
                    },
                    "evidence_refs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": catalog["evidence_kinds"]},
                                "ref": {"type": "string", "minLength": 1},
                            },
                            "required": ["kind", "ref"],
                            "additionalProperties": False,
                        },
                    },
                    "refusal_reason": {"type": ["string", "null"]},
                    "promises": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "owner_id": {"type": "string", "minLength": 1},
                                "channel": {"type": "string", "minLength": 1},
                                "deadline": {"type": "string", "minLength": 1},
                            },
                            "required": ["owner_id", "channel", "deadline"],
                            "additionalProperties": False,
                        },
                    },
                    "answer_summary": {"type": ["string", "null"]},
                },
                "required": list(OPERATOR_TICKET_FIELDS),
                "additionalProperties": False,
            },
        },
    }


def _validate_ticket(value: Any, scenario_id: str, contract: dict[str, Any]) -> dict[str, Any]:
    ticket = _mapping(value, scenario_id, "operator_ticket_artifact.ticket")
    if set(ticket) != set(OPERATOR_TICKET_FIELDS):
        _stop(
            scenario_id,
            "operator_ticket_artifact.ticket",
            f"must contain exactly {list(OPERATOR_TICKET_FIELDS)}",
        )
    catalog = contract["enum_catalog"]
    if ticket["handling_type"] not in catalog["handling_types"]:
        _stop(scenario_id, "operator_ticket_artifact.ticket.handling_type", "unsupported value")
    for field in TICKET_COMPARISON_FIELDS[1:] + ("promises",):
        if not isinstance(ticket[field], list):
            _stop(scenario_id, f"operator_ticket_artifact.ticket.{field}", "array is required")
    for index, item in enumerate(ticket["target_ids"]):
        if not isinstance(item, dict) or set(item) != {"id_type", "value"}:
            _stop(scenario_id, f"operator_ticket_artifact.ticket.target_ids[{index}]", "fields are invalid")
        if item["id_type"] not in catalog["id_types"] or not isinstance(item["value"], str) or not item["value"]:
            _stop(scenario_id, f"operator_ticket_artifact.ticket.target_ids[{index}]", "value is invalid")
    for index, item in enumerate(ticket["performed_actions"]):
        if not isinstance(item, dict) or set(item) != {"action_code", "target_ref", "outcome", "receipt_refs"}:
            _stop(scenario_id, f"operator_ticket_artifact.ticket.performed_actions[{index}]", "fields are invalid")
        if item["action_code"] not in catalog["action_codes"]:
            _stop(
                scenario_id,
                f"operator_ticket_artifact.ticket.performed_actions[{index}].action_code",
                f"must be one of {catalog['action_codes']}",
            )
        if item["outcome"] not in {"succeeded", "failed"}:
            _stop(
                scenario_id,
                f"operator_ticket_artifact.ticket.performed_actions[{index}].outcome",
                "must be one of ['succeeded', 'failed']",
            )
        if item["target_ref"] is not None and not isinstance(item["target_ref"], str):
            _stop(scenario_id, f"operator_ticket_artifact.ticket.performed_actions[{index}].target_ref", "string or null is required")
        if not isinstance(item["receipt_refs"], list) or any(not isinstance(ref, str) or not ref for ref in item["receipt_refs"]):
            _stop(scenario_id, f"operator_ticket_artifact.ticket.performed_actions[{index}].receipt_refs", "string array is required")
    for index, item in enumerate(ticket["evidence_refs"]):
        if not isinstance(item, dict) or set(item) != {"kind", "ref"}:
            _stop(scenario_id, f"operator_ticket_artifact.ticket.evidence_refs[{index}]", "fields are invalid")
        if item["kind"] not in catalog["evidence_kinds"] or not isinstance(item["ref"], str) or not item["ref"]:
            _stop(scenario_id, f"operator_ticket_artifact.ticket.evidence_refs[{index}]", "value is invalid")
    refusal_reason = ticket["refusal_reason"]
    if refusal_reason is not None and (not isinstance(refusal_reason, str) or not refusal_reason):
        _stop(scenario_id, "operator_ticket_artifact.ticket.refusal_reason", "non-empty string or null is required")
    for index, promise in enumerate(ticket["promises"]):
        if not isinstance(promise, dict) or set(promise) != {"owner_id", "channel", "deadline"}:
            _stop(scenario_id, f"operator_ticket_artifact.ticket.promises[{index}]", "fields are invalid")
        if any(not isinstance(promise[field], str) or not promise[field] for field in promise):
            _stop(scenario_id, f"operator_ticket_artifact.ticket.promises[{index}]", "non-empty string values are required")
    answer_summary = ticket["answer_summary"]
    if answer_summary is not None and (not isinstance(answer_summary, str) or not answer_summary):
        _stop(scenario_id, "operator_ticket_artifact.ticket.answer_summary", "non-empty string or null is required")
    return ticket


def _generate_operator_ticket(
    *,
    scenario_id: str,
    endpoint: str,
    model: str,
    headers: dict[str, str],
    timeout_sec: float,
    extra_body: dict[str, Any],
    conversation: list[dict[str, Any]],
    call_events: list[dict[str, Any]],
    ticket_contract: dict[str, Any],
    transport: Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]],
    partial_record: dict[str, Any] | None = None,
    context_length_retry: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    """Generate one ticket from public run evidence only.

    ``conversation`` and ``call_events`` are decoded object arrays from the
    completed call. Malformed or non-schema model output raises
    ``PackageError`` containing the scenario ID and ticket field.
    """

    prompt = (
        "あなたは直前の通話を担当した同一オペレータです。通話後の応対記録を作成してください。"
        "入力の会話ログと実行証跡だけを使い、正解や期待結果を推測しないでください。"
        "handling_type、target_ids、performed_actions、evidence_refsに加え、会話中に実際に伝えた"
        "拒否理由を自然な文章でrefusal_reason、折返し約束をpromises、回答要約をanswer_summaryへ記録してください。"
        "型規則(厳守): promisesは常に配列型で、該当なしはnullではなく[]とします。"
        "refusal_reasonとanswer_summaryは該当なしのときnullとします。"
        "target_idsは{\"id_type\": <契約のid_type>, \"value\": <値>}のオブジェクト配列とします。"
        "performed_actionsの各要素はaction_code/target_ref/outcome/receipt_refsを必ず持ちます。"
        "enum値は契約のenum_catalogだけを使います。"
        "target_idsは成功した非SOPツール結果の文字列型*_idを"
        "結果とフィールドの出現順に重複を除いて記録します。performed_actionsは各非SOPツール結果を"
        "action_code/target_ref/outcome/receipt_refsで記録し、target_refはその結果の最初のtarget_id、"
        "対象なしはnull、outcomeはtool_resultのokがtrueならsucceeded(該当0件でも)、falseならfailed、receipt_refsはtool_resultのevent_refです。evidence_refsは成功した非SOP"
        "tool_resultをtool_receipt、取得したSOPをsopのSOP:<id>として出現順に記録し、他の参照は"
        "加えません。handling_typeはchange→change_procedure、guidance→guidance_inquiry_answer、"
        "callback→callback_commitmentと正規化します。応対記録の登録(case_recordsへの書き込み)が成功していない場合、handling_typeはincompleteとします。指定JSON以外は返さないでください。"
    )
    public_call_events = deepcopy(call_events)
    for event in public_call_events:
        if isinstance(event, dict) and event.get("event_type") == "message":
            event.pop("content", None)
            # Audio runs attach ASR candidates, wav paths and clocks to messages; the ticket needs none of it.
            event.pop("metadata", None)
    public_input = {
        "conversation": deepcopy(conversation),
        "call_events": public_call_events,
        "ticket_contract": deepcopy(ticket_contract),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "PUBLIC_RUN_EVIDENCE_JSON=" + _json_text(public_input)},
        ],
        **extra_body,
    }
    payload["response_format"] = _ticket_response_format(ticket_contract)
    apply_chat_provider_profile(
        payload,
        profile="auto",
        endpoint=endpoint,
        temperature=0.0,
        max_tokens=TICKET_MAX_COMPLETION_TOKENS,
    )
    repair_events: list[dict[str, Any]] = []
    fallback_parse = False
    if partial_record is not None:
        partial_record.update(ticket_repair_count=0, ticket_repair_events=repair_events)
    for attempt in range(MAX_TICKET_REPAIRS + 1):
        response = chat_with_context_retry(
            transport,
            chat_completions_url(endpoint),
            payload,
            headers,
            timeout_sec,
            phase="ticket",
            events=context_length_retry if context_length_retry is not None else [],
        )
        if partial_record is not None:
            partial_record["operator_ticket_response"] = deepcopy(response)
        message = extract_chat_message(response)
        if response["choices"][0].get("finish_reason") == "length":
            # The operator failed to produce a ticket; score the run without one.
            if partial_record is not None:
                partial_record["operator_ticket_failure"] = {
                    "reason": "output token limit reached "
                    f"(finish_reason=length, max_completion_tokens={TICKET_MAX_COMPLETION_TOKENS})",
                }
            return None, payload, fallback_parse
        content = message.get("content")
        try:
            if not isinstance(content, str) or not content.strip():
                raise ValueError("operator response is required")
            candidate, used_fallback = _parse_response_json(content, label="operator post-call ticket")
            fallback_parse = fallback_parse or used_fallback
            ticket = _validate_ticket(candidate, scenario_id, ticket_contract)
        except (ValueError, PackageError) as exc:
            if attempt == MAX_TICKET_REPAIRS:
                if partial_record is not None:
                    partial_record["operator_ticket_failure"] = {"reason": str(exc)}
                return None, payload, fallback_parse
            violation = str(exc)
            if isinstance(content, str) and content:
                violation = violation.replace(content, "<previous output omitted>")
            repair_events.append({"attempt": attempt + 1, "violation": violation})
            if partial_record is not None:
                partial_record["ticket_repair_count"] = len(repair_events)
            if isinstance(content, str) and content:
                payload["messages"].append(
                    {"role": "assistant", "content": content[:TICKET_REPAIR_OUTPUT_CHARS]}
                )
            payload["messages"].append(
                {
                    "role": "user",
                    "content": (
                        f"前回の応答は次の違反があります: {violation}。"
                        "以下の期待スキーマにある型、必須フィールド、enum許容値に従い、"
                        "修正したJSONだけを再出力してください。EXPECTED_SCHEMA_JSON="
                        + _json_text(payload["response_format"]["json_schema"]["schema"])
                    ),
                }
            )
            continue
        return {"status": "submitted", "reason": "schema_valid", "ticket": ticket}, payload, fallback_parse
    raise AssertionError("unreachable")


class PackageRuntime:
    """Execute the runtime-visible fields of one package.

    Accepted input is a decoded task mapping with the fields in
    ``PACKAGE_REQUIRED_FIELDS`` and optional ``error_injection``. Optional
    grading fields enable completion checks and post-call tickets; without
    them, the task can run but cannot produce a scorable record. Missing or
    wrongly shaped task values raise ``PackageError`` with the scenario ID
    and field path.
    """

    def __init__(self, package: dict[str, Any], *, scenario_id: str) -> None:
        self.scenario_id = scenario_id
        if not isinstance(package, dict):
            _stop(scenario_id, "package", "mapping is required")
        for field in PACKAGE_REQUIRED_FIELDS:
            _required(package, field, scenario_id)
        declared_id = package.get("scenario_id")
        if declared_id != scenario_id:
            _stop(scenario_id, "package.scenario_id", f"expected {scenario_id!r}, got {declared_id!r}")

        self.business = _mapping(package["business"], scenario_id, "business")
        self.inputs = _mapping(package["inputs"], scenario_id, "inputs")
        self.domain = _mapping(package["domain"], scenario_id, "domain")
        self.world_schema = _mapping(package["world_schema"], scenario_id, "world_schema")
        self.world = deepcopy(_mapping(package["world"], scenario_id, "world"))
        self.tools = _mapping(package["tools"], scenario_id, "tools")
        self.sop_catalog = _mapping(package["sop_catalog"], scenario_id, "sop_catalog")
        self.persona = _mapping(package["persona"], scenario_id, "persona")
        contracts = package.get("contracts")
        self.completion: dict[str, Any] | None = None
        if contracts is not None:
            self.completion = _mapping(
                _required(_mapping(contracts, scenario_id, "contracts"), "completion", scenario_id, "contracts"),
                scenario_id,
                "contracts.completion",
            )
        ticket_contract = package.get("post_call_ticket_contract")
        self.ticket_contract: dict[str, Any] | None = None
        if ticket_contract is not None:
            self.ticket_contract = _validate_ticket_contract(ticket_contract, scenario_id)
            if set(self.ticket_contract["enum_catalog"]["action_codes"]) != set(self.tools) | set(TICKET_CATALOG_ACTION_CODES):
                _stop(scenario_id, "post_call_ticket_contract.enum_catalog.action_codes", "must match package tools")
        self.fault = FaultInjector(package.get("error_injection"), scenario_id=scenario_id)
        self.verified_customer: str | None = None
        self.result_rows: list[dict[str, Any]] = []
        self.tool_log: list[dict[str, Any]] = []
        self.schema_repair_events: list[dict[str, Any]] = []
        self._schema_repair_attempts: dict[str, int] = {}
        self.categories: dict[str, dict[str, Any]] = {}
        self.sops: dict[str, dict[str, Any]] = {}
        self.sop_categories: dict[str, str] = {}
        self.available_category_ids: set[str] = set()
        self.available_sop_ids: set[str] = set()
        self.sop_search_top_k = DEFAULT_SOP_SEARCH_TOP_K
        self._validate_runtime_contract()

    def set_sop_search_top_k(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            _stop(self.scenario_id, "runtime.sop_search_top_k", "positive integer is required")
        self.sop_search_top_k = value

    @property
    def initial_user_request(self) -> str:
        return _string(self.business, "initial_user_request", self.scenario_id, "business")

    def _validate_runtime_contract(self) -> None:
        _ = self.initial_user_request
        for group in ("identity", "declared"):
            items = _required(self.inputs, group, self.scenario_id, "inputs")
            if not isinstance(items, list):
                _stop(self.scenario_id, f"inputs.{group}", "list is required")
            for index, value in enumerate(items):
                item = _mapping(value, self.scenario_id, f"inputs.{group}[{index}]")
                _string(item, "name", self.scenario_id, f"inputs.{group}[{index}]")
                _required(item, "value", self.scenario_id, f"inputs.{group}[{index}]")
                _string(item, "question", self.scenario_id, f"inputs.{group}[{index}]")
                _string(item, "question_step_id", self.scenario_id, f"inputs.{group}[{index}]")
        _required(self.inputs, "reference_date_spec", self.scenario_id, "inputs")
        entities = _mapping(_required(self.world_schema, "entities", self.scenario_id, "world_schema"), self.scenario_id, "world_schema.entities")
        customer_table = _string(self.world_schema, "customer_table", self.scenario_id, "world_schema")
        search_policy = _mapping(
            _required(self.world_schema, "search_policy", self.scenario_id, "world_schema"),
            self.scenario_id,
            "world_schema.search_policy",
        )
        if _string(search_policy, "order_by", self.scenario_id, "world_schema.search_policy") != "row_identity":
            _stop(self.scenario_id, "world_schema.search_policy.order_by", "must be row_identity")
        limit = _required(search_policy, "default_limit", self.scenario_id, "world_schema.search_policy")
        maximum = _required(search_policy, "max_limit", self.scenario_id, "world_schema.search_policy")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            _stop(self.scenario_id, "world_schema.search_policy.default_limit", "positive integer is required")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < limit:
            _stop(self.scenario_id, "world_schema.search_policy.max_limit", "must be at least default_limit")
        if customer_table not in self.world:
            _stop(self.scenario_id, f"world.{customer_table}", "customer table is required")
        entity_columns: dict[str, dict[str, str]] = {}
        for entity, raw_schema in entities.items():
            schema = _mapping(raw_schema, self.scenario_id, f"world_schema.entities.{entity}")
            columns = _mapping(
                _required(schema, "columns", self.scenario_id, f"world_schema.entities.{entity}"),
                self.scenario_id,
                f"world_schema.entities.{entity}.columns",
            )
            for column, value_type in columns.items():
                if not isinstance(column, str) or not column or value_type not in JSON_SCHEMA_TYPES:
                    _stop(self.scenario_id, f"world_schema.entities.{entity}.columns.{column}", f"unsupported type {value_type!r}")
            entity_columns[entity] = columns
        for table, rows in self.world.items():
            if not isinstance(rows, list):
                _stop(self.scenario_id, f"world.{table}", "row list is required")
            for index, row in enumerate(rows):
                item = _mapping(row, self.scenario_id, f"world.{table}[{index}]")
                _mapping(_required(item, "row_identity", self.scenario_id, f"world.{table}[{index}]"), self.scenario_id, f"world.{table}[{index}].row_identity")
                values = _mapping(_required(item, "values", self.scenario_id, f"world.{table}[{index}]"), self.scenario_id, f"world.{table}[{index}].values")
                for column, value in values.items():
                    if table in entity_columns and column not in entity_columns[table]:
                        _stop(self.scenario_id, f"world.{table}[{index}].values.{column}", "column is absent from world schema")
                    if table in entity_columns and not value_matches_type(value, entity_columns[table][column]):
                        _stop(self.scenario_id, f"world.{table}[{index}].values.{column}", f"value must be {entity_columns[table][column]}")
        for tool_id, value in self.tools.items():
            if not isinstance(tool_id, str) or not tool_id:
                _stop(self.scenario_id, "tools", "tool IDs must be non-empty strings")
            tool = _mapping(value, self.scenario_id, f"tools.{tool_id}")
            if _string(tool, "id", self.scenario_id, f"tools.{tool_id}") != tool_id:
                _stop(self.scenario_id, f"tools.{tool_id}.id", "must equal its mapping key")
            operation = _string(tool, "operation", self.scenario_id, f"tools.{tool_id}")
            if operation not in {"filtered_search", "update", "create"}:
                _stop(self.scenario_id, f"tools.{tool_id}.operation", "unsupported operation")
            entity = _string(tool, "entity", self.scenario_id, f"tools.{tool_id}")
            if entity not in self.world:
                _stop(self.scenario_id, f"tools.{tool_id}.entity", "table is absent from world")
            arguments = _required(tool, "arguments", self.scenario_id, f"tools.{tool_id}")
            if not isinstance(arguments, list):
                _stop(self.scenario_id, f"tools.{tool_id}.arguments", "list is required")
            argument_names: set[str] = set()
            for index, argument_value in enumerate(arguments):
                field = f"tools.{tool_id}.arguments[{index}]"
                argument = _mapping(argument_value, self.scenario_id, field)
                name = _string(argument, "name", self.scenario_id, field)
                if name in argument_names:
                    _stop(self.scenario_id, f"{field}.name", "must be unique within the tool")
                argument_names.add(name)
                _string(argument, "type", self.scenario_id, field)
                required = _required(argument, "required", self.scenario_id, field)
                if not isinstance(required, bool):
                    _stop(self.scenario_id, f"{field}.required", "boolean is required")
            filters = _required(tool, "filters", self.scenario_id, f"tools.{tool_id}")
            if not isinstance(filters, list):
                _stop(self.scenario_id, f"tools.{tool_id}.filters", "list is required")
            if entity != customer_table and entity not in entities:
                _stop(self.scenario_id, f"world_schema.entities.{entity}", "entity schema is required")
            columns = entities.get(entity, {}).get("columns", {}) if entity != customer_table else {}
            if entity != customer_table and not isinstance(columns, dict):
                _stop(self.scenario_id, f"world_schema.entities.{entity}.columns", "mapping is required")
            for index, filter_value in enumerate(filters):
                field = f"tools.{tool_id}.filters[{index}]"
                rule = _mapping(filter_value, self.scenario_id, field)
                column = _string(rule, "column", self.scenario_id, field)
                if columns and column not in columns:
                    _stop(self.scenario_id, f"world_schema.entities.{entity}.columns.{column}", "column type is required")
                if set(rule) == {"column", "argument", "operator"}:
                    name = _string(rule, "argument", self.scenario_id, field)
                    if name not in argument_names:
                        _stop(self.scenario_id, f"{field}.argument", "must name a tool argument")
                elif set(rule) != {"column", "value", "operator"}:
                    _stop(self.scenario_id, field, "must contain exactly column, operator, and argument or value")
                elif columns and not _matches_runtime_type(rule["value"], columns[column]):
                    _stop(self.scenario_id, f"{field}.value", f"value must be {columns[column]}")
                if rule["operator"] not in {"eq", "lte", "gte"}:
                    _stop(self.scenario_id, f"{field}.operator", "unsupported operator")
            mutation = tool.get("mutation")
            if operation == "filtered_search":
                if mutation is not None:
                    _stop(self.scenario_id, f"tools.{tool_id}.mutation", "filtered_search must not mutate")
            else:
                self._validate_mutation(tool, argument_names, columns)
            for argument in arguments:
                self._argument_type(tool, argument["name"], argument)
        if self.fault.value is not None:
            for field in ("tool_id", "recovery_tool_id"):
                tool_id = self.fault.value[field]
                if tool_id not in self.tools:
                    _stop(self.scenario_id, f"error_injection.{field}", "does not name a package tool")
            recovery_tool = self.tools[self.fault.value["recovery_tool_id"]]
            recovery_names = {
                item["name"]
                for item in recovery_tool["arguments"]
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            }
            required_recovery_names = {
                item["name"]
                for item in recovery_tool["arguments"]
                if isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and item.get("required") is True
            }
            supplied_recovery_names = set(self.fault.value["recovery_arguments"])
            if supplied_recovery_names - recovery_names:
                _stop(
                    self.scenario_id,
                    "error_injection.recovery_arguments",
                    "names an unknown recovery-tool argument",
                )
            missing_recovery_names = required_recovery_names - supplied_recovery_names
            if missing_recovery_names:
                _stop(
                    self.scenario_id,
                    "error_injection.recovery_arguments",
                    f"missing required recovery-tool arguments {sorted(missing_recovery_names)}",
                )
        categories = _mapping(
            _required(self.sop_catalog, "categories", self.scenario_id, "sop_catalog"),
            self.scenario_id,
            "sop_catalog.categories",
        )
        if not categories:
            _stop(self.scenario_id, "sop_catalog.categories", "at least one category is required")
        for category_id, value in categories.items():
            category = _mapping(value, self.scenario_id, f"sop_catalog.categories.{category_id}")
            _string(category, "title", self.scenario_id, f"sop_catalog.categories.{category_id}")
            sops = _required(category, "sops", self.scenario_id, f"sop_catalog.categories.{category_id}")
            if not isinstance(sops, list) or not sops:
                _stop(self.scenario_id, f"sop_catalog.categories.{category_id}.sops", "non-empty list is required")
            self.categories[category_id] = category
            for sop_index, value in enumerate(sops):
                field = f"sop_catalog.categories.{category_id}.sops[{sop_index}]"
                sop = _mapping(value, self.scenario_id, field)
                sop_id = _string(sop, "id", self.scenario_id, field)
                if sop_id in self.sops:
                    _stop(self.scenario_id, f"{field}.id", "must be unique")
                for name in ("title", "applicability", "completion", "prohibition", "escalation"):
                    _string(sop, name, self.scenario_id, field)
                steps = _required(sop, "steps", self.scenario_id, field)
                if not isinstance(steps, list) or not steps:
                    _stop(self.scenario_id, f"{field}.steps", "non-empty list is required")
                self.sops[sop_id] = sop
                self.sop_categories[sop_id] = category_id
                for step_index, step_value in enumerate(steps):
                    step_field = f"{field}.steps[{step_index}]"
                    step = _mapping(step_value, self.scenario_id, step_field)
                    _string(step, "step_id", self.scenario_id, step_field)
                    kind = _string(step, "kind", self.scenario_id, step_field)
                    if kind == "tool":
                        _string(step, "tool_id", self.scenario_id, step_field)
                    elif kind in {"question", "explain"}:
                        _string(step, "description", self.scenario_id, step_field)
                    else:
                        _stop(self.scenario_id, f"{step_field}.kind", "must be question, explain, or tool")
        for name, value in self.persona.items():
            if name == "baseline_spec" or value is None:
                continue
            item = _mapping(value, self.scenario_id, f"persona.{name}")
            _string(item, "utterance", self.scenario_id, f"persona.{name}")
            _string(item, "fires_when", self.scenario_id, f"persona.{name}")

    def _validate_mutation(
        self,
        tool: dict[str, Any],
        argument_names: set[str],
        columns: dict[str, str],
    ) -> None:
        field = f"tools.{tool['id']}.mutation"
        mutation = tool.get("mutation")
        if not isinstance(mutation, dict) or set(mutation) != {"set"}:
            _stop(self.scenario_id, field, "must be exactly {set: {...}}")
        set_values = mutation["set"]
        if not isinstance(set_values, dict) or not set_values:
            _stop(self.scenario_id, f"{field}.set", "non-empty mapping is required")
        for column, source in set_values.items():
            source_field = f"{field}.set.{column}"
            if column not in columns:
                _stop(self.scenario_id, f"world_schema.entities.{tool['entity']}.columns.{column}", "column type is required")
            if not isinstance(source, dict) or set(source) not in ({"from_argument"}, {"value"}):
                _stop(self.scenario_id, source_field, "must contain exactly from_argument or value")
            if "from_argument" in source and source["from_argument"] not in argument_names:
                _stop(self.scenario_id, f"{source_field}.from_argument", "must name a tool argument")
            if "value" in source and not _matches_runtime_type(source["value"], columns[column]):
                _stop(self.scenario_id, source_field, f"value must be {columns[column]}")

    def operator_system_prompt(self) -> str:
        return (
            "あなたは問い合わせ窓口の担当者です。顧客に必要事項を一度に詰め込みすぎず確認し、"
            f"業務操作の前に {SOP_CATEGORY_SEARCH_TOOL} でカテゴリを検索し、{SOP_SEARCH_TOOL} でカテゴリ内を再検索して、"
            f"{SOP_DETAIL_TOOL} で候補の本文を確認してください。各検索は上位{self.sop_search_top_k}件です。"
            "候補に当てはまるものが見当たらないときは、検索語やカテゴリを変えて何度でも検索し直してかまいません。"
            "ツール結果にない値を作らず、必要な処理後は結果を説明して終話確認をしてください。\n"
            "この対話は電話窓口です。あなたの応答はそのまま読み上げられます。"
            "話し言葉だけで応答し、記号による装飾を使わないでください。"
            "使えないものは、太字や強調の記号(**、__)、見出し記号(#)、"
            "箇条書きの記号(-、*、・で始まる行)、表の罫線(|)、水平線(---)です。"
            "項目を並べるときは「一つ目は〜、二つ目は〜」のように文章で述べてください。"
            "改行で段落を分けず、ひと続きの文章として述べてください。"
            "同じ内容を繰り返さないでください(ただし、次に述べる重要な値の復唱確認は除きます)。"
            "電話番号や予約番号、生年月日、氏名などの重要な値を顧客から聞き取ったときは、"
            "必ずその値を復唱して確認してください。\n"
            f"ドメイン共通規定: {_json_text(self.domain)}"
        )

    def customer_context(self) -> dict[str, Any]:
        return {
            "business": {"initial_user_request": self.initial_user_request},
            "inputs": deepcopy(self.inputs),
            "persona": deepcopy(self.persona),
        }

    def tool_definitions(self) -> list[dict[str, Any]]:
        definitions = [
            self._function_tool(
                SOP_CATEGORY_SEARCH_TOOL,
                "問い合わせ内容に合うSOPカテゴリを検索する",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "問い合わせ内容"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            self._function_tool(
                SOP_SEARCH_TOOL,
                "選んだカテゴリ内で問い合わせ内容に合うSOP候補を検索する",
                {
                    "type": "object",
                    "properties": {
                        "category_id": {"type": "string", "description": "カテゴリ検索結果のカテゴリID"},
                        "query": {"type": "string", "description": "問い合わせ内容"},
                    },
                    "required": ["category_id", "query"],
                    "additionalProperties": False,
                },
            ),
            self._function_tool(
                SOP_DETAIL_TOOL,
                "検索結果のSOP本文を取得する",
                {
                    "type": "object",
                    "properties": {"sop_id": {"type": "string", "description": "検索結果にあるSOP ID"}},
                    "required": ["sop_id"],
                    "additionalProperties": False,
                },
            ),
        ]
        for tool_id, tool in self.tools.items():
            properties: dict[str, Any] = {}
            required_names: list[str] = []
            for spec in tool["arguments"]:
                name = spec["name"]
                value_type = self._argument_type(tool, name, spec)
                property_schema: dict[str, Any] = {
                    "type": JSON_SCHEMA_TYPES[value_type],
                    "description": str(spec.get("description") or name),
                }
                if value_type == "date":
                    property_schema["format"] = "date"
                if isinstance(spec.get("enum"), list):
                    property_schema["enum"] = deepcopy(spec["enum"])
                properties[name] = property_schema
                if spec["required"]:
                    required_names.append(name)
            returns = tool.get("returns")
            description = tool.get("description")
            if not isinstance(description, str) or not description.strip():
                _stop(self.scenario_id, f"tools.{tool_id}.description", "non-empty string is required")
            if not isinstance(returns, str) or not returns.strip():
                _stop(self.scenario_id, f"tools.{tool_id}.returns", "non-empty string is required")
            description = f"{description}。戻り値: {returns}"
            definitions.append(
                self._function_tool(
                    tool_id,
                    description,
                    {
                        "type": "object",
                        "properties": properties,
                        "required": required_names,
                        "additionalProperties": False,
                    },
                )
            )
        return definitions

    @staticmethod
    def _function_tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}

    def _argument_type(self, tool: dict[str, Any], name: str, spec: dict[str, Any]) -> str:
        entity = tool["entity"]
        entities = self.world_schema["entities"]
        columns = entities.get(entity, {}).get("columns", {}) if isinstance(entities.get(entity), dict) else {}
        filter_types = {
            columns[item["column"]]
            for item in tool["filters"]
            if isinstance(item, dict)
            and item.get("argument") == name
            and item.get("column") in columns
        }
        mutation = tool.get("mutation")
        if isinstance(mutation, dict) and isinstance(mutation.get("set"), dict):
            filter_types.update(
                columns[column]
                for column, source in mutation["set"].items()
                if column in columns and source == {"from_argument": name}
            )
        if len(filter_types) > 1:
            _stop(self.scenario_id, f"tools.{tool['id']}.arguments.{name}", "world schema columns disagree on type")
        value_type = next(iter(filter_types), _string(spec, "type", self.scenario_id, f"tools.{tool['id']}.arguments.{name}"))
        if value_type not in JSON_SCHEMA_TYPES:
            _stop(self.scenario_id, f"world_schema.entities.{entity}.columns", f"unsupported type {value_type!r}")
        return value_type

    def argument_provenance(
        self,
        tool_id: str,
        arguments: Any,
        prior_events: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Classify each object argument from the model-visible prior event array."""

        if not isinstance(arguments, dict):
            return {}
        tool = self.tools.get(tool_id)
        specs: dict[str, dict[str, Any]] = {}
        if isinstance(tool, dict):
            specs = {
                spec["name"]: spec
                for spec in tool.get("arguments", [])
                if isinstance(spec, dict) and isinstance(spec.get("name"), str)
            }
        ordered_events = sorted(
            (
                event
                for event in prior_events
                if isinstance(event, dict)
                and isinstance(event.get("seq"), int)
                and not isinstance(event.get("seq"), bool)
            ),
            key=lambda event: event["seq"],
            reverse=True,
        )
        provenance: dict[str, dict[str, Any]] = {}
        for name, value in arguments.items():
            spec = specs.get(name)
            required = name in SOP_TOOL_ARGUMENTS.get(tool_id, set())
            value_type = "string" if required else None
            if spec is not None and isinstance(tool, dict):
                required = spec.get("required") is True
                value_type = self._argument_type(tool, name, spec)
            source = {
                "source_kind": "unknown",
                "source_seq": None,
                "source_ref": None,
                "source_tool": None,
                "source_path": None,
                "required": required,
            }
            for event in ordered_events:
                if event.get("actor") != "tool" or event.get("event_type") != "tool_result":
                    continue
                result = event.get("result")
                if not isinstance(result, dict):
                    continue
                payload = {
                    key: item
                    for key, item in result.items()
                    if key not in PROVENANCE_RESULT_METADATA_FIELDS
                }
                matched = _matching_value_path(
                    payload,
                    value,
                    column=name,
                    value_type=value_type,
                    contains_text=False,
                    path="result",
                )
                if matched is not None:
                    source.update(
                        source_kind="tool_result",
                        source_seq=event["seq"],
                        source_ref=event.get("event_ref"),
                        source_tool=event.get("tool"),
                        source_path=matched,
                    )
                    break
            if source["source_kind"] == "unknown":
                for event in ordered_events:
                    if event.get("actor") != "user" or event.get("event_type") != "message":
                        continue
                    matched = _matching_value_path(
                        event.get("content"),
                        value,
                        column=name,
                        value_type=value_type,
                        contains_text=True,
                        path="content",
                    )
                    if matched is not None:
                        source.update(
                            source_kind="customer_utterance",
                            source_seq=event["seq"],
                            source_ref=event.get("event_ref"),
                            source_path=matched,
                        )
                        break
            if source["source_kind"] == "unknown":
                for event in ordered_events:
                    if event.get("tool") != SOP_DETAIL_TOOL or event.get("event_type") != "tool_result":
                        continue
                    result = event.get("result")
                    sop = result.get("sop") if isinstance(result, dict) else None
                    steps = sop.get("steps") if isinstance(sop, dict) else None
                    if not isinstance(steps, list):
                        continue
                    for index, step in enumerate(steps):
                        description = step.get("description") if isinstance(step, dict) else None
                        matched = _matching_value_path(
                            description,
                            value,
                            column=name,
                            value_type=value_type,
                            contains_text=True,
                            path=f"result.sop.steps[{index}].description",
                        )
                        if matched is None:
                            continue
                        source.update(
                            source_kind="sop_step",
                            source_seq=event["seq"],
                            source_ref=event.get("event_ref"),
                            source_tool=SOP_DETAIL_TOOL,
                            source_path=matched,
                        )
                        break
                    if source["source_kind"] != "unknown":
                        break
            if source["source_kind"] == "unknown":
                matched = _matching_value_path(
                    self.domain,
                    value,
                    column=name,
                    value_type=value_type,
                    contains_text=True,
                    path="domain",
                )
                if matched is not None:
                    source.update(
                        source_kind="domain_policy",
                        source_seq=0,
                        source_ref="operator_system_prompt",
                        source_path=matched,
                    )
            provenance[name] = source
        return provenance

    def call(self, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        before = _json_text(self.world)
        if tool_id not in SOP_TOOL_ARGUMENTS and tool_id not in self.tools:
            result = {"ok": False, "error": "unknown_tool"}
            return self._record(tool_id, arguments, result, before)
        violations = self._argument_violations(tool_id, arguments)
        if violations:
            attempt = self._schema_repair_attempts.get(tool_id, 0) + 1
            self._schema_repair_attempts[tool_id] = attempt
            if attempt <= MAX_SCHEMA_REPAIRS:
                event = {"tool": tool_id, "violations": violations, "attempt": attempt}
                self.schema_repair_events.append(event)
                result = {
                    "ok": False,
                    "error": "schema_repair",
                    "violations": deepcopy(violations),
                    "expected_schema": self._argument_schema_summary(tool_id),
                    "instruction": "修正して同じツールを再試行せよ",
                }
            else:
                result = {"ok": False, "error": "invalid_arguments", "message": "; ".join(violations)}
            return self._record(tool_id, arguments, result, before)
        self._schema_repair_attempts.pop(tool_id, None)
        if tool_id == SOP_CATEGORY_SEARCH_TOOL:
            result = self._search_categories(arguments["query"])
            return self._record(tool_id, arguments, result, before)
        if tool_id == SOP_SEARCH_TOOL:
            result = self._search_sops(arguments["category_id"], arguments["query"])
            return self._record(tool_id, arguments, result, before)
        if tool_id == SOP_DETAIL_TOOL:
            result = self._get_sop(arguments["sop_id"])
            return self._record(tool_id, arguments, result, before)
        tool = self.tools[tool_id]
        fault_result = self.fault.before_call(
            tool_id,
            arguments,
            result_rows=self.result_rows,
            world_snapshot=before,
        )
        if fault_result is not None:
            return self._record(tool_id, arguments, fault_result, before)

        # A model that omits a filter argument made a bad call; tell it, do not end the run.
        missing = [
            rule["argument"]
            for rule in tool["filters"]
            if isinstance(rule, dict) and isinstance(rule.get("argument"), str) and rule["argument"] not in arguments
        ]
        if missing:
            result = {"ok": False, "error": "invalid_arguments", "message": f"missing filter argument: {', '.join(missing)}"}
            return self._record(tool_id, arguments, result, before)
        rows = self._filtered_rows(tool, arguments)
        operation = tool["operation"]
        changes: list[dict[str, Any]] = []
        if operation == "filtered_search":
            if tool["entity"] == self.world_schema["customer_table"] and len(rows) == 1:
                self.verified_customer = str(rows[0]["row_identity"].get("customer") or "") or None
        else:
            bad = self._mutation_argument_problems(tool, arguments)
            if bad:
                result = {"ok": False, "error": "invalid_arguments", "message": "; ".join(bad)}
                return self._record(tool_id, arguments, result, before)
            values = self._mutation_values(tool, arguments)
            if operation == "create":
                created = self._create(tool, values)
                if isinstance(created, dict):
                    return self._record(tool_id, arguments, created, before)
                rows, changes = created
            else:
                if not rows:
                    result = {"ok": False, "error": "no_matching_row", "rows": []}
                    return self._record(tool_id, arguments, result, before)
                for row in rows:
                    old = deepcopy(row["values"])
                    row["values"].update(deepcopy(values))
                    changes.append(
                        {
                            "operation": "update",
                            "entity": tool["entity"],
                            "row_identity": deepcopy(row["row_identity"]),
                            "before": old,
                            "after": deepcopy(row["values"]),
                        }
                    )
        self.result_rows.extend(_public_rows(rows))
        self.fault.after_success(tool_id, arguments)
        result = {"ok": True, "rows": _public_rows(rows), "changes": changes}
        return self._record(tool_id, arguments, result, before)

    def _record(self, tool_id: str, arguments: dict[str, Any], result: dict[str, Any], before: str) -> dict[str, Any]:
        record = {
            "tool_id": tool_id,
            "arguments": deepcopy(arguments),
            "result": deepcopy(result),
            "world_changed": before != _json_text(self.world),
        }
        self.tool_log.append(record)
        return result

    def _argument_schema_summary(self, tool_id: str) -> dict[str, Any]:
        definition = next(item for item in self.tool_definitions() if item["function"]["name"] == tool_id)
        return deepcopy(definition["function"]["parameters"])

    def _argument_violations(self, tool_id: str, arguments: Any) -> list[str]:
        """List violations for an argument object using the existing runtime rules."""

        if not isinstance(arguments, dict):
            return ["arguments: object is required"]
        if tool_id in SOP_TOOL_ARGUMENTS:
            names = SOP_TOOL_ARGUMENTS[tool_id]
            violations = [f"{name}: required field is missing" for name in sorted(names - set(arguments))]
            violations.extend(f"{name}: unknown argument" for name in sorted(set(arguments) - names))
            violations.extend(
                f"{name}: non-empty string is required"
                for name in sorted(names & set(arguments))
                if not isinstance(arguments[name], str) or not arguments[name].strip()
            )
            return violations
        tool = self.tools[tool_id]
        specs = {item["name"]: item for item in tool["arguments"] if isinstance(item, dict) and isinstance(item.get("name"), str)}
        violations = [f"{name}: unknown argument" for name in sorted(set(arguments) - set(specs))]
        violations.extend(
            f"{name}: required field is missing"
            for name, spec in sorted(specs.items())
            if spec.get("required") is True and name not in arguments
        )
        for name in sorted(set(arguments) & set(specs)):
            expected = self._argument_type(tool, name, specs[name])
            if not _matches_runtime_type(arguments[name], expected):
                violations.append(f"{name}: {expected} is required")
            enum = specs[name].get("enum")
            if isinstance(enum, list) and arguments[name] not in enum:
                violations.append(f"{name}: must be one of {enum}")
        return violations

    def _search_categories(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return {"ok": False, "error": "invalid_arguments", "message": "query must be a non-empty string"}
        ranked = _rank_texts(
            query,
            [
                (
                    category_id,
                    " ".join(
                        [category["title"]]
                        + [f"{sop['title']} {sop['applicability']}" for sop in category["sops"]]
                    ),
                )
                for category_id, category in self.categories.items()
            ],
        )[: self.sop_search_top_k]
        self.available_category_ids = {category_id for category_id, _score in ranked}
        return {
            "ok": True,
            "categories": [
                {"category_id": category_id, "title": self.categories[category_id]["title"]}
                for category_id, _score in ranked
            ],
        }

    def _search_sops(self, category_id: str, query: str) -> dict[str, Any]:
        if not isinstance(category_id, str) or category_id not in self.available_category_ids:
            return {"ok": False, "error": "category_not_available"}
        if not isinstance(query, str) or not query.strip():
            return {"ok": False, "error": "invalid_arguments", "message": "query must be a non-empty string"}
        category = self.categories[category_id]
        ranked = _rank_texts(
            query,
            [(sop["id"], f"{sop['title']} {sop['applicability']}") for sop in category["sops"]],
        )[: self.sop_search_top_k]
        self.available_sop_ids.update(sop_id for sop_id, _score in ranked)
        candidates = [
            {
                "sop_id": sop_id,
                "title": self.sops[sop_id]["title"],
                "applicability": self.sops[sop_id]["applicability"],
            }
            for sop_id, _score in ranked
        ]
        return {"ok": True, "category_id": category_id, "candidates": candidates}

    def _get_sop(self, sop_id: str) -> dict[str, Any]:
        if not isinstance(sop_id, str) or not sop_id.strip():
            return {"ok": False, "error": "invalid_arguments", "message": "sop_id must be a non-empty string"}
        if sop_id not in self.available_sop_ids:
            return {"ok": False, "error": "sop_not_available"}
        sop = self.sops.get(sop_id)
        if not isinstance(sop, dict):
            return {"ok": False, "error": "sop_not_found"}
        return {"ok": True, "sop": deepcopy(sop)}

    def _filtered_rows(self, tool: dict[str, Any], arguments: dict[str, Any]) -> list[dict[str, Any]]:
        rows = list(self.world[tool["entity"]])
        columns = self.world_schema["entities"].get(tool["entity"], {}).get("columns", {})
        for rule in tool["filters"]:
            column = rule["column"]
            if columns and column not in columns:
                _stop(self.scenario_id, f"world_schema.entities.{tool['entity']}.columns.{column}", "column type is required")
            if "argument" in rule:
                name = rule["argument"]
                if name not in arguments:
                    _stop(self.scenario_id, f"tools.{tool['id']}.arguments.{name}", "filter argument is required")
                expected = columns.get(column)
                if expected is not None and not _matches_runtime_type(arguments[name], expected):
                    return []
                value = arguments[name]
            else:
                value = rule["value"]
            operator = rule["operator"]
            rows = [
                row
                for row in rows
                if _compare(
                    row["values"].get(column),
                    value,
                    operator,
                    column=column,
                    value_type=columns.get(column),
                )
            ]
        limit = self.world_schema["search_policy"]["default_limit"]
        return sorted(rows, key=_row_sort_key)[:limit] if tool["operation"] == "filtered_search" else rows

    def _mutation_argument_problems(self, tool: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
        """Model-side errors _mutation_values would otherwise stop the run on: a missing or mistyped argument."""

        mutation = tool.get("mutation")
        set_values = mutation.get("set") if isinstance(mutation, dict) else None
        if not isinstance(set_values, dict):
            return []
        columns = self.world_schema["entities"].get(tool["entity"], {}).get("columns", {})
        problems = []
        for column, source in set_values.items():
            name = source.get("from_argument") if isinstance(source, dict) else None
            if not isinstance(name, str):
                continue
            if name not in arguments:
                problems.append(f"{name}: required for this operation")
            elif column in columns and not _matches_runtime_type(arguments[name], columns[column]):
                problems.append(f"{name}: {columns[column]} is required")
        return problems

    def _mutation_values(self, tool: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for column, source in tool["mutation"]["set"].items():
            if "from_argument" in source:
                name = source["from_argument"]
                if name not in arguments:
                    _stop(self.scenario_id, f"tools.{tool['id']}.arguments.{name}", "mutation argument is required")
                values[column] = deepcopy(arguments[name])
            else:
                values[column] = deepcopy(source["value"])
        columns = self.world_schema["entities"][tool["entity"]]["columns"]
        for column, value in values.items():
            if not _matches_runtime_type(value, columns[column]):
                _stop(self.scenario_id, f"tools.{tool['id']}.mutation.{column}", f"value must be {columns[column]}")
            values[column] = normalize_typed_value(value, columns[column])
        return values

    def _create(
        self, tool: dict[str, Any], values: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | dict[str, Any]:
        rows = self.world[tool["entity"]]
        requirements = [] if self.completion is None else self.completion.get("required_mutations")
        if not isinstance(requirements, list):
            _stop(
                self.scenario_id,
                "contracts.completion.required_mutations",
                "list is required",
            )
        matches = [
            item
            for item in requirements
            if isinstance(item, dict)
            and item.get("table") == tool["entity"]
        ]
        if len(matches) > 1:
            matches = [item for item in matches if _values_include(values, item.get("set"))]
        if len(matches) > 1:
            # The arguments do not pick one planned target: the model's call is wrong, the run is not.
            return {"ok": False, "error": "ambiguous_target", "message": "the arguments match more than one target"}
        if matches:
            requirement = matches[0]
            selector = _mapping(
                requirement.get("record"),
                self.scenario_id,
                "contracts.completion.required_mutations.record",
            )
            operation = requirement.get("kind", "update")
            selected = [row for row in rows if _values_include(row["row_identity"], selector)]
            if operation == "update":
                if len(selected) != 1:
                    return {"ok": False, "error": "no_matching_row", "rows": []}
                row = selected[0]
                before = deepcopy(row["values"])
                row["values"].update(deepcopy(values))
                change = {
                    "operation": "update",
                    "entity": tool["entity"],
                    "row_identity": deepcopy(row["row_identity"]),
                    "before": before,
                    "after": deepcopy(row["values"]),
                }
                return [row], [change]
            if operation != "create":
                _stop(
                    self.scenario_id,
                    "contracts.completion.required_mutations.kind",
                    "must be create or update",
                )
            if selected:
                return {
                    "ok": False,
                    "error": "create_target_already_exists",
                    "message": "create target already exists; existing row was not modified",
                }
            row = {"row_identity": deepcopy(selector), "values": deepcopy(values)}
            rows.append(row)
            change = {
                "operation": "create",
                "entity": tool["entity"],
                "row_identity": deepcopy(selector),
                "before": None,
                "after": deepcopy(values),
            }
            return [row], [change]
        if self.verified_customer is None:
            # The model wrote before identifying the customer: a bad call it can recover from, not a broken run.
            return {
                "ok": False,
                "error": "identity_not_verified",
                "message": "verify the customer before this operation",
            }
        # An unplanned create still gets a complete, unique row identity.
        row_identity = {
            "role": "unplanned",
            "role_name": tool["id"],
            "customer": self.verified_customer,
            "sequence": 1 + sum(
                1 for existing in rows if (existing.get("row_identity") or {}).get("role") == "unplanned"
            ),
        }
        row = {"row_identity": row_identity, "values": deepcopy(values)}
        rows.append(row)
        change = {
            "operation": "create",
            "entity": tool["entity"],
            "row_identity": deepcopy(row_identity),
            "before": None,
            "after": deepcopy(row["values"]),
        }
        return [row], [change]

def _compare(
    left: Any,
    right: Any,
    operator: str,
    *,
    column: str,
    value_type: str | None,
) -> bool:
    if operator == "eq":
        return _normalized_match_value(left, column, value_type) == _normalized_match_value(
            right, column, value_type
        )
    if left is None:
        return False
    if operator == "lte":
        return left <= right
    return left >= right


def _matches_runtime_type(value: Any, expected: str) -> bool:
    if expected == "date":
        return isinstance(value, str) and _normalized_date(value) is not None
    return value_matches_type(value, expected)


def _normalized_text(value: str) -> str:
    return "".join(character.lower() for character in unicodedata.normalize("NFKC", value) if character.isalnum())


def _rank_texts(query: str, values: list[tuple[str, str]]) -> list[tuple[str, float]]:
    normalized_query = _normalized_text(query)
    query_bigrams = {normalized_query[index : index + 2] for index in range(max(len(normalized_query) - 1, 0))}
    scored = []
    for item_id, text in values:
        normalized = _normalized_text(text)
        overlap = sum(1 for bigram in query_bigrams if bigram in normalized)
        score = overlap + SequenceMatcher(None, normalized_query, normalized).ratio()
        scored.append((item_id, score))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def _row_sort_key(row: dict[str, Any]) -> str:
    return _json_text(row["row_identity"])


def _values_include(actual: Any, expected: Any) -> bool:
    return isinstance(actual, dict) and isinstance(expected, dict) and all(
        actual.get(key) == value for key, value in expected.items()
    )


def _values_match_for_completion(actual: Any, expected: Any) -> bool:
    """Like ``_values_include``, but with the per-column normalization ordinary tool matching uses."""

    return isinstance(actual, dict) and isinstance(expected, dict) and all(
        _normalized_match_value(actual.get(key), key, None) == _normalized_match_value(value, key, None)
        for key, value in expected.items()
    )


def _successful_world_changes(call_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for event in call_events:
        result = event.get("result")
        if (
            event.get("event_type") != "tool_result"
            or not isinstance(result, dict)
            or result.get("ok") is False
            or result.get("error") is not None
        ):
            continue
        event_changes = result.get("changes")
        if isinstance(event_changes, list):
            changes.extend(change for change in event_changes if isinstance(change, dict))
    return changes


def _mutation_observed(requirement: Any, changes: list[dict[str, Any]]) -> bool:
    if not isinstance(requirement, dict):
        raise ValueError("completion mutation must be an object")
    table = requirement.get("table")
    record = requirement.get("record")
    set_values = requirement.get("set")
    operation = requirement.get("kind", "update")
    if (
        not isinstance(table, str)
        or not isinstance(record, dict)
        or not isinstance(set_values, dict)
        or operation not in {"create", "update"}
    ):
        raise ValueError("completion mutation fields are invalid")
    return any(
        change.get("operation") == operation
        and change.get("entity") == table
        and _values_match_for_completion(change.get("row_identity"), record)
        and _values_match_for_completion(change.get("after"), set_values)
        for change in changes
    )


def _observed_handling_type(changes: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    observed = {
        HANDLING_TYPE_ALIASES[value]
        for change in changes
        for value in ((change.get("after") or {}).get("handling_type"),)
        if value in HANDLING_TYPE_ALIASES
    }
    if not observed:
        return None, "handling_type_not_observed"
    if len(observed) != 1:
        return None, "handling_type_ambiguous"
    return observed.pop(), None


def _run_outcome_from_completion(
    completion: dict[str, Any], call_events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate decoded completion mutations against successful call events."""

    required = completion.get("required_mutations")
    forbidden = completion.get("forbidden_mutations")
    if not isinstance(required, list) or not isinstance(forbidden, list):
        raise ValueError("completion required_mutations and forbidden_mutations must be arrays")
    changes = _successful_world_changes(call_events)
    handling_type, handling_error = _observed_handling_type(changes)
    if any(not _mutation_observed(item, changes) for item in required):
        return build_run_outcome_from_facts(
            completed=False,
            handling_type=handling_type,
            reason="required_mutation_missing",
        )
    if any(_mutation_observed(item, changes) for item in forbidden):
        return build_run_outcome_from_facts(
            completed=False,
            handling_type=handling_type,
            reason="forbidden_mutation_observed",
        )
    if handling_error is not None:
        return build_run_outcome_from_facts(
            completed=False,
            handling_type=None,
            reason=handling_error,
        )
    return build_run_outcome_from_facts(
        completed=True,
        handling_type=handling_type,
        reason="completion_contract_satisfied",
    )


def _operator_payload(
    *, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], extra_body: dict[str, Any], endpoint: str
) -> dict[str, Any]:
    payload = {"model": model, "messages": messages, **extra_body}
    if tools:
        payload["tools"] = tools
    apply_chat_provider_profile(
        payload,
        profile="auto",
        endpoint=endpoint,
        temperature=0.0,
        max_tokens=MAX_COMPLETION_TOKENS,
    )
    return payload


def _active_persona(runtime: PackageRuntime) -> dict[str, dict[str, Any]]:
    return {
        name: deepcopy(event)
        for name, event in runtime.persona.items()
        if name != "baseline_spec" and isinstance(event, dict)
    }


PERSONA_DEADLINE_TOOL_FIELDS = {
    "customer_pressure": "critical_action_tool_id",
    "customer_misconception": "deadline_tool_id",
}
# The only runtime-written tool result: a fact, not a business outcome.
PERSONA_INTERRUPT_TOOL_RESULT = {
    "ok": False,
    "not_executed": True,
    "error": "お客様が話し始めたため、この操作は実行していません。",
}


def _deadline_persona_interrupts(
    runtime: PackageRuntime,
    *,
    variant_id: str,
    fired_persona: set[str],
    requested_tool_ids: list[str],
) -> list[str]:
    """Return unfired hard-variant personas whose deadline tool the operator is about to call.

    ``fires_when`` still decides the natural firing point; this only guarantees
    that pressure and misconception fire at the latest before their deadline tool.
    """

    if variant_id != "hard":
        return []
    interrupts = []
    for name, field in PERSONA_DEADLINE_TOOL_FIELDS.items():
        event = runtime.persona.get(name)
        if not isinstance(event, dict) or name in fired_persona:
            continue
        deadline_tool_id = _string(event, field, runtime.scenario_id, f"persona.{name}")
        if deadline_tool_id in requested_tool_ids:
            interrupts.append(name)
    return interrupts


def _customer_reply(
    *,
    runtime: PackageRuntime,
    endpoint: str,
    model: str,
    headers: dict[str, str],
    timeout_sec: float,
    extra_body: dict[str, Any],
    dialogue: list[dict[str, str]],
    called_tools: list[str],
    variant_id: str,
    fired_persona: set[str],
    end_deferred: bool,
    transport: Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]],
    context_length_retry: list[dict[str, Any]] | None = None,
) -> tuple[str, bool, list[str], dict[str, Any], bool]:
    active_persona = _active_persona(runtime) if variant_id == "hard" else {}
    unfired = {
        name: event for name, event in active_persona.items() if name not in fired_persona
    }
    correction = active_persona.get("customer_correction")
    pending_correction = correction is not None and "customer_correction" not in fired_persona
    incorrect_spoken = False
    slot_name = None
    incorrect_value = None
    if correction is not None:
        field = "persona.customer_correction"
        slot_name = _string(correction, "declared_input_name", runtime.scenario_id, field)
        incorrect_value = _string(correction, "incorrect_value", runtime.scenario_id, field)
        incorrect_spoken = any(
            incorrect_value in turn["content"]
            for turn in dialogue
            if turn.get("role") == "assistant" and isinstance(turn.get("content"), str)
        )
        if pending_correction and not incorrect_spoken:
            unfired.pop("customer_correction")
    customer_context = runtime.customer_context()
    customer_context.pop("persona")
    visible_persona = {
        name: event for name, event in active_persona.items()
        if name != "customer_correction" or not pending_correction or incorrect_spoken
    }
    if visible_persona:
        customer_context["persona"] = visible_persona
    if correction is not None:
        value_name = "corrected_value" if not pending_correction else "incorrect_value"
        value = _required(correction, value_name, runtime.scenario_id, field)
        matching_inputs = [
            item
            for group in ("identity", "declared")
            for item in customer_context["inputs"][group]
            if item["name"] == slot_name
        ]
        if len(matching_inputs) != 1:
            _stop(runtime.scenario_id, f"{field}.declared_input_name", "must identify exactly one input")
        matching_inputs[0]["value"] = deepcopy(value)
        matching_inputs[0].pop("spoken", None)
    instructions = (
        "あなたは問い合わせをした顧客本人です。以下の保持情報だけを使い、担当者の直前発話へ自然な日本語で答えてください。"
        "聞かれた値だけを申告し、値を作らないでください。未発火 persona の fires_when を担当者の直前発話や振る舞いと照合し、"
        "条件を満たした種類名を persona_fired に一度だけ入れてください。persona の utterance は runner が message の前へ付けるため、"
        "message には含めないでください。担当者の終話確認に同意する場合と、担当者が終話のあいさつをして顧客に残りの用件がない場合だけ end_conversation=true にしてください。"
        "JSONオブジェクト {\"message\": string, \"end_conversation\": boolean, \"persona_fired\": string[]} だけを返してください。\n"
        + (
            "未発火の customer_correction に該当する項目を聞かれたら、保持情報の inputs にある値をそのまま答えてください。"
            "訂正の発話より前に corrected_value を答えないでください。\n"
            if variant_id == "hard" and "customer_correction" in unfired
            else ""
        )
        + (
            "担当者の直前発話が空なのは、担当者が何も話さなかったことを意味します。呼びかけるなど自然に応じてください。\n"
            if dialogue and dialogue[-1].get("role") == "user" and not str(dialogue[-1].get("content") or "").strip()
            else ""
        )
        + f"顧客情報: {_json_text(customer_context)}\n"
        f"未発火 persona: {_json_text(unfired)}\n"
        f"これまでに呼ばれたツール: {_json_text(called_tools)}"
    )
    payload = {"model": model, "messages": [{"role": "system", "content": instructions}, *dialogue], **extra_body}
    payload["response_format"] = _customer_response_format(unfired)
    apply_chat_provider_profile(
        payload,
        profile="auto",
        endpoint=endpoint,
        temperature=0.0,
        max_tokens=USER_MAX_COMPLETION_TOKENS,
    )
    response = chat_with_context_retry(
        transport,
        chat_completions_url(endpoint),
        payload,
        headers,
        timeout_sec,
        phase="customer",
        events=context_length_retry if context_length_retry is not None else [],
    )
    message_response = extract_chat_message(response)
    content = message_response.get("content") or message_response.get("refusal")
    if not isinstance(content, str):
        _stop(runtime.scenario_id, "customer_response", "string content is required")
    parsed, fallback_parse = _parse_response_json(content, label="customer response")
    message = parsed.get("message")
    end = parsed.get("end_conversation")
    newly_fired = parsed.get("persona_fired")
    if (
        not isinstance(message, str)
        or not message.strip()
        or not isinstance(end, bool)
        or not isinstance(newly_fired, list)
        or any(not isinstance(name, str) or not name for name in newly_fired)
        or len(newly_fired) != len(set(newly_fired))
    ):
        _stop(
            runtime.scenario_id,
            "customer_response",
            "message string, end_conversation boolean, and unique persona_fired string array are required",
        )
    invalid_fires = set(newly_fired) - set(unfired)
    if invalid_fires:
        _stop(
            runtime.scenario_id,
            "customer_response.persona_fired",
            f"unknown, disabled, or repeated persona types: {sorted(invalid_fires)}",
        )
    message = message.strip()
    customer_record = {"messages": payload["messages"]}
    if end_deferred and pending_correction:
        if incorrect_spoken:
            # The operator has now had a turn after the incorrect declaration.
            if "customer_correction" not in newly_fired:
                newly_fired.append("customer_correction")
        else:
            # An earlier attempt to end the call cannot skip the declaration.
            message = f"{slot_name}は{incorrect_value}です。"
            end = False
    if end and not end_deferred and (unfired or pending_correction):
        newly_fired.extend(name for name in unfired if name not in newly_fired)
        if newly_fired:
            message = "\n".join(active_persona[name]["utterance"] for name in newly_fired)
        end = False
        customer_record["conversation_end_deferred_for_persona"] = True
    elif newly_fired:
        message = "\n".join([*(active_persona[name]["utterance"] for name in newly_fired), message])
    return message, end, newly_fired, customer_record, fallback_parse


def run_package_chat(
    *,
    endpoint: str,
    model: str,
    operator_headers: dict[str, str] | None,
    scenario_id: str,
    scenario_entry: dict[str, Any],
    output_dir: Path,
    timeout_sec: float,
    extra_body: dict[str, Any],
    user_controller_endpoint: str,
    user_controller_model: str,
    user_controller_extra_body: dict[str, Any],
    user_controller_headers: dict[str, str],
    max_tool_rounds: int,
    max_turns: int,
    variant_id: str,
    run_number: int = 1,
    sop_search_top_k: int = DEFAULT_SOP_SEARCH_TOP_K,
    run_id: str | None = None,
    transport: Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]] = post_json,
    partial_record: dict[str, Any] | None = None,
    operator_backend: Any = None,
    operator_uses_responses: bool = False,
    operator_reasoning: dict | None = None,
    operator_prompt_cache: bool = False,
    audio_user_channel: Any = None,
    io_mode: str = "text-text",
) -> list[dict[str, Any]]:
    """Run one decoded task package and return one scenario record.

    ``scenario_entry`` must contain the package mapping under ``scenario`` and
    its source path under ``path``. Invalid package fields stop with
    ``PackageError``; ``variant_id`` accepts ``baseline`` or ``hard``, and
    ``run_number`` must be positive. No aliases or substitute defaults are
    accepted. When supplied, ``partial_record`` is populated only after package
    validation and retains the available run state if any exception propagates.
    """

    started = time.perf_counter()
    try:
        variant_id = normalize_variant(variant_id)
    except ValueError as exc:
        raise PackageError(f"{scenario_id}: runtime.variant_id: {exc}") from exc
    package = _mapping(_required(scenario_entry, "scenario", scenario_id, "scenario_entry"), scenario_id, "scenario_entry.scenario")
    source_path = _required(scenario_entry, "path", scenario_id, "scenario_entry")
    if not isinstance(source_path, (str, Path)):
        _stop(scenario_id, "scenario_entry.path", "path is required")
    if not isinstance(user_controller_model, str) or not user_controller_model.strip():
        _stop(scenario_id, "runtime.user_controller_model", "non-empty string is required")
    if isinstance(run_number, bool) or not isinstance(run_number, int) or run_number < 1:
        _stop(scenario_id, "runtime.run_number", "positive integer is required")
    resolved_run_id = run_id or f"{scenario_id}-{uuid4().hex}"
    if not isinstance(resolved_run_id, str) or not resolved_run_id:
        _stop(scenario_id, "runtime.run_id", "non-empty string is required")
    runtime = PackageRuntime(package, scenario_id=scenario_id)
    runtime.set_sop_search_top_k(sop_search_top_k)
    initial_world = deepcopy(runtime.world)
    operator_messages: list[dict[str, Any]] = [
        {"role": "system", "content": runtime.operator_system_prompt()},
        {"role": "user", "content": runtime.initial_user_request},
    ]
    customer_dialogue: list[dict[str, str]] = [{"role": "assistant", "content": runtime.initial_user_request}]
    events: list[dict[str, Any]] = []
    initial_event_fields: dict[str, Any] = {"content": runtime.initial_user_request}
    if operator_backend is not None:
        if io_mode in {"audio-text", "audio-audio"}:
            if audio_user_channel is None:
                _stop(scenario_id, "runtime.audio_user_channel", "is required for audio input")
            initial_audio = audio_user_channel.convert(runtime.initial_user_request)
            operator_backend.send_user_audio(initial_audio["wav_path"])
            initial_event_fields.update(_audio_event_fields(initial_audio["metadata"]))
        else:
            operator_backend.send_user_text(runtime.initial_user_request)
    initial_event = _event(
        events,
        run_id=resolved_run_id,
        epoch="call",
        actor="user",
        event_type="message",
        role="customer",
        **initial_event_fields,
    )
    conversation = [
        {
            "actor": "customer",
            "content": runtime.initial_user_request,
            "event_id": initial_event["event_id"],
            "event_ref": initial_event["event_ref"],
            "seq": initial_event["seq"],
            "epoch": "call",
        }
    ]
    tool_definitions = runtime.tool_definitions()
    model_contexts: list[dict[str, Any]] = []
    customer_contexts: list[dict[str, Any]] = []
    called_tools: list[str] = []
    fired_persona: set[str] = set()
    correction = runtime.persona.get("customer_correction") if variant_id == "hard" else None
    correction_incorrect_value_spoken = False
    conversation_end_deferred_for_persona = False
    end_conversation = False
    # 上限に達した通話は失敗にせず、その時点で打ち切ってここまでの記録で採点する。
    call_limit: str | None = None
    fallback_parse = False
    tool_rounds = 0
    operator_silence_count = 0
    context_length_retry: list[dict[str, Any]] = []
    operator_usage = empty_chat_usage()
    if operator_backend is not None and isinstance(getattr(operator_backend, "context_length_retry", None), list):
        context_length_retry = operator_backend.context_length_retry
    if partial_record is not None:
        partial_record.update(
            {
                "runtime_schema_version": RUN_SCHEMA_VERSION,
                "run_id": resolved_run_id,
                "scenario_id": scenario_id,
                "variant_id": variant_id,
                "run_number": run_number,
                "source_path": str(source_path),
                "user_controller_model": user_controller_model,
                "initial_world": initial_world,
                "final_world": runtime.world,
                "conversation": conversation,
                "tool_calls": runtime.tool_log,
                "event_log": events,
                "fallback_parse": False,
                "schema_repair_events": runtime.schema_repair_events,
                "schema_repair_count": 0,
                "ticket_repair_count": 0,
                "ticket_repair_events": [],
                "operator_silence_count": 0,
                "context_length_retry": context_length_retry,
                "usage": operator_usage,
            }
        )
        if isinstance(correction, dict):
            partial_record["correction_incorrect_value_spoken"] = False

    for _turn_index in range(max_turns):
        persona_interrupt: list[str] = []
        while True:
            model_contexts.append({"phase": "conversation", "messages": deepcopy(operator_messages), "tools": deepcopy(tool_definitions)})
            payload = _operator_payload(
                model=model,
                messages=operator_messages,
                tools=tool_definitions,
                extra_body=extra_body,
                endpoint=endpoint,
            )
            backend_response = None
            if operator_backend is None:
                operator_call = transport
                if operator_uses_responses:
                    # gpt-5.6 系は reasoning 有効時に chat/completions で
                    # 関数呼び出しを受け付けないため /v1/responses を使う。
                    def operator_call(url, body, hdrs, tmo, _t=transport, _r=operator_reasoning):
                        if _r:
                            body = {**body, "reasoning": _r}
                        return post_chat_via_responses(_t, url, body, hdrs, tmo)
                elif operator_prompt_cache:
                    def operator_call(url, body, hdrs, tmo, _t=transport):
                        return post_chat_with_prompt_cache(
                            _t, url, body, hdrs, tmo, enabled=True
                        )
                response = chat_with_context_retry(
                    operator_call,
                    chat_completions_url(endpoint),
                    payload,
                    dict(operator_headers or {}),
                    timeout_sec,
                    phase="operator",
                    events=context_length_retry,
                )
                accumulate_chat_usage(operator_usage, response)
                message = extract_chat_message(response)
            else:
                backend_response = operator_backend.request_response()
                message = _mapping(
                    backend_response.get("message"),
                    scenario_id,
                    "operator_response.message",
                )
            raw_calls = message.get("tool_calls") or []
            silence_reason = _invalid_tool_calls_reason(raw_calls) if raw_calls else None
            if raw_calls and silence_reason is None:
                persona_interrupt = _deadline_persona_interrupts(
                    runtime,
                    variant_id=variant_id,
                    fired_persona=fired_persona,
                    requested_tool_ids=[raw_call["function"]["name"] for raw_call in raw_calls],
                )
                if persona_interrupt:
                    # The customer interrupts before the deadline tool runs. No call in
                    # this response executes; each gets the same not-executed result on
                    # every transport (text and audio backends alike), then the customer speaks.
                    operator_messages.append(deepcopy(message))
                    for raw_call in raw_calls:
                        operator_messages.append(
                            {"role": "tool", "tool_call_id": raw_call["id"], "content": _json_text(PERSONA_INTERRUPT_TOOL_RESULT)}
                        )
                        if operator_backend is not None:
                            operator_backend.submit_tool_result(raw_call["id"], deepcopy(PERSONA_INTERRUPT_TOOL_RESULT))
                    _event(
                        events,
                        run_id=resolved_run_id,
                        epoch="call",
                        actor="operator",
                        event_type="interrupted_by_persona",
                        persona=list(persona_interrupt),
                        operator_response_raw=deepcopy(message),
                        tool_result=deepcopy(PERSONA_INTERRUPT_TOOL_RESULT),
                    )
                    break
            if raw_calls and silence_reason is None:
                tool_rounds += 1
                if tool_rounds > max_tool_rounds:
                    call_limit = "max_tool_rounds"
                    break
                operator_messages.append(deepcopy(message))
                provenance_events = deepcopy(events)
                for raw_call in raw_calls:
                    function = raw_call["function"]
                    tool_id = function["name"]
                    call_id = raw_call["id"]
                    raw_arguments = function.get("arguments", "{}")
                    try:
                        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    except json.JSONDecodeError as error:
                        arguments = {"_invalid_json": str(error)}
                    argument_provenance = runtime.argument_provenance(
                        tool_id,
                        arguments,
                        provenance_events,
                    )
                    call_event = _event(
                        events,
                        run_id=resolved_run_id,
                        epoch="call",
                        actor="operator",
                        event_type="tool_call",
                        tool=tool_id,
                        arguments=arguments,
                        argument_provenance=argument_provenance,
                        tool_call_id=call_id,
                    )
                    prior_tool_call_count = len(runtime.tool_log)
                    try:
                        result = runtime.call(tool_id, arguments)
                    except Exception:
                        if len(runtime.tool_log) == prior_tool_call_count:
                            runtime.tool_log.append(
                                {
                                    "tool_id": tool_id,
                                    "arguments": deepcopy(arguments),
                                    "argument_provenance": deepcopy(argument_provenance),
                                    "event_id": call_event["event_id"],
                                    "event_ref": call_event["event_ref"],
                                    "seq": call_event["seq"],
                                    "epoch": "call",
                                    "tool_call_id": call_id,
                                }
                            )
                        raise
                    if partial_record is not None:
                        partial_record["schema_repair_count"] = len(runtime.schema_repair_events)
                    result_event = _event(
                        events,
                        run_id=resolved_run_id,
                        epoch="call",
                        actor="tool",
                        event_type="tool_result",
                        tool=tool_id,
                        result=result,
                        tool_call_id=call_id,
                    )
                    runtime.tool_log[-1].update(
                        {
                            "event_id": call_event["event_id"],
                            "event_ref": call_event["event_ref"],
                            "seq": call_event["seq"],
                            "result_event_id": result_event["event_id"],
                            "result_event_ref": result_event["event_ref"],
                            "result_seq": result_event["seq"],
                            "epoch": "call",
                            "tool_call_id": call_id,
                            "argument_provenance": deepcopy(argument_provenance),
                        }
                    )
                    called_tools.append(tool_id)
                    conversation.append(
                        {
                            "actor": "operator",
                            "tool_id": tool_id,
                            "arguments": deepcopy(arguments),
                            "argument_provenance": deepcopy(argument_provenance),
                            "result": deepcopy(result),
                            "event_id": call_event["event_id"],
                            "event_ref": call_event["event_ref"],
                            "seq": call_event["seq"],
                            "result_event_id": result_event["event_id"],
                            "result_event_ref": result_event["event_ref"],
                            "result_seq": result_event["seq"],
                            "epoch": "call",
                        }
                    )
                    operator_messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": _json_text(result)}
                    )
                    if operator_backend is not None:
                        operator_backend.submit_tool_result(call_id, result)
                continue
            content = message.get("content")
            if silence_reason is None and (not isinstance(content, str) or not content.strip()):
                silence_reason = "operator_response.content: non-empty string is required when no tool is called"
            audio_path = backend_response.get("audio_path") if backend_response else None
            if silence_reason is None and io_mode == "audio-audio" and not isinstance(audio_path, Path):
                silence_reason = "operator_response.audio_path: native audio output is required"
            # An invalid operator response is the operator's silence: no tool runs,
            # nothing is substituted, and the customer turn follows as usual.
            operator_text = "" if silence_reason is not None else content.strip()
            operator_messages.append({"role": "assistant", "content": operator_text})
            customer_dialogue.append({"role": "user", "content": operator_text})
            operator_event_fields: dict[str, Any] = {"content": operator_text}
            if silence_reason is not None:
                operator_silence_count += 1
                if partial_record is not None:
                    partial_record["operator_silence_count"] = operator_silence_count
                operator_event_fields["operator_silence"] = {"reason": silence_reason}
                operator_event_fields["operator_silence_raw"] = deepcopy(message)
            elif io_mode == "audio-audio":
                captured = audio_user_channel.capture_operator_audio(
                    source_wav=audio_path,
                    transcript=operator_text,
                    started_ns=backend_response["started_ns"],
                    completed_ns=backend_response["completed_ns"],
                )
                operator_event_fields.update(_audio_event_fields(captured["metadata"]))
            operator_event = _event(
                events,
                run_id=resolved_run_id,
                epoch="call",
                actor="operator",
                event_type="message",
                **operator_event_fields,
            )
            operator_turn = {
                "actor": "operator",
                "content": operator_text,
                "event_id": operator_event["event_id"],
                "event_ref": operator_event["event_ref"],
                "seq": operator_event["seq"],
                "epoch": "call",
            }
            if silence_reason is not None:
                operator_turn["operator_silence"] = {"reason": silence_reason}
            conversation.append(operator_turn)
            break

        if call_limit is not None:
            break
        if persona_interrupt:
            customer_text = "\n".join(
                _string(runtime.persona[name], "utterance", scenario_id, f"persona.{name}")
                for name in persona_interrupt
            )
            end_conversation = False
            newly_fired = list(persona_interrupt)
            customer_context = {"persona_interrupt": list(persona_interrupt)}
            customer_fallback_parse = False
        else:
            customer_text, end_conversation, newly_fired, customer_context, customer_fallback_parse = _customer_reply(
                runtime=runtime,
                endpoint=user_controller_endpoint,
                model=user_controller_model,
                headers=user_controller_headers,
                timeout_sec=timeout_sec,
                extra_body=user_controller_extra_body,
                dialogue=customer_dialogue,
                called_tools=called_tools,
                variant_id=variant_id,
                fired_persona=fired_persona,
                end_deferred=conversation_end_deferred_for_persona,
                transport=transport,
                context_length_retry=context_length_retry,
            )
        fallback_parse = fallback_parse or customer_fallback_parse
        if customer_context.get("conversation_end_deferred_for_persona") is True:
            conversation_end_deferred_for_persona = True
            if partial_record is not None:
                partial_record["conversation_end_deferred_for_persona"] = True
        if isinstance(correction, dict):
            incorrect_value = _string(correction, "incorrect_value", scenario_id, "persona.customer_correction")
            correction_incorrect_value_spoken |= incorrect_value in customer_text
            if partial_record is not None:
                partial_record["correction_incorrect_value_spoken"] = correction_incorrect_value_spoken
        if partial_record is not None:
            partial_record["fallback_parse"] = fallback_parse
        fired_persona.update(newly_fired)
        customer_contexts.append(customer_context)
        operator_messages.append({"role": "user", "content": customer_text})
        customer_dialogue.append({"role": "assistant", "content": customer_text})
        customer_event_fields: dict[str, Any] = {
            "role": "customer",
            "content": customer_text,
        }
        if customer_context.get("conversation_end_deferred_for_persona") is True:
            customer_event_fields["conversation_end_deferred_for_persona"] = True
        if persona_interrupt:
            customer_event_fields["persona_interrupt"] = True
        if operator_backend is not None:
            if io_mode in {"audio-text", "audio-audio"}:
                customer_audio = audio_user_channel.convert(customer_text)
                operator_backend.send_user_audio(customer_audio["wav_path"])
                customer_event_fields.update(_audio_event_fields(customer_audio["metadata"]))
            else:
                operator_backend.send_user_text(customer_text)
        if newly_fired:
            if len(newly_fired) == 1:
                customer_event_fields["persona_fired"] = newly_fired[0]
            else:
                customer_event_fields["persona_fired"] = newly_fired
        customer_event = _event(
            events,
            run_id=resolved_run_id,
            epoch="call",
            actor="user",
            event_type="message",
            **customer_event_fields,
        )
        conversation_event = {
            "actor": "customer",
            "content": customer_text,
            "event_id": customer_event["event_id"],
            "event_ref": customer_event["event_ref"],
            "seq": customer_event["seq"],
            "epoch": "call",
        }
        if newly_fired:
            conversation_event["persona_fired"] = customer_event["persona_fired"]
        conversation.append(conversation_event)
        if end_conversation:
            _event(
                events,
                run_id=resolved_run_id,
                epoch="call",
                actor="user",
                event_type="conversation_end_candidate",
                content=customer_text,
            )
            break
    if not end_conversation and call_limit is None:
        call_limit = "max_turns"

    call_events = deepcopy(events)
    operator_ticket_artifact = None
    ticket_record = partial_record if partial_record is not None else {}
    if runtime.ticket_contract is not None:
        operator_ticket_artifact, ticket_payload, ticket_fallback_parse = _generate_operator_ticket(
            scenario_id=scenario_id,
            endpoint=endpoint,
            model=model,
            headers=dict(operator_headers or {}),
            timeout_sec=timeout_sec,
            extra_body=extra_body,
            conversation=conversation,
            call_events=call_events,
            ticket_contract=runtime.ticket_contract,
            transport=operator_backend.chat_transport if operator_backend is not None else transport,
            partial_record=ticket_record,
            context_length_retry=context_length_retry,
        )
        fallback_parse = fallback_parse or ticket_fallback_parse
        if partial_record is not None:
            partial_record["fallback_parse"] = fallback_parse
        model_contexts.append(
            {"phase": "post_call_ticket", "messages": deepcopy(ticket_payload["messages"]), "tools": []}
        )
        _event(
            events,
            run_id=resolved_run_id,
            epoch="post_call",
            actor="operator",
            event_type="post_session_ticket",
            result=operator_ticket_artifact,
        )
        if partial_record is not None:
            partial_record["operator_ticket_artifact"] = operator_ticket_artifact
            partial_record["schema_repair_count"] = len(runtime.schema_repair_events)

    record = {
        "runtime_schema_version": RUN_SCHEMA_VERSION,
        "run_id": resolved_run_id,
        "scenario_id": scenario_id,
        "variant_id": variant_id,
        "run_number": run_number,
        "status": "success",
        "fallback_parse": fallback_parse,
        "run_outcome": (
            _run_outcome_from_completion(runtime.completion, call_events)
            if runtime.completion is not None
            else None
        ),
        "source_path": str(source_path),
        "user_controller_model": user_controller_model,
        "initial_world": initial_world,
        "final_world": deepcopy(runtime.world),
        "operator_ticket_artifact": operator_ticket_artifact,
        "conversation": conversation,
        "tool_calls": deepcopy(runtime.tool_log),
        "schema_repair_events": deepcopy(runtime.schema_repair_events),
        "schema_repair_count": len(runtime.schema_repair_events),
        "ticket_repair_count": ticket_record.get("ticket_repair_count", 0),
        "ticket_repair_events": deepcopy(ticket_record.get("ticket_repair_events", [])),
        "operator_silence_count": operator_silence_count,
        "context_length_retry": deepcopy(context_length_retry),
        "usage": deepcopy(operator_usage),
        "event_log": events,
        "wall_time_sec": round(time.perf_counter() - started, 6),
    }
    if isinstance(correction, dict):
        record["correction_incorrect_value_spoken"] = correction_incorrect_value_spoken
    if conversation_end_deferred_for_persona:
        record["conversation_end_deferred_for_persona"] = True
    if call_limit is not None:
        record["call_limit_reached"] = call_limit
    if "operator_ticket_failure" in ticket_record:
        record["operator_ticket_failure"] = deepcopy(ticket_record["operator_ticket_failure"])
    scenario_output = run_output_dir(
        output_dir,
        scenario_id=scenario_id,
        variant_id=variant_id,
        run_number=run_number,
    )
    scenario_output.mkdir(parents=True, exist_ok=True)
    (scenario_output / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (scenario_output / "event_log.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    (scenario_output / "model_context.json").write_text(json.dumps(model_contexts, ensure_ascii=False, indent=2), encoding="utf-8")
    (scenario_output / "customer_context.json").write_text(json.dumps(customer_contexts, ensure_ascii=False, indent=2), encoding="utf-8")
    return [record]


def _audio_event_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    """Expose measured audio metadata in the shape consumed by the evidence builder."""

    segment = metadata["audio_metric_segment"]
    return {
        "metadata": metadata,
        "audio_segment_id": segment["audio_segment_id"],
        "clock_id": segment["clock_id"],
        "monotonic_ns": segment["monotonic_range_ns"][1],
    }


def run_output_dir(
    out_dir: Path,
    *,
    scenario_id: str,
    variant_id: str,
    run_number: int,
) -> Path:
    return out_dir / _safe_component(scenario_id) / _safe_component(variant_id) / f"run_{run_number:03d}"


def _safe_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe output path component: {value!r}")
    return value
