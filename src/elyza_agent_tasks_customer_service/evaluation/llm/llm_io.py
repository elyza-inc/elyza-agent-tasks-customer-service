"""HTTP and JSON helpers for evaluator-side LLM calls."""

from __future__ import annotations

import http.client
from copy import deepcopy
import json
import re
import time
from typing import Any, Callable
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit


TRANSIENT_HTTP_CODES = {408, 409, 429}


def is_transient_http_code(code: int) -> bool:
    """Return whether an HTTP status is worth resending: 408/409/429 and every 5xx (Cloudflare 520-524 included)."""

    return code in TRANSIENT_HTTP_CODES or 500 <= code < 600


PROVIDER_PROFILES = frozenset(("auto", "openai", "vllm", "anthropic"))
OPENAI_MODELS_SUPPORTING_TEMPERATURE = frozenset(
    ("gpt-5.4-mini-2026-03-17",)
)
CONTEXT_LENGTH_RESERVE_TOKENS = 64
MIN_CONTEXT_RETRY_OUTPUT_TOKENS = 256
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_MESSAGES_PATH = "/v1/messages"
ANTHROPIC_EPHEMERAL_CACHE_CONTROL = {"type": "ephemeral"}
VERTEX_API_VERSION = "v1"
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
CONTEXT_LENGTH_PATTERN = re.compile(
    r"maximum context length is (?P<maximum>\d+) tokens.*?prompt contains at least (?P<input>\d+) input tokens",
    re.IGNORECASE | re.DOTALL,
)


class TransientTransportError(RuntimeError):
    """Raised after retryable HTTP transport attempts are exhausted."""


def context_length_limits(error: BaseException) -> tuple[int, int] | None:
    """Extract ``(model maximum, input tokens)`` from a context-length HTTP 400.

    Only the provider message containing both documented token counts is
    accepted. Other HTTP errors and incomplete messages return ``None``.
    """

    message = str(error)
    if "HTTP 400" not in message:
        return None
    matched = CONTEXT_LENGTH_PATTERN.search(message)
    if matched is None:
        return None
    return int(matched["maximum"]), int(matched["input"])


def chat_with_context_retry(
    transport: Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]],
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: float,
    *,
    phase: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Send one chat payload, shrinking its output allowance once on context HTTP 400."""

    try:
        return transport(url, payload, headers, timeout_sec)
    except RuntimeError as exc:
        limits = context_length_limits(exc)
        token_field = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
        if limits is None or token_field not in payload:
            raise
        old_value = payload[token_field]
        if isinstance(old_value, bool) or not isinstance(old_value, int):
            raise
        maximum, input_tokens = limits
        new_value = min(old_value, maximum - input_tokens - CONTEXT_LENGTH_RESERVE_TOKENS)
        event = {
            "phase": phase,
            "token_field": token_field,
            "old_value": old_value,
            "new_value": new_value,
            "retried": new_value >= MIN_CONTEXT_RETRY_OUTPUT_TOKENS,
        }
        if not event["retried"]:
            event["reason"] = f"recalculated output tokens below {MIN_CONTEXT_RETRY_OUTPUT_TOKENS}"
            events.append(event)
            raise
        retry_payload = deepcopy(payload)
        retry_payload[token_field] = new_value
        events.append(event)
        return transport(url, retry_payload, headers, timeout_sec)


def resolve_provider_profile(profile: str, *, endpoint: str) -> str:
    """Resolve an explicit provider profile, or infer one from ``endpoint`` for ``auto``."""

    if profile not in PROVIDER_PROFILES:
        raise ValueError(f"unknown provider profile: {profile}")
    if profile != "auto":
        return profile
    lowered = endpoint.lower()
    if "api.openai.com" in lowered:
        return "openai"
    if "api.anthropic.com" in lowered:
        return "anthropic"
    return "vllm"


def apply_chat_provider_profile(
    payload: dict[str, Any],
    *,
    profile: str,
    endpoint: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Add provider-specific token and sampling fields to a chat payload."""

    resolved = resolve_provider_profile(profile, endpoint=endpoint)
    if resolved == "openai":
        payload["max_completion_tokens"] = max_tokens
        # A conservative allowlist avoids a second billable request after a
        # capability-probing 400. Unknown OpenAI models retain the API default
        # until their temperature support has been verified and added here.
        if payload.get("model") in OPENAI_MODELS_SUPPORTING_TEMPERATURE:
            payload["temperature"] = temperature
    elif resolved == "anthropic":
        # claude-opus-5 などは temperature を受け付けない。
        payload["max_tokens"] = max_tokens
    else:
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return resolved


def chat_completions_url(endpoint: str) -> str:
    """chat/completions の URL を組み立てる。

    Vertex AI の OpenAI 互換エンドポイントは末尾が ``/endpoints/openapi`` で、
    その配下は ``/chat/completions`` になる。``/v1`` を挟まない。
    """

    base = endpoint.rstrip("/")
    if base.endswith("/endpoints/openapi"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_sec: float) -> dict[str, Any]:
    if "tools" in payload and payload["tools"] in ([], None):
        payload = {key: value for key, value in payload.items() if key != "tools"}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: BaseException | None = None
    last_error_body = ""
    for attempt in range(1, 4):
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_sec) as resp:
                response = json.loads(resp.read().decode("utf-8"))
            choices = response.get("choices") if isinstance(response, dict) else None
            choice = choices[0] if isinstance(choices, list) and choices else None
            if (
                isinstance(choice, dict)
                and choice.get("finish_reason") == "malformed_function_call"
                and not isinstance(choice.get("message"), dict)
                and attempt < 3
            ):
                time.sleep(min(2.0 * attempt, 5.0))
                continue
            return response
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if is_transient_http_code(exc.code):
                last_error = exc
                last_error_body = error_body
                if attempt == 3:
                    break
                time.sleep(min(2.0 * attempt, 5.0))
                continue
            raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc
        except (URLError, http.client.IncompleteRead, ConnectionError) as exc:
            # A connection dropped mid-response is as transient as a refused one.
            last_error = exc
            if attempt == 3:
                break
            time.sleep(min(2.0 * attempt, 5.0))
    body_suffix = f": {last_error_body}" if last_error_body else ""
    raise TransientTransportError(
        f"POST {url} failed after 3 attempts: {last_error}{body_suffix}"
    ) from last_error


def extract_chat_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat tool response missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError(f"chat tool response missing message: {choices[0]}")
    return message


# --- /v1/responses 経路 -------------------------------------------------
# gpt-5.6 系は reasoning が有効な状態だと /v1/chat/completions で関数呼び出しを
# 受け付けない。chat/completions の要求と応答を responses 形式へ相互変換して、
# 後段は従来どおり message 形式で扱えるようにする。

def chat_payload_to_responses(payload: dict[str, Any]) -> dict[str, Any]:
    """chat/completions の要求を responses の要求へ変換する。"""

    out: dict[str, Any] = {"model": payload["model"]}
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in payload.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if isinstance(content, str):
                instructions.append(content)
            continue
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id"),
                "output": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
            })
            continue
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": call.get("id"),
                    "name": function.get("name"),
                    "arguments": function.get("arguments") or "{}",
                })
            if isinstance(content, str) and content.strip():
                items.append({"role": "assistant",
                              "content": [{"type": "output_text", "text": content}]})
            continue
        if isinstance(content, str):
            items.append({"role": role or "user",
                          "content": [{"type": "input_text", "text": content}]})
    if instructions:
        out["instructions"] = "\n\n".join(instructions)
    out["input"] = items
    tools = payload.get("tools")
    if tools:
        out["tools"] = [{"type": "function", **(t.get("function") or {})} for t in tools]
    for key in ("temperature", "max_output_tokens", "reasoning"):
        if key in payload:
            out[key] = payload[key]
    if "max_completion_tokens" in payload:
        out["max_output_tokens"] = payload["max_completion_tokens"]
    return out


def responses_to_chat_response(response: dict[str, Any]) -> dict[str, Any]:
    """responses の応答を chat/completions の応答形へ変換する。"""

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    output = [item for item in response.get("output") or [] if isinstance(item, dict)]
    # A reply may carry a commentary message and a final answer with the same text;
    # joining both doubled the operator's utterance (X\nX). Keep the final answer only.
    if any(item.get("type") == "message" and item.get("phase") == "final_answer" for item in output):
        output = [item for item in output if item.get("type") != "message" or item.get("phase") == "final_answer"]
    for item in output:
        if item.get("type") == "function_call":
            tool_calls.append({
                "id": item.get("call_id"),
                "type": "function",
                "function": {"name": item.get("name"),
                             "arguments": item.get("arguments") or "{}"},
            })
        elif item.get("type") == "message":
            for content_item in item.get("content") or []:
                text = content_item.get("text") if isinstance(content_item, dict) else None
                if isinstance(text, str) and text.strip() and (not text_parts or text_parts[-1].strip() != text.strip()):
                    text_parts.append(text)
    message: dict[str, Any] = {"role": "assistant",
                               "content": "\n".join(text_parts) if text_parts else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = "tool_calls" if tool_calls else "stop"
    if (response.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
        finish = "length"
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": response.get("usage") or {}}


def post_chat_via_responses(
    transport: Any,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: float,
) -> dict[str, Any]:
    """chat/completions 相当の呼び出しを /v1/responses で行う。"""

    responses_url = url.replace("/v1/chat/completions", "/v1/responses")
    raw = transport(responses_url, chat_payload_to_responses(payload), headers, timeout_sec)
    return responses_to_chat_response(raw)


def _json_object(value: Any, *, label: str) -> dict[str, Any]:
    """Return a JSON object from a mapping or encoded object string.

    Arrays, scalars, and malformed JSON raise ``ValueError`` because native
    tool calls and tool responses require object roots.
    """

    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return decoded


def chat_payload_to_anthropic(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert one chat/completions request to Anthropic Messages format."""

    out: dict[str, Any] = {
        "model": payload["model"],
        "max_tokens": payload.get("max_tokens", payload.get("max_completion_tokens")),
        "messages": [],
    }
    for message in payload.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            out["system"] = content
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id"),
                "content": content,
            }
            if (
                out["messages"]
                and out["messages"][-1]["role"] == "user"
                and isinstance(out["messages"][-1]["content"], list)
            ):
                out["messages"][-1]["content"].append(block)
            else:
                out["messages"].append({"role": "user", "content": [block]})
            continue
        if role == "assistant":
            native_content = message.get("_anthropic_content")
            if isinstance(native_content, list):
                out["messages"].append({"role": "assistant", "content": deepcopy(native_content)})
                continue
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id"),
                        "name": function.get("name"),
                        "input": _json_object(function.get("arguments") or "{}", label="tool arguments"),
                    }
                )
            out["messages"].append({"role": "assistant", "content": blocks})
            continue
        out["messages"].append({"role": role or "user", "content": content})
    tools = []
    for tool in payload.get("tools") or []:
        function = tool.get("function") or {}
        native_tool = {
            "name": function.get("name"),
            "description": function.get("description", ""),
            "input_schema": deepcopy(function.get("parameters") or {}),
        }
        tools.append(native_tool)
    if tools:
        tools[-1]["cache_control"] = ANTHROPIC_EPHEMERAL_CACHE_CONTROL
        out["tools"] = tools
    if "system" in out:
        system = out["system"]
        if isinstance(system, str):
            system = [{"type": "text", "text": system}]
        else:
            system = deepcopy(system)
        if isinstance(system, list) and system:
            system[-1]["cache_control"] = ANTHROPIC_EPHEMERAL_CACHE_CONTROL
        out["system"] = system
    if out["messages"]:
        last_message = out["messages"][-1]
        content = last_message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        else:
            content = deepcopy(content)
        if isinstance(content, list) and content:
            content[-1]["cache_control"] = ANTHROPIC_EPHEMERAL_CACHE_CONTROL
        last_message["content"] = content
    if "temperature" in payload:
        out["temperature"] = payload["temperature"]
    if "stop" in payload:
        out["stop_sequences"] = payload["stop"]
    return out


def anthropic_to_chat_response(response: dict[str, Any]) -> dict[str, Any]:
    """Convert one Anthropic Messages response to chat/completions format."""

    content = response.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
        "_anthropic_content": deepcopy(content),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish_reason = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(response.get("stop_reason"), response.get("stop_reason"))
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": response.get("usage") or {},
    }


def _vertex_schema(value: Any) -> Any:
    """Copy a JSON schema while uppercasing values of every ``type`` field."""

    if isinstance(value, dict):
        return {
            key: item.upper() if key == "type" and isinstance(item, str) else _vertex_schema(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_vertex_schema(item) for item in value]
    return value


def chat_payload_to_vertex(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert one chat/completions request to Vertex generateContent format."""

    out: dict[str, Any] = {"contents": []}
    call_names: dict[str, str] = {}
    system_parts: list[dict[str, Any]] = []
    for message in payload.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            system_parts.append({"text": content})
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            name = call_names.get(call_id)
            if not name:
                raise ValueError(f"tool response has no matching function call: {call_id}")
            block = {
                "functionResponse": {
                    "name": name,
                    "response": _json_object(content, label="tool response"),
                }
            }
            if (
                out["contents"]
                and out["contents"][-1]["role"] == "user"
                and all(
                    "functionResponse" in part
                    for part in out["contents"][-1]["parts"]
                )
            ):
                out["contents"][-1]["parts"].append(block)
            else:
                out["contents"].append({"role": "user", "parts": [block]})
            continue
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                call_names[call.get("id")] = function.get("name")
            native_content = message.get("_vertex_content")
            if isinstance(native_content, dict):
                out["contents"].append(deepcopy(native_content))
                continue
            parts: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                parts.append({"text": content})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                parts.append(
                    {
                        "functionCall": {
                            "name": function.get("name"),
                            "args": _json_object(function.get("arguments") or "{}", label="tool arguments"),
                        }
                    }
                )
            out["contents"].append({"role": "model", "parts": parts})
            continue
        out["contents"].append({"role": "user", "parts": [{"text": content}]})
    if system_parts:
        out["systemInstruction"] = {"parts": system_parts}
    declarations = []
    for tool in payload.get("tools") or []:
        function = tool.get("function") or {}
        declarations.append(
            {
                "name": function.get("name"),
                "description": function.get("description", ""),
                "parameters": _vertex_schema(function.get("parameters") or {}),
            }
        )
    if declarations:
        out["tools"] = [{"functionDeclarations": declarations}]
    generation_config = {}
    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "topP"),
        ("seed", "seed"),
        ("max_tokens", "maxOutputTokens"),
        ("max_completion_tokens", "maxOutputTokens"),
    ):
        if source in payload:
            generation_config[target] = payload[source]
    if "stop" in payload:
        generation_config["stopSequences"] = payload["stop"]
    if generation_config:
        out["generationConfig"] = generation_config
    return out


def vertex_to_chat_response(response: dict[str, Any]) -> dict[str, Any]:
    """Convert one Vertex generateContent response to chat/completions format."""

    candidates = response.get("candidates") or []
    if not candidates:
        return {"choices": [], "usage": response.get("usageMetadata") or {}}
    candidate = candidates[0]
    native_content = candidate.get("content") or {}
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    response_id = response.get("responseId") or "vertex"
    for index, part in enumerate(native_content.get("parts") or []):
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        function_call = part.get("functionCall")
        if isinstance(function_call, dict):
            tool_calls.append(
                {
                    "id": function_call.get("id") or f"{response_id}_{index}",
                    "type": "function",
                    "function": {
                        "name": function_call.get("name"),
                        "arguments": json.dumps(function_call.get("args") or {}, ensure_ascii=False),
                    },
                }
            )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
        "_vertex_content": deepcopy(native_content),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if tool_calls:
        finish_reason = "tool_calls"
    else:
        finish_reason = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
        }.get(candidate.get("finishReason"), candidate.get("finishReason"))
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": response.get("usageMetadata") or {},
    }


def _vertex_generate_content_url(chat_url: str, model: str) -> str:
    """Build a global Vertex v1 URL from an OpenAPI endpoint URL.

    ``chat_url`` must contain ``/projects/<project>/locations/<location>/``;
    malformed paths raise ``ValueError``.
    """

    parsed = urlsplit(chat_url)
    segments = parsed.path.strip("/").split("/")
    try:
        project = segments[segments.index("projects") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Vertex endpoint is missing a project: {chat_url}") from exc
    model_id = model.rsplit("/", 1)[-1]
    return (
        f"{parsed.scheme}://{parsed.netloc}/{VERTEX_API_VERSION}/projects/{project}"
        f"/locations/global/publishers/google/models/{model_id}:generateContent"
    )


def post_chat_with_prompt_cache(
    transport: Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]],
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: float,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Send chat unchanged, or use a provider-native cache-capable endpoint."""

    if not enabled:
        return transport(url, payload, headers, timeout_sec)
    lowered = url.lower()
    if "api.anthropic.com" in lowered:
        parsed = urlsplit(url)
        native_url = f"{parsed.scheme}://{parsed.netloc}{ANTHROPIC_MESSAGES_PATH}"
        native_headers = {key: value for key, value in headers.items() if key.lower() != "authorization"}
        authorization = next((value for key, value in headers.items() if key.lower() == "authorization"), "")
        if authorization.startswith("Bearer "):
            native_headers["x-api-key"] = authorization.removeprefix("Bearer ")
        native_headers["anthropic-version"] = ANTHROPIC_API_VERSION
        raw = transport(native_url, chat_payload_to_anthropic(payload), native_headers, timeout_sec)
        return anthropic_to_chat_response(raw)
    if "aiplatform.googleapis.com" in lowered:
        native_url = _vertex_generate_content_url(url, payload["model"])
        raw = transport(native_url, chat_payload_to_vertex(payload), headers, timeout_sec)
        return vertex_to_chat_response(raw)
    return transport(url, payload, headers, timeout_sec)


def empty_chat_usage() -> dict[str, int | None]:
    """Return the normalized token counters stored in one run record."""

    return {field: None for field in USAGE_FIELDS}


def accumulate_chat_usage(total: dict[str, int | None], response: dict[str, Any]) -> None:
    """Add OpenAI, Anthropic, Responses, or Vertex usage fields to ``total``."""

    usage = response.get("usage") or {}
    input_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    values = {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", usage.get("promptTokenCount"))),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", usage.get("candidatesTokenCount"))),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get(
            "cache_read_input_tokens",
            input_details.get("cached_tokens", usage.get("cachedContentTokenCount")),
        ),
    }
    for field, value in values.items():
        if isinstance(value, int) and not isinstance(value, bool):
            total[field] = (total[field] or 0) + value
