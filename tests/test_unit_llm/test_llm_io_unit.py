"""Unit checks for evaluator LLM transport and provider conversions."""

from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from elyza_agent_tasks_customer_service.evaluation.llm import llm_io


def test_context_retry_reduces_copy_and_records_event() -> None:
    payload = {"max_completion_tokens": 900}
    calls: list[dict[str, int]] = []

    def transport(_url, request, _headers, _timeout):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError(
                "HTTP 400: maximum context length is 1000 tokens; prompt contains at least 650 input tokens"
            )
        return {"ok": True}

    events: list[dict[str, object]] = []
    assert llm_io.chat_with_context_retry(transport, "url", payload, {}, 1, phase="p", events=events) == {"ok": True}
    assert payload["max_completion_tokens"] == 900
    assert calls[1]["max_completion_tokens"] == 286
    assert events == [{"phase": "p", "token_field": "max_completion_tokens", "old_value": 900, "new_value": 286, "retried": True}]


def test_context_retry_refuses_too_small_or_unrelated_errors() -> None:
    events: list[dict[str, object]] = []
    error = RuntimeError("HTTP 400 maximum context length is 1000 tokens; prompt contains at least 800 input tokens")
    with pytest.raises(RuntimeError, match="HTTP 400"):
        llm_io.chat_with_context_retry(lambda *_: (_ for _ in ()).throw(error), "url", {"max_tokens": 500}, {}, 1, phase="p", events=events)
    assert events[0]["reason"] == "recalculated output tokens below 256"
    assert llm_io.context_length_limits(RuntimeError("HTTP 500 nope")) is None


@pytest.mark.parametrize(
    ("profile", "endpoint", "expected"),
    [
        ("auto", "https://api.openai.com", "openai"),
        ("auto", "https://api.anthropic.com", "anthropic"),
        ("auto", "http://localhost:8000", "vllm"),
    ],
)
def test_provider_profile_resolution_and_payload_fields(profile, endpoint, expected) -> None:
    payload = {"model": "gpt-5.4-mini-2026-03-17"}
    assert llm_io.apply_chat_provider_profile(payload, profile=profile, endpoint=endpoint, temperature=0.2, max_tokens=9) == expected
    if expected == "openai":
        assert payload == {"model": payload["model"], "max_completion_tokens": 9, "temperature": 0.2}
    elif expected == "anthropic":
        assert payload["max_tokens"] == 9 and "temperature" not in payload
    else:
        assert payload["max_tokens"] == 9 and payload["temperature"] == 0.2
    with pytest.raises(ValueError, match="unknown provider"):
        llm_io.resolve_provider_profile("bad", endpoint="x")


def test_post_json_retries_http_and_formats_terminal_errors(monkeypatch) -> None:
    failures = [
        HTTPError("https://x", 429, "busy", {}, BytesIO(b"slow")),
        HTTPError("https://x", 429, "busy", {}, BytesIO(b"slow")),
        HTTPError("https://x", 429, "busy", {}, BytesIO(b"slow")),
    ]
    monkeypatch.setattr(llm_io.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(failures.pop(0)))
    monkeypatch.setattr(llm_io.time, "sleep", lambda _: None)
    with pytest.raises(llm_io.TransientTransportError, match="slow"):
        llm_io.post_json("https://x", {"tools": []}, {}, 1)

    error = HTTPError("https://x", 400, "bad", {}, BytesIO("壊れた".encode()))
    monkeypatch.setattr(llm_io.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(RuntimeError, match="HTTP 400.*壊れた"):
        llm_io.post_json("https://x", {}, {}, 1)


def test_post_json_retries_url_errors(monkeypatch) -> None:
    attempts = [URLError("offline"), URLError("offline"), URLError("offline")]
    monkeypatch.setattr(llm_io.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(attempts.pop(0)))
    monkeypatch.setattr(llm_io.time, "sleep", lambda _: None)
    with pytest.raises(llm_io.TransientTransportError, match="offline"):
        llm_io.post_json("https://x", {}, {}, 1)


def test_post_json_retries_malformed_function_calls(monkeypatch) -> None:
    """Resend malformed tool-call responses and retain the final invalid response."""

    responses = iter(
        [
            b'{"choices":[{"finish_reason":"malformed_function_call"}]}',
            b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}',
        ]
    )

    class Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return self.body

    monkeypatch.setattr(llm_io.request, "urlopen", lambda *_args, **_kwargs: Response(next(responses)))
    monkeypatch.setattr(llm_io.time, "sleep", lambda _delay: None)

    assert llm_io.post_json("https://x", {}, {}, 1)["choices"][0]["message"]["content"] == "ok"


def test_post_json_raises_after_three_malformed_function_calls(monkeypatch) -> None:
    """Keep the current missing-message error once the retry budget is exhausted."""

    attempts = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b'{"choices":[{"finish_reason":"malformed_function_call"}]}'

    monkeypatch.setattr(llm_io.request, "urlopen", lambda *_args, **_kwargs: attempts.append(1) or Response())
    monkeypatch.setattr(llm_io.time, "sleep", lambda _delay: None)

    with pytest.raises(ValueError, match="missing message"):
        llm_io.extract_chat_message(llm_io.post_json("https://x", {}, {}, 1))
    assert len(attempts) == 3


def test_openai_responses_conversion_and_usage() -> None:
    payload = {"model": "m", "messages": [{"role": "system", "content": "s"}, {"role": "assistant", "content": "ok", "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "c", "content": {"ok": True}}], "tools": [{"function": {"name": "f", "parameters": {}}}], "max_completion_tokens": 12}
    converted = llm_io.chat_payload_to_responses(payload)
    assert converted["instructions"] == "s" and converted["max_output_tokens"] == 12
    response = llm_io.responses_to_chat_response({"output": [{"type": "message", "content": [{"text": "hello"}]}, {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}], "incomplete_details": {"reason": "max_output_tokens"}})
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["choices"][0]["message"]["tool_calls"][0]["id"] == "c"
    total = llm_io.empty_chat_usage()
    llm_io.accumulate_chat_usage(total, {"usage": {"prompt_tokens": 2, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 1}}})
    assert total == {"input_tokens": 2, "output_tokens": 3, "cache_creation_input_tokens": None, "cache_read_input_tokens": 1}


def test_anthropic_and_vertex_conversion_round_trips() -> None:
    payload = {"model": "m", "max_completion_tokens": 8, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": '{"x":1}'}}]}, {"role": "tool", "tool_call_id": "c", "content": '{"done":true}'}], "tools": [{"function": {"name": "f", "parameters": {"type": "object"}}}]}
    anthropic = llm_io.chat_payload_to_anthropic(payload)
    assert anthropic["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert llm_io.anthropic_to_chat_response({"content": [{"type": "tool_use", "id": "c", "name": "f", "input": {"x": 1}}], "stop_reason": "tool_use"})["choices"][0]["finish_reason"] == "tool_calls"
    vertex = llm_io.chat_payload_to_vertex(payload)
    assert vertex["tools"][0]["functionDeclarations"][0]["parameters"]["type"] == "OBJECT"
    response = llm_io.vertex_to_chat_response({"responseId": "r", "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}, {"functionCall": {"name": "f", "args": {"x": 1}}}]}}]})
    assert response["choices"][0]["message"]["tool_calls"][0]["id"] == "r_1"


def test_prompt_cache_native_routes_and_vertex_url() -> None:
    seen = []
    def transport(url, payload, headers, timeout):
        seen.append((url, payload, headers))
        return {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    result = llm_io.post_chat_with_prompt_cache(transport, "https://api.anthropic.com/v1/chat/completions", {"model": "m", "max_tokens": 2, "messages": [{"role": "user", "content": "u"}]}, {"Authorization": "Bearer secret"}, 1, enabled=True)
    assert result["choices"][0]["message"]["content"] == "ok"
    assert seen[0][0] == "https://api.anthropic.com/v1/messages" and seen[0][2]["x-api-key"] == "secret"
    assert llm_io._vertex_generate_content_url("https://x/projects/p/locations/us/endpoints/openapi/chat/completions", "google/m") == "https://x/v1/projects/p/locations/global/publishers/google/models/m:generateContent"


def test_post_json_uses_exact_three_attempts(monkeypatch) -> None:
    attempts = []
    monkeypatch.setattr(llm_io.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(llm_io.request, "urlopen", lambda *_args, **_kwargs: attempts.append(1) or (_ for _ in ()).throw(URLError("offline")))
    with pytest.raises(llm_io.TransientTransportError):
        llm_io.post_json("https://x", {}, {}, 1)
    assert len(attempts) == 3


def test_post_json_resends_cloudflare_5xx_but_not_4xx(monkeypatch) -> None:
    """HTTP 520 (seen from the customer-role call) is resent like 500; 400 fails at once."""

    import io
    import json as _json
    from urllib.error import HTTPError

    from elyza_agent_tasks_customer_service.evaluation.llm import llm_io

    monkeypatch.setattr(llm_io.time, "sleep", lambda seconds: None)
    replies = []

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout):
        code = replies.pop(0)
        if code == 200:
            return _Response(_json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())
        raise HTTPError(req.full_url, code, "err", {}, io.BytesIO(b"error code: %d" % code))

    monkeypatch.setattr(llm_io.request, "urlopen", fake_urlopen)
    replies[:] = [520, 200]
    assert llm_io.post_json("http://x", {}, {}, 1)["choices"][0]["message"]["content"] == "ok"
    replies[:] = [400]
    try:
        llm_io.post_json("http://x", {}, {}, 1)
    except RuntimeError as error:
        assert "HTTP 400" in str(error)
    else:
        raise AssertionError("HTTP 400 was resent")
    assert llm_io.is_transient_http_code(524) and not llm_io.is_transient_http_code(404)


def test_responses_reply_with_repeated_message_is_not_doubled() -> None:
    """A commentary + final answer pair (or two identical messages) yields the text once."""

    message = lambda text, phase=None: {"type": "message", **({"phase": phase} if phase else {}), "content": [{"type": "output_text", "text": text}]}
    same = llm_io.responses_to_chat_response({"output": [message("確認します。"), message("確認します。")]})
    assert same["choices"][0]["message"]["content"] == "確認します。"
    phased = llm_io.responses_to_chat_response({"output": [message("途中です。", "commentary"), message("確認しました。", "final_answer")]})
    assert phased["choices"][0]["message"]["content"] == "確認しました。"
    two = llm_io.responses_to_chat_response({"output": [message("一文目。"), message("二文目。")]})
    assert two["choices"][0]["message"]["content"] == "一文目。\n二文目。"
