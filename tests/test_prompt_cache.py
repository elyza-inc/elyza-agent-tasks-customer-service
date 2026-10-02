"""Provider-native prompt-cache conversion and usage checks."""

from copy import deepcopy
import json

from elyza_agent_tasks_customer_service.evaluation.llm.llm_io import (
    accumulate_chat_usage,
    chat_payload_to_anthropic,
    chat_payload_to_vertex,
    empty_chat_usage,
    post_json,
    post_chat_with_prompt_cache,
)


CHAT_PAYLOAD = {
    "model": "provider/model",
    "messages": [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up one value",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "temperature": 0.0,
    "max_tokens": 128,
}


def test_post_json_omits_only_empty_tools(monkeypatch):
    bodies = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"choices": []}'

    def urlopen(req, *, timeout):
        bodies.append(json.loads(req.data.decode("utf-8")))
        return Response()

    monkeypatch.setattr(
        "elyza_agent_tasks_customer_service.evaluation.llm.llm_io.request.urlopen",
        urlopen,
    )
    for tools in ([], None, CHAT_PAYLOAD["tools"]):
        post_json(
            "https://example.test/v1/chat/completions",
            {"model": "test", "tools": tools},
            {},
            30,
        )

    assert "tools" not in bodies[0]
    assert "tools" not in bodies[1]
    assert bodies[2]["tools"] == CHAT_PAYLOAD["tools"]


def test_disabled_prompt_cache_preserves_openai_request():
    calls = []
    expected = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def transport(url, payload, headers, timeout_sec):
        calls.append((url, payload, headers, timeout_sec))
        return expected

    payload = deepcopy(CHAT_PAYLOAD)
    headers = {"Authorization": "Bearer key"}
    actual = post_chat_with_prompt_cache(
        transport,
        "https://api.openai.com/v1/chat/completions",
        payload,
        headers,
        30,
        enabled=False,
    )

    assert actual is expected
    assert calls == [("https://api.openai.com/v1/chat/completions", payload, headers, 30)]
    assert calls[0][1] is payload
    assert calls[0][2] is headers
    assert "cache_control" not in json.dumps(calls[0][1])


def test_anthropic_request_and_response_conversion():
    payload = chat_payload_to_anthropic(CHAT_PAYLOAD)

    assert payload["system"] == [
        {
            "type": "text",
            "text": "system text",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "user text",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    assert payload["tools"][0]["input_schema"]["type"] == "object"
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}

    calls = []

    def transport(url, body, headers, timeout_sec):
        calls.append((url, body, headers, timeout_sec))
        return {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {"query": "x"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 2,
                "output_tokens": 1,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 0,
            },
        }

    response = post_chat_with_prompt_cache(
        transport,
        "https://api.anthropic.com/v1/chat/completions",
        CHAT_PAYLOAD,
        {"Authorization": "Bearer anthropic-key"},
        30,
        enabled=True,
    )

    assert calls[0][0] == "https://api.anthropic.com/v1/messages"
    assert calls[0][2] == {
        "x-api-key": "anthropic-key",
        "anthropic-version": "2023-06-01",
    }
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup"


def test_anthropic_cache_marks_final_tool_result_block():
    payload = chat_payload_to_anthropic(
        {
            **CHAT_PAYLOAD,
            "messages": [
                *CHAT_PAYLOAD["messages"],
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"value": 1}'},
            ],
        }
    )

    assert payload["messages"][-1]["role"] == "user"
    assert payload["messages"][-1]["content"][-1] == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": '{"value": 1}',
        "cache_control": {"type": "ephemeral"},
    }


def test_vertex_request_response_and_thought_signature_round_trip():
    vertex_payload = {**CHAT_PAYLOAD, "model": "google/gemini-test"}
    payload = chat_payload_to_vertex(vertex_payload)

    declaration = payload["tools"][0]["functionDeclarations"][0]
    assert payload["systemInstruction"] == {"parts": [{"text": "system text"}]}
    assert declaration["parameters"]["type"] == "OBJECT"
    assert declaration["parameters"]["properties"]["query"]["type"] == "STRING"

    native_content = {
        "role": "model",
        "parts": [
            {
                "functionCall": {"name": "lookup", "args": {"query": "x"}},
                "thoughtSignature": "signature",
            }
        ],
    }
    calls = []

    def transport(url, body, headers, timeout_sec):
        calls.append((url, body, headers, timeout_sec))
        return {
            "responseId": "response",
            "candidates": [{"content": native_content, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 1,
                "cachedContentTokenCount": 50,
            },
        }

    response = post_chat_with_prompt_cache(
        transport,
        "https://aiplatform.googleapis.com/v1beta1/projects/project-id/locations/global/endpoints/openapi/chat/completions",
        vertex_payload,
        {"Authorization": "Bearer token"},
        30,
        enabled=True,
    )
    assert calls[0][0] == (
        "https://aiplatform.googleapis.com/v1/projects/project-id/locations/global/"
        "publishers/google/models/gemini-test:generateContent"
    )
    assert calls[0][1] == payload
    message = response["choices"][0]["message"]
    follow_up = chat_payload_to_vertex(
        {
            **CHAT_PAYLOAD,
            "messages": [
                *CHAT_PAYLOAD["messages"],
                message,
                {
                    "role": "tool",
                    "tool_call_id": message["tool_calls"][0]["id"],
                    "content": '{"value": 1}',
                },
            ],
        }
    )

    assert follow_up["contents"][-2] == native_content
    assert follow_up["contents"][-1]["parts"][0]["functionResponse"] == {
        "name": "lookup",
        "response": {"value": 1},
    }


def test_usage_normalization_and_nulls():
    total = empty_chat_usage()
    accumulate_chat_usage(
        total,
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 8},
            }
        },
    )
    accumulate_chat_usage(
        total,
        {
            "usage": {
                "input_tokens": 3,
                "output_tokens": 1,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 0,
            }
        },
    )
    accumulate_chat_usage(
        total,
        {
            "usage": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 1,
                "cachedContentTokenCount": 30,
            }
        },
    )

    assert total == {
        "input_tokens": 17,
        "output_tokens": 4,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 38,
    }
    unavailable = empty_chat_usage()
    accumulate_chat_usage(unavailable, {"usage": {}})
    assert unavailable == {field: None for field in unavailable}
