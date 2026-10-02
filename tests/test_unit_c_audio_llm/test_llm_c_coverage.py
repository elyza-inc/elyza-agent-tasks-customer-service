"""Focused no-network checks for untested LLM conversion and validation paths."""

from __future__ import annotations


import pytest

from elyza_agent_tasks_customer_service.evaluation.llm import llm_io


def _catalog() -> dict:
    return {
        "action_codes": ["act"], "id_types": ["order"], "queue_ids": ["queue"],
        "reason_codes": ["reason"], "refusal_reason_codes": ["refuse"],
        "diagnostic_conclusion_codes": ["diagnosis"], "escalation_response_codes": ["ack"],
        "code_descriptions": {key: key for key in ("act", "order", "queue", "reason", "refuse", "diagnosis", "ack")},
    }


def test_llm_io_native_provider_conversions_cover_grouped_tools() -> None:
    payload = {
        "model": "m", "temperature": .1, "top_p": .9, "seed": 1, "stop": ["END"],
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "assistant", "content": "call", "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": '{"x": 1}'}}]},
            {"role": "tool", "tool_call_id": "c", "content": {"ok": True}},
            {"role": "tool", "tool_call_id": "c", "content": {"again": True}},
        ],
        "tools": [{"function": {"name": "f", "parameters": {"type": "object"}}}],
    }
    responses = llm_io.chat_payload_to_responses(payload)
    assert responses["input"][2]["type"] == "function_call_output"
    anthropic = llm_io.chat_payload_to_anthropic(payload)
    assert anthropic["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    vertex = llm_io.chat_payload_to_vertex(payload)
    assert len(vertex["contents"][-1]["parts"]) == 2
    assert vertex["tools"][0]["functionDeclarations"][0]["parameters"]["type"] == "OBJECT"
    assert llm_io.vertex_to_chat_response({"candidates": []}) == {"choices": [], "usage": {}}
    assert llm_io.anthropic_to_chat_response({"content": [{"type": "text", "text": "ok"}], "stop_reason": "max_tokens"})["choices"][0]["finish_reason"] == "length"
    with pytest.raises(ValueError, match="matching"):
        llm_io.chat_payload_to_vertex({"messages": [{"role": "tool", "tool_call_id": "none", "content": "{}"}]})


