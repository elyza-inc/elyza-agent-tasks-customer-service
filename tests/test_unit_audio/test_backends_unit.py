"""Backend payload and event interpretation tests with no network transport."""

from __future__ import annotations

import base64
from pathlib import Path

from elyza_agent_tasks_customer_service.evaluation.audio import gemini_live_operator_backend as gemini
from elyza_agent_tasks_customer_service.evaluation.audio import omni_operator_backend as omni
from elyza_agent_tasks_customer_service.evaluation.audio import realtime_operator_backend as realtime


def test_realtime_url_json_extraction_and_transport_payload() -> None:
    """Realtime maps endpoints and extracts a strict structured response from stub events."""

    backend = object.__new__(realtime.RealtimeOperatorBackend)
    backend.model = "model"
    sent: list[dict] = []
    events = iter([{"type": "response.function_call_arguments.done", "name": realtime.STRUCTURED_OUTPUT_TOOL_NAME, "arguments": 'prefix {"ok": true}'}, {"type": "response.done"}])
    backend._send = sent.append
    backend._recv = lambda: next(events)
    result = backend.chat_transport("unused", {"messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}], "response_format": {"json_schema": {"schema": {"type": "object"}}}}, {}, 1)

    assert realtime.RealtimeOperatorBackend._url("https://host/v1", "a b").startswith("wss://host/v1/realtime?model=a+b")
    assert result["choices"][0]["message"]["content"] == '{"ok": true}'
    assert sent[0]["response"]["tool_choice"] == "required"


def test_gemini_schema_and_omni_response_extraction() -> None:
    """Provider-specific schemas and split Omni choices normalize deterministically."""

    schema = gemini._json_schema_to_gemini({"type": "object", "properties": {"x": {"type": "array", "items": {"type": "string"}}}, "$schema": "ignored"})
    message, audio = omni._extract_omni_response({"choices": [{"message": {"content": "first"}}, {"message": {"content": "second", "audio": {"data": base64.b64encode(b"wav").decode()}}}]})

    assert schema == {"type": "OBJECT", "properties": {"x": {"type": "ARRAY", "items": {"type": "STRING"}}}}
    assert message["content"] == "first\nsecond"
    assert audio == b"wav"


def test_omni_request_uses_stubbed_transport(tmp_path: Path, monkeypatch) -> None:
    """Omni payload keeps history and makes its request through the injected client."""

    backend = omni.OmniOperatorBackend(endpoint="http://endpoint", model="m", api_key="key", instructions="rules", tools=[], out_dir=tmp_path, temperature=0, seed=3, max_tokens=10, provider_profile="openai", output_modality="text")
    captured: dict[str, object] = {}
    monkeypatch.setattr(omni, "apply_chat_provider_profile", lambda payload, **kwargs: captured.update(payload=payload))
    monkeypatch.setattr(omni, "chat_with_context_retry", lambda *args, **kwargs: {"model": "m2", "choices": [{"message": {"content": "done"}}]})

    result = backend.request_response(force_instruction="return JSON")

    assert result["message"]["content"] == "done"
    assert captured["payload"]["tool_choice"] == "none"


def test_realtime_and_gemini_interpret_stubbed_turn_events(tmp_path: Path) -> None:
    """Persistent backends translate provider events without opening a socket."""

    realtime_backend = object.__new__(realtime.RealtimeOperatorBackend)
    realtime_backend.output_modality = "audio"
    realtime_backend.out_dir = tmp_path
    realtime_backend._turn = 0
    realtime_sent: list[dict] = []
    realtime_events = iter([
        {"type": "response.created"},
        {"type": "response.audio.delta", "delta": base64.b64encode(b"\0\0").decode()},
        {"type": "response.audio_transcript.delta", "delta": "確認"},
        {"type": "response.function_call_arguments.done", "call_id": "c1", "name": "lookup", "arguments": "{}"},
        {"type": "response.done"},
    ])
    realtime_backend._send = realtime_sent.append
    realtime_backend._recv = lambda: next(realtime_events)
    realtime_result = realtime_backend.request_response(force_instruction="go")
    assert realtime_result["message"]["content"] == "確認"
    assert realtime_result["message"]["tool_calls"][0]["id"] == "c1"
    assert realtime_result["audio_path"].is_file()

    gemini_backend = object.__new__(gemini.GeminiLiveOperatorBackend)
    gemini_backend.output_modality = "text"
    gemini_backend.out_dir = tmp_path
    gemini_backend._turn = 0
    gemini_backend._audio_pending = False
    gemini_backend._pending_calls = {}
    gemini_sent: list[dict] = []
    gemini_events = iter([{"serverContent": {"modelTurn": {"parts": [{"text": "完了"}]}, "turnComplete": True}}])
    gemini_backend._send = gemini_sent.append
    gemini_backend._recv = lambda: next(gemini_events)
    gemini_backend._drain = lambda: None
    gemini_result = gemini_backend.request_response(force_instruction="go")
    assert gemini_result["message"]["content"] == "完了"
    assert gemini_sent[0]["clientContent"]["turnComplete"] is True
