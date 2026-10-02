"""Persistent OpenAI Realtime (websocket) operator session for audio runs."""

from __future__ import annotations

import base64
import json
import tempfile
import time
import urllib.parse
import wave
from pathlib import Path
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.audio.omni_audio_common import (
    ffmpeg_to_mono_pcm16,
    write_mono_pcm16_wav,
)


STRUCTURED_OUTPUT_TOOL_NAME = "submit_structured_output"
PCM_RATE = 24000


class RealtimeOperatorBackend:
    """Small adapter around the existing websocket realtime protocol."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str,
        instructions: str,
        tools: list[dict[str, Any]],
        out_dir: Path,
        temperature: float,
        output_modality: str = "audio",
        voice: str = "alloy",
    ) -> None:
        if output_modality not in {"text", "audio"}:
            raise ValueError("output_modality must be text or audio")
        try:
            from websockets.sync.client import connect as _ws_connect
        except ImportError as exc:
            raise RuntimeError(
                "realtime backend requires the websockets package already used by realtime runners"
            ) from exc
        self.model = model
        self.out_dir = out_dir / "realtime_operator"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._turn = 0
        self.output_modality = output_modality
        self._ws = _ws_connect(
            self._url(endpoint, model),
            additional_headers={
                "Authorization": f"Bearer {api_key}",
            },
            open_timeout=900,
            close_timeout=30,
            max_size=None,
        )
        realtime_tools = []
        for definition in tools:
            function = definition.get("function") if isinstance(definition, dict) else None
            if not isinstance(function, dict):
                continue
            realtime_tools.append(
                {
                    "type": "function",
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        session = {
            "type": "realtime",
            "instructions": instructions,
            "output_modalities": [output_modality],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": PCM_RATE},
                    "turn_detection": None,
                },
                "output": {"format": {"type": "audio/pcm", "rate": PCM_RATE}, "voice": voice},
            },
            "tools": realtime_tools,
            "tool_choice": "auto",
        }
        self._send({"type": "session.update", "session": session})
        self._wait_for({"session.updated"})

    @staticmethod
    def _url(endpoint: str, model: str) -> str:
        base = endpoint.rstrip("/")
        if base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        elif base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        if "/realtime" not in base:
            if base.endswith("/v1"):
                base = base[:-3]
            base += "/v1/realtime"
        separator = "&" if "?" in base else "?"
        return base + separator + urllib.parse.urlencode({"model": model})

    def _send(self, payload: dict[str, Any]) -> None:
        self._ws.send(json.dumps(payload, ensure_ascii=False))

    def _recv(self) -> dict[str, Any]:
        payload = json.loads(self._ws.recv())
        if payload.get("type") == "error":
            raise RuntimeError(f"realtime API error: {payload.get('error')}")
        return payload

    def _wait_for(self, event_types: set[str]) -> dict[str, Any]:
        while True:
            event = self._recv()
            if event.get("type") in event_types:
                return event

    def send_user_audio(self, wav_path: Path) -> None:
        with wave.open(str(wav_path), "rb") as probe:
            source_rate = probe.getframerate()
        send_path = wav_path
        if source_rate != PCM_RATE:
            resampled = Path(tempfile.mkstemp(suffix=".wav", dir=str(self.out_dir))[1])
            ffmpeg_to_mono_pcm16(wav_path, resampled, sample_rate=PCM_RATE)
            send_path = resampled
        with wave.open(str(send_path), "rb") as source:
            if source.getsampwidth() != 2 or source.getnchannels() != 1:
                raise ValueError(f"realtime input must be mono PCM16 WAV: {wav_path}")
            frames = source.readframes(source.getnframes())
        if send_path != wav_path:
            send_path.unlink(missing_ok=True)
        self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frames).decode("ascii"),
            }
        )
        self._send({"type": "input_audio_buffer.commit"})

    def send_user_text(self, text: str) -> None:
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    def submit_tool_result(self, call_id: str, result: Any) -> None:
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                },
            }
        )

    def request_response(self, *, force_instruction: str | None = None) -> dict[str, Any]:
        if force_instruction:
            self._send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": force_instruction}],
                    },
                }
            )
        self._send(
            {
                "type": "response.create",
                "response": {"output_modalities": [self.output_modality]},
            }
        )
        started_ns: int | None = None
        completed_ns = time.monotonic_ns()
        transcript_parts: list[str] = []
        text_parts: list[str] = []
        audio = bytearray()
        calls: dict[str, dict[str, Any]] = {}
        while True:
            event = self._recv()
            event_type = str(event.get("type") or "")
            if event_type in {"response.created", "response.output_item.added"} and started_ns is None:
                started_ns = time.monotonic_ns()
            if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                delta = event.get("delta")
                if isinstance(delta, str):
                    audio.extend(base64.b64decode(delta))
            elif event_type in {
                "response.audio_transcript.delta",
                "response.output_audio_transcript.delta",
            }:
                transcript_parts.append(str(event.get("delta") or ""))
            elif event_type in {"response.text.delta", "response.output_text.delta"}:
                text_parts.append(str(event.get("delta") or ""))
            elif event_type == "response.function_call_arguments.done":
                call_id = str(event.get("call_id") or event.get("item_id") or "")
                calls[call_id] = {
                    "call_id": call_id,
                    "name": str(event.get("name") or ""),
                    "arguments": str(event.get("arguments") or "{}"),
                }
            if event_type != "response.done":
                continue
            completed_ns = time.monotonic_ns()
            response = event.get("response")
            for item in response.get("output", []) if isinstance(response, dict) else []:
                if not isinstance(item, dict) or item.get("type") != "function_call":
                    continue
                call_id = str(item.get("call_id") or item.get("id") or "")
                calls[call_id] = {
                    "call_id": call_id,
                    "name": str(item.get("name") or ""),
                    "arguments": str(item.get("arguments") or "{}"),
                }
            break
        transcript = "".join(transcript_parts).strip() or "".join(text_parts).strip()
        audio_path = None
        if audio:
            self._turn += 1
            audio_path = self.out_dir / f"response-{self._turn:04d}.wav"
            write_mono_pcm16_wav(audio_path, bytes(audio), sample_rate=PCM_RATE)
        tool_calls = [
            {
                "id": call["call_id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in calls.values()
        ]
        return {
            "message": {"role": "assistant", "content": transcript, "tool_calls": tool_calls},
            "audio_path": audio_path,
            "started_ns": started_ns or completed_ns,
            "completed_ns": completed_ns,
        }

    def chat_transport(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_sec: float,
    ) -> dict[str, Any]:
        """Continue post-call JSON/text work in-session for realtime-only models."""

        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("realtime post-call adapter requires messages")
        response_format = payload.get("response_format")
        if not isinstance(response_format, dict):
            raise ValueError("realtime post-call adapter requires response_format")
        json_schema = response_format.get("json_schema")
        schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
        if not isinstance(schema, dict):
            raise ValueError("realtime post-call adapter requires a JSON schema")
        instructions = "\n".join(
            str(row.get("content") or "")
            for row in messages
            if isinstance(row, dict) and row.get("role") == "system"
        )
        instructions += (
            f"\nCall {STRUCTURED_OUTPUT_TOOL_NAME} exactly once. Its arguments are the entire "
            "response. Do not return assistant text. Use this strict response format: "
            + json.dumps(response_format, ensure_ascii=False, sort_keys=True)
        )
        response_input = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": str(row.get("content") or "")}],
            }
            for row in messages
            if isinstance(row, dict) and row.get("role") == "user"
        ]
        self._send(
            {
                "type": "response.create",
                "response": {
                    "conversation": "none",
                    "input": response_input,
                    "instructions": instructions,
                    "output_modalities": ["text"],
                    "tools": [
                        {
                            "type": "function",
                            "name": STRUCTURED_OUTPUT_TOOL_NAME,
                            "description": "Submit the complete structured response.",
                            "parameters": schema,
                        }
                    ],
                    "tool_choice": "required",
                },
            }
        )
        text_parts: list[str] = []
        tool_arguments: str | None = None
        while True:
            event = self._recv()
            if event.get("type") in {"response.text.delta", "response.output_text.delta"}:
                text_parts.append(str(event.get("delta") or ""))
            if (
                event.get("type") == "response.function_call_arguments.done"
                and event.get("name") == STRUCTURED_OUTPUT_TOOL_NAME
            ):
                tool_arguments = str(event.get("arguments") or "")
            if event.get("type") == "response.done":
                response = event.get("response")
                for item in response.get("output", []) if isinstance(response, dict) else []:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "function_call"
                        and item.get("name") == STRUCTURED_OUTPUT_TOOL_NAME
                    ):
                        tool_arguments = str(item.get("arguments") or "")
                break
        content = tool_arguments or "".join(text_parts)
        extracted = _extract_json_block(content)
        return {
            "model": self.model,
            "choices": [{"message": {"role": "assistant", "content": extracted or content}}],
        }

    def close(self) -> None:
        self._ws.close()

def _extract_json_block(text: str) -> str | None:
    """Return the first balanced top-level JSON object/array embedded in text."""

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(text)):
                ch = text[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : index + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)
    return None
