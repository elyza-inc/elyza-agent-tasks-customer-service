"""Persistent vLLM-Omni chat-completions operator session for audio runs."""

from __future__ import annotations

import base64
import binascii
import json
import time
import wave
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.llm.llm_io import apply_chat_provider_profile, chat_with_context_retry, post_json


CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 900
OUTPUT_MODALITIES = ("text", "audio")


class OmniOperatorBackend:
    """Adapt vLLM-Omni chat completions to the realtime backend boundary."""

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
        seed: int,
        max_tokens: int,
        provider_profile: str,
        output_modality: str = "audio",
    ) -> None:
        if output_modality not in OUTPUT_MODALITIES:
            raise ValueError(f"output_modality must be one of {list(OUTPUT_MODALITIES)}")
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self._tools = deepcopy(tools)
        self._temperature = temperature
        self._seed = seed
        self._max_tokens = max_tokens
        self._provider_profile = provider_profile
        self.output_modality = output_modality
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions}
        ]
        self._pending_audio: dict[str, Any] | None = None
        self._request_round = 0
        self.context_length_retry: list[dict[str, Any]] = []
        self._turn = 0
        self.out_dir = out_dir / "omni_operator"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def send_user_audio(self, wav_path: Path) -> None:
        """Queue one non-empty RIFF/WAVE file as an ``input_audio`` part.

        The next ``request_response`` consumes the queued WAV. A second WAV
        before that request, a missing file, or malformed/empty WAV raises an
        error instead of replacing audio evidence.
        """

        if self._pending_audio is not None:
            raise RuntimeError("omni backend already has unconsumed user audio")
        try:
            with wave.open(str(wav_path), "rb") as source:
                if source.getnframes() <= 0:
                    raise ValueError(f"omni input WAV is empty: {wav_path}")
        except (OSError, EOFError, wave.Error) as exc:
            raise ValueError(f"omni input must be a readable RIFF/WAVE file: {wav_path}") from exc
        self._pending_audio = {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(wav_path.read_bytes()).decode("ascii"),
                "format": "wav",
            },
        }

    def submit_tool_result(self, call_id: str, result: Any) -> None:
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    def request_response(self, *, force_instruction: str | None = None) -> dict[str, Any]:
        if self._pending_audio is not None:
            self._messages.append(
                {"role": "user", "content": [self._pending_audio]}
            )
            self._pending_audio = None
        if force_instruction:
            self._messages.append({"role": "user", "content": force_instruction})

        # text: ["text"]; audio: ["text", "audio"]
        output_modalities = list(OUTPUT_MODALITIES) if self.output_modality == "audio" else ["text"]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": deepcopy(self._messages),
            "tools": deepcopy(self._tools),
            "tool_choice": "none" if force_instruction else "auto",
            "modalities": output_modalities,
            "seed": self._seed + self._request_round,
        }
        self._request_round += 1
        apply_chat_provider_profile(
            payload,
            profile=self._provider_profile,
            endpoint=self.endpoint,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        started_ns = time.monotonic_ns()
        response = chat_with_context_retry(
            post_json,
            f"{self.endpoint.rstrip('/')}{CHAT_COMPLETIONS_PATH}",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
            REQUEST_TIMEOUT_SECONDS,
            phase="operator",
            events=self.context_length_retry,
        )
        completed_ns = time.monotonic_ns()
        message, audio = _extract_omni_response(response)
        content = str(message.get("content") or "").strip()
        if self.output_modality == "audio" and content and audio is None:
            raise RuntimeError(
                "vLLM-Omni returned operator text without native audio; "
                "audio metrics cannot be measured"
            )
        audio_path = self._persist_audio(audio) if audio is not None else None
        self._messages.append(deepcopy(message))
        return {
            "model": response.get("model") or self.model,
            "message": message,
            "audio_path": audio_path,
            "started_ns": started_ns,
            "completed_ns": completed_ns,
        }

    def chat_transport(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_sec: float,
    ) -> dict[str, Any]:
        """Continue post-call JSON/text work in the same chat history.

        ``payload.messages`` must be an array of message objects. The remaining
        chat-completions fields, including structured-output constraints, are
        preserved. Invalid message shapes raise ``ValueError``.
        """

        messages = payload.get("messages")
        if not isinstance(messages, list) or any(
            not isinstance(row, dict) for row in messages
        ):
            raise ValueError("omni post-call adapter requires an object message array")
        # realtime backend の post-call(conversation: "none")と同じく、
        # セッション履歴(音声込み)は含めず独立リクエストで送る。
        # 会話の全文は payload 側の証跡JSONに含まれている。
        request_payload = deepcopy(payload)
        request_payload["modalities"] = ["text"]
        request_payload.pop("tools", None)
        request_payload.pop("tool_choice", None)
        response = chat_with_context_retry(
            post_json,
            url,
            request_payload,
            headers,
            timeout_sec,
            phase="ticket",
            events=self.context_length_retry,
        )
        _, audio = _extract_omni_response(response)
        if audio is not None:
            raise RuntimeError("vLLM-Omni returned audio for a text-only post-call request")
        # 独立リクエストのためセッション履歴には追記しない
        return response

    def close(self) -> None:
        """Match the persistent backend contract; HTTP needs no explicit close."""

    def _persist_audio(self, audio: bytes) -> Path:
        try:
            with wave.open(BytesIO(audio), "rb") as source:
                if source.getnframes() <= 0:
                    raise ValueError("vLLM-Omni returned an empty WAV")
        except (EOFError, wave.Error) as exc:
            raise ValueError("vLLM-Omni message.audio.data is not RIFF/WAVE") from exc
        self._turn += 1
        audio_path = self.out_dir / f"response-{self._turn:04d}.wav"
        audio_path.write_bytes(audio)
        return audio_path


def _extract_omni_response(
    response: dict[str, Any],
) -> tuple[dict[str, Any], bytes | None]:
    """Extract one assistant message and at most one WAV from chat choices.

    vLLM-Omni may return text/tool calls and ``message.audio.data`` in
    separate choices. Missing/non-object choices, malformed base64, and more
    than one audio payload raise ``ValueError``.
    """

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("vLLM-Omni response missing choices")

    messages: list[dict[str, Any]] = []
    text_parts: list[str] = []
    audio_values: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError(f"vLLM-Omni response choice missing message: {choice}")
        message = choice["message"]
        messages.append(message)
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text_parts.append(content.strip())
        raw_audio = message.get("audio")
        if raw_audio is None:
            continue
        if not isinstance(raw_audio, dict) or not isinstance(raw_audio.get("data"), str):
            raise ValueError(f"vLLM-Omni response audio is invalid: {raw_audio}")
        audio_values.append(raw_audio["data"])
    if len(audio_values) > 1:
        raise ValueError("vLLM-Omni response contains multiple audio payloads")

    primary = next(
        (
            message
            for message in messages
            if message.get("tool_calls")
            or (isinstance(message.get("content"), str) and message["content"].strip())
        ),
        messages[0],
    )
    result = deepcopy(primary)
    if text_parts:
        result["content"] = "\n".join(text_parts)
    result.pop("audio", None)
    audio = None
    if audio_values:
        try:
            audio = base64.b64decode(audio_values[0], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("vLLM-Omni response audio is not valid base64") from exc
    return result, audio
