"""Gemini Live API を使うオペレータ役のバックエンド。

RealtimeOperatorBackend と同じ公開面を持つ。呼び出し側は差し替えるだけで、
OpenAI Realtime と同様に音声の往復と関数呼び出しを扱える。
プロトコルは BidiGenerateContent(WebSocket)で、setup → clientContent →
toolCall → toolResponse の順に往復する。
"""

from __future__ import annotations

import base64
import json
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from elyza_agent_tasks_customer_service.evaluation.audio.omni_audio_common import (
    ffmpeg_to_mono_pcm16,
    write_mono_pcm16_wav,
)
from elyza_agent_tasks_customer_service.evaluation.llm.google_credentials import access_token

LIVE_URL = (
    "wss://aiplatform.googleapis.com/ws/"
    "google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
)
INPUT_RATE = 16000
OUTPUT_RATE = 24000
MAX_EMPTY_TURN_CONTINUES = 3


def _json_schema_to_gemini(schema: Any) -> Any:
    """JSON Schema の型名を Gemini の大文字表記へ変換する。"""

    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out["type"] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {k: _json_schema_to_gemini(v) for k, v in value.items()}
        elif key == "items":
            out["items"] = _json_schema_to_gemini(value)
        elif key in {"additionalProperties", "$schema"}:
            continue
        else:
            out[key] = value
    return out


class GeminiLiveOperatorBackend:
    """Gemini Live API を OpenAI Realtime と同じ公開面で扱う。"""

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
        voice: str = "Puck",
        post_call_model: str = "google/gemini-3.6-flash",
    ) -> None:
        if output_modality not in {"text", "audio"}:
            raise ValueError("output_modality must be text or audio")
        try:
            from websockets.sync.client import connect as _ws_connect
        except ImportError as exc:
            raise RuntimeError("Gemini Live backend requires the websockets package") from exc
        self.model = model
        self.out_dir = out_dir / "gemini_live_operator"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._turn = 0
        self.output_modality = output_modality
        self._pending_calls: dict[str, str] = {}
        self._audio_pending = False
        # 通話後の文字処理は OpenAI 互換エンドポイントへ送る。
        # model は "projects/.../models/<name>" 形式なので短縮名へ直す。
        # Live 用モデルは WebSocket 専用で chat/completions では使えないため、
        # 通話後の文字処理には通常のテキストモデルを使う。
        self._post_call_model = post_call_model
        base = endpoint.rstrip("/")
        project_path = model.split("/publishers/")[0] if "/publishers/" in model else ""
        self._post_call_url = f"{base}/v1beta1/{project_path}/endpoints/openapi/chat/completions"
        declarations = []
        for definition in tools:
            function = definition.get("function") if isinstance(definition, dict) else None
            if not isinstance(function, dict):
                continue
            declarations.append({
                "name": function.get("name"),
                "description": function.get("description") or "",
                "parameters": _json_schema_to_gemini(function.get("parameters") or {}),
            })
        token = access_token() if api_key == "ADC" else api_key
        self._ws = _ws_connect(
            LIVE_URL,
            additional_headers={"Authorization": f"Bearer {token}"},
            open_timeout=900,
            close_timeout=30,
            max_size=None,
        )
        setup: dict[str, Any] = {
            "model": model,
            "realtimeInputConfig": {"automaticActivityDetection": {"disabled": True}},
            "generationConfig": {
                "responseModalities": ["AUDIO" if output_modality == "audio" else "TEXT"],
                "temperature": temperature,
            },
            "systemInstruction": {"parts": [{"text": instructions}]},
        }
        if output_modality == "audio":
            setup["generationConfig"]["speechConfig"] = {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            }
            # 音声応答は文字を返さないため、転写を有効にして採点対象の発話を得る。
            setup["outputAudioTranscription"] = {}
            setup["inputAudioTranscription"] = {}
        if declarations:
            setup["tools"] = [{"functionDeclarations": declarations}]
        self._send({"setup": setup})
        first = self._recv()
        if "setupComplete" not in first:
            raise RuntimeError(f"Gemini Live setup failed: {json.dumps(first, ensure_ascii=False)[:200]}")

    # --- 送受信 -----------------------------------------------------
    def _send(self, payload: dict[str, Any]) -> None:
        self._ws.send(json.dumps(payload, ensure_ascii=False))

    def _recv(self) -> dict[str, Any]:
        raw = self._ws.recv(timeout=900)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    # --- 公開面 -----------------------------------------------------

    def send_user_audio(self, wav_path: Path) -> None:
        with wave.open(str(wav_path), "rb") as probe:
            source_rate = probe.getframerate()
        send_path = wav_path
        if source_rate != INPUT_RATE:
            resampled = Path(tempfile.mkstemp(suffix=".wav", dir=str(self.out_dir))[1])
            ffmpeg_to_mono_pcm16(wav_path, resampled, sample_rate=INPUT_RATE)
            send_path = resampled
        with wave.open(str(send_path), "rb") as source:
            if source.getsampwidth() != 2 or source.getnchannels() != 1:
                raise ValueError(f"Gemini Live input must be mono PCM16 WAV: {wav_path}")
            frames = source.readframes(source.getnframes())
        if send_path != wav_path:
            send_path.unlink(missing_ok=True)
        self._send({"realtimeInput": {"activityStart": {}}})
        self._send({
            "realtimeInput": {
                "audio": {
                    "mimeType": f"audio/pcm;rate={INPUT_RATE}",
                    "data": base64.b64encode(frames).decode("ascii"),
                }
            }
        })
        self._send({"realtimeInput": {"activityEnd": {}}})
        self._audio_pending = True

    def send_user_text(self, text: str) -> None:
        self._send({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turnComplete": False,
            }
        })

    def submit_tool_result(self, call_id: str, result: Any) -> None:
        self._send({
            "toolResponse": {
                "functionResponses": [{
                    "id": call_id,
                    "name": self._pending_calls.get(call_id, ""),
                    "response": result if isinstance(result, dict) else {"result": result},
                }]
            }
        })

    def request_response(self, *, force_instruction: str | None = None) -> dict[str, Any]:
        LATE_TRANSCRIPTION_RECV_TIMEOUT_SEC = 2.0
        MAX_LATE_TRANSCRIPTION_RECVS = 5
        if force_instruction:
            self._send({
                "clientContent": {
                    "turns": [{"role": "user", "parts": [{"text": force_instruction}]}],
                    "turnComplete": True,
                }
            })
        elif self._audio_pending:
            # activityEnd 済みの音声入力は Live API 側が応答を始める。
            # 空のターン確定は送らない。
            self._audio_pending = False
        else:
            self._send({"clientContent": {"turnComplete": True}})
        started_ns: int | None = None
        text_parts: list[str] = []
        audio = bytearray()
        calls: dict[str, dict[str, Any]] = {}
        empty_turn_continues = 0
        while True:
            message = self._recv()
            if started_ns is None:
                started_ns = time.monotonic_ns()
            if "toolCall" in message:
                for call in message["toolCall"].get("functionCalls") or []:
                    call_id = str(call.get("id") or "")
                    name = str(call.get("name") or "")
                    self._pending_calls[call_id] = name
                    calls[call_id] = {
                        "call_id": call_id,
                        "name": name,
                        "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False),
                    }
                break
            server = message.get("serverContent") or {}
            if server.get("interrupted"):
                # 打ち切り分は同じ内容で作り直されるため、重複を防ぐために破棄する。
                text_parts.clear()
                audio.clear()
                started_ns = None
                continue
            transcription = server.get("outputTranscription") or {}
            # AUDIO 応答では音声書き起こしのみを応答テキストとして扱う。
            if self.output_modality == "audio" and isinstance(transcription.get("text"), str):
                # Live API may resend the whole transcript after streaming it in chunks;
                # a chunk that starts with everything so far replaces it instead of doubling it.
                chunk = transcription["text"]
                joined = "".join(text_parts)
                if joined and chunk.startswith(joined):
                    text_parts[:] = [chunk]
                else:
                    text_parts.append(chunk)
            for part in (server.get("modelTurn") or {}).get("parts") or []:
                # TEXT 応答では modelTurn のテキストを応答テキストとして扱う。
                if self.output_modality != "audio" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                inline = part.get("inlineData") or {}
                if isinstance(inline.get("data"), str):
                    audio.extend(base64.b64decode(inline["data"]))
            if server.get("turnComplete"):
                if not text_parts and not audio and not calls and empty_turn_continues < MAX_EMPTY_TURN_CONTINUES:
                    # toolResponse の直後は、本応答前に空の turnComplete が来ることがある。
                    # 本応答を取りこぼさないよう、上限まで次のメッセージを待つ。
                    empty_turn_continues += 1
                    continue
                if audio and not text_parts:
                    # toolResponse 直後は turnComplete 後に音声書き起こしが遅れて届くことがある。
                    # 短時間だけ受信を続け、遅延した outputTranscription を回収する。
                    for _ in range(MAX_LATE_TRANSCRIPTION_RECVS):
                        try:
                            raw = self._ws.recv(timeout=LATE_TRANSCRIPTION_RECV_TIMEOUT_SEC)
                        except Exception:  # noqa: BLE001 - タイムアウト時は書き起こしなしで確定する
                            break
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        delayed_server = json.loads(raw).get("serverContent") or {}
                        delayed_transcription = delayed_server.get("outputTranscription") or {}
                        if isinstance(delayed_transcription.get("text"), str):
                            text_parts.append(delayed_transcription["text"])
                break
        completed_ns = time.monotonic_ns()
        self._drain()
        audio_path = None
        if audio:
            self._turn += 1
            audio_path = self.out_dir / f"response-{self._turn:04d}.wav"
            write_mono_pcm16_wav(audio_path, bytes(audio), sample_rate=OUTPUT_RATE)
        tool_calls = [
            {"id": c["call_id"], "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in calls.values()
        ]
        return {
            "message": {"role": "assistant", "content": _undouble_transcript("".join(text_parts).strip()),
                        "tool_calls": tool_calls},
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
        """通話後の応対記録など、文字だけのやり取りを行う。

        Live セッションには音声履歴が積まれているため、そこには追記せず、
        OpenAI 互換エンドポイントへ独立した要求として送る。会話の全文は
        payload 側の証跡に含まれている。
        """

        from elyza_agent_tasks_customer_service.evaluation.llm.llm_io import chat_with_context_retry, post_json

        messages = payload.get("messages")
        if not isinstance(messages, list) or any(not isinstance(row, dict) for row in messages):
            raise ValueError("Gemini Live post-call adapter requires an object message array")
        request_payload = dict(payload)
        request_payload["model"] = self._post_call_model
        request_payload.pop("tools", None)
        request_payload.pop("tool_choice", None)
        return chat_with_context_retry(
            post_json,
            self._post_call_url,
            request_payload,
            headers,
            timeout_sec,
            phase="ticket",
            events=[],
        )

    def _drain(self) -> None:
        """ターン終了後に残っているメッセージを読み捨てる。

        残したまま次の入力を送ると、割り込みとして扱われて応答が欠ける。
        """

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                raw = self._ws.recv(timeout=1.0)
            except Exception:  # noqa: BLE001 - 残りが無ければ読み取りは失敗する
                return
            # 残りが来ている間は期限を延ばし、静かになるまで読み捨てる
            deadline = time.monotonic() + 1.5
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "ignore")
            if '"setupComplete"' in raw:
                return

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001 - 切断の失敗は評価に影響しない
            pass


def _undouble_transcript(text: str) -> str:
    """Collapse a transcript the Live API delivered twice in full (X + X) to X.

    The spoken audio carries the utterance once (checked against the ASR of the
    same audio); only the streamed transcription is repeated.
    """

    for cut in range(len(text) // 2 - 3, len(text) // 2 + 4):
        head, tail = text[:cut].strip(), text[cut:].strip()
        if head and head == tail:
            return head
    return text
