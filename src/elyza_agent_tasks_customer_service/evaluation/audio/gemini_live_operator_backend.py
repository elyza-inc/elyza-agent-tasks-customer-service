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
from elyza_agent_tasks_customer_service.evaluation.audio.realtime_operator_backend import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    _extract_json_block,
)

LIVE_PATH = "/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"


def _location(model: str) -> str:
    """モデルのリソース名からロケーションを取り出す。指定がなければ global。"""

    parts = model.split("/")
    return parts[parts.index("locations") + 1] if "locations" in parts else "global"


def _regional_host(location: str) -> str:
    return "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
INPUT_RATE = 16000
OUTPUT_RATE = 24000
MAX_EMPTY_TURN_CONTINUES = 3
# 3.8 Live は話し終えたあとも無音の音声を流し続け、ターンの終わりを数分送らないことがある。
# 書き起こしが出たあと、この秒数だけ無音しか届かなければ応答の終わりとみなす。
SILENCE_END_SEC = 4.0
SILENT_PEAK = 300  # PCM16 の振幅がこれ以下のチャンクを無音とみなす
DRAIN_MAX_SEC = 8.0


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
        self._tool_response_pending = False
        # 受信したメッセージの種類だけを残す(音声データは残さない)。応答が止まったときの切り分けに使う。
        self._event_log = (self.out_dir / "live_events.jsonl").open("a", encoding="utf-8")
        host = _regional_host(_location(model))
        declarations = []
        for definition in tools:
            function = definition.get("function") if isinstance(definition, dict) else None
            if not isinstance(function, dict):
                continue
            declarations.append({
                "name": function.get("name"),
                "description": function.get("description") or "",
                "parameters": _json_schema_to_gemini(function.get("parameters") or {}),
                # ツールの結果を受け取ってから応答を続ける(3.8 Live の既定は NON_BLOCKING)。
                "behavior": "BLOCKING",
            })
        # 通話後の応対記録も同じ Live セッションで書かせる。3.8 Live は文字で応答できないため、
        # 記録の JSON はこの関数の引数として受け取る。ツールはセッション開始時にしか登録できない。
        declarations.append({
            "name": STRUCTURED_OUTPUT_TOOL_NAME,
            "description": "通話後に応対記録の作成を指示されたときだけ呼ぶ。通話中は呼ばない。",
            "parameters": {
                "type": "OBJECT",
                "properties": {"ticket_json": {"type": "STRING", "description": "応対記録の JSON 全体"}},
                "required": ["ticket_json"],
            },
        })
        self._api_key = api_key
        self._url = f"wss://{host}{LIVE_PATH}"
        self._resume_handle: str | None = None
        self._awaiting_response = False
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
        # Live API のセッションは約10分で切られる。再開用のハンドルを受け取り、goAway や切断のときは
        # そのハンドルで接続し直して同じ会話を続ける。
        setup["sessionResumption"] = {}
        self._setup = setup
        self._open()

    def _open(self) -> None:
        from websockets.sync.client import connect as _ws_connect

        token = access_token() if self._api_key == "ADC" else self._api_key
        self._ws = _ws_connect(
            self._url,
            additional_headers={"Authorization": f"Bearer {token}"},
            open_timeout=900,
            close_timeout=30,
            max_size=None,
            # クライアント側の生存確認は送らない。応答の生成中に pong が遅れると
            # websockets が自分で接続を切ってしまう(1011 keepalive ping timeout)。
            ping_interval=None,
        )
        setup = dict(self._setup)
        if self._resume_handle:
            setup["sessionResumption"] = {"handle": self._resume_handle}
        self._ws.send(json.dumps({"setup": setup}, ensure_ascii=False))
        self._log_event("send", {"setup": True})
        while True:
            raw = self._ws.recv(timeout=900)
            first = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            self._log_event("recv", first)
            if "setupComplete" in first:
                return
            if "sessionResumptionUpdate" not in first:
                raise RuntimeError(f"Gemini Live setup failed: {json.dumps(first, ensure_ascii=False)[:200]}")

    def _reconnect(self) -> None:
        if not self._resume_handle:
            raise RuntimeError("Gemini Live session ended without a resumption handle")
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001 - 既に切れていることがある
            pass
        self._log_event("reconnect", {"handle": True})
        self._open()
        if self._awaiting_response:
            # 生成の途中で切れた応答は再開後に続かないことがあるので、応答を促し直す。
            self._send({"clientContent": {"turnComplete": True}})

    # --- 送受信 -----------------------------------------------------
    def _send(self, payload: dict[str, Any]) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            self._ws.send(json.dumps(payload, ensure_ascii=False))
        except ConnectionClosed:
            # 送る瞬間に切れていた場合も、再開用のハンドルがあれば接続し直して送り直す。
            if not getattr(self, "_resume_handle", None):
                raise
            self._reconnect()
            self._ws.send(json.dumps(payload, ensure_ascii=False))
        if hasattr(self, "_event_log"):
            self._log_event("send", {k: (sorted(v) if isinstance(v, dict) else v) for k, v in payload.items() if k != "setup"} or {"setup": True})

    def _recv(self) -> dict[str, Any]:
        from websockets.exceptions import ConnectionClosed

        while True:
            try:
                raw = self._ws.recv(timeout=900)
            except ConnectionClosed:
                if not getattr(self, "_resume_handle", None):
                    raise
                self._reconnect()
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            message = json.loads(raw)
            self._log_event("recv", message)
            update = message.get("sessionResumptionUpdate") or {}
            if update.get("resumable") and update.get("newHandle"):  # 再開用ハンドルを最新に保つ
                self._resume_handle = update["newHandle"]
            if "goAway" in message and getattr(self, "_resume_handle", None):
                self._reconnect()
                continue
            return message

    def _log_event(self, direction: str, message: dict[str, Any]) -> None:
        if not hasattr(self, "_event_log"):
            return
        server = message.get("serverContent") or {}
        summary = {
            "t": round(time.monotonic(), 3),
            "dir": direction,
            "keys": sorted(message),
            "server": sorted(k for k in server if k != "modelTurn"),
            "audio_parts": sum(1 for part in (server.get("modelTurn") or {}).get("parts") or [] if "inlineData" in part),
        }
        self._event_log.write(json.dumps(summary, ensure_ascii=False) + "\n")
        self._event_log.flush()

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
        self._tool_response_pending = True

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
        elif getattr(self, "_tool_response_pending", False):
            # toolResponse を受けると Live API 側が応答を始める。ここで turnComplete を送ると
            # 3.8 Live では始まった生成を中断してしまい、応答が返らないまま待ち続ける。
            pass
        else:
            self._send({"clientContent": {"turnComplete": True}})
        after_tool_response = getattr(self, "_tool_response_pending", False)
        self._tool_response_pending = False
        self._awaiting_response = True
        nudged = False
        started_ns: int | None = None
        text_parts: list[str] = []
        audio = bytearray()
        calls: dict[str, dict[str, Any]] = {}
        empty_turn_continues = 0
        voiced_len, last_voice_t = 0, time.monotonic()
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
                    chunk_bytes = base64.b64decode(inline["data"])
                    audio.extend(chunk_bytes)
                    if _peak(chunk_bytes) > SILENT_PEAK:
                        voiced_len, last_voice_t = len(audio), time.monotonic()
            if isinstance(transcription.get("text"), str):
                last_voice_t = time.monotonic()
            if (
                self.output_modality == "audio" and text_parts and audio and not server.get("turnComplete")
                and time.monotonic() - last_voice_t > SILENCE_END_SEC
            ):
                # 話し終えたあとの無音は採点に関係しないので切り捨て、ここで応答を確定する。
                del audio[voiced_len + OUTPUT_RATE * 2 // 2:]
                self._log_event("silence_end", {"seconds": round(len(audio) / (OUTPUT_RATE * 2), 1)})
                break
            if server.get("turnComplete"):
                if not text_parts and not audio and not calls and after_tool_response and not nudged:
                    # 3.8 Live は toolResponse のあと何も話さずにターンを終えることがある。
                    # 生成中ではないので、ここでは turnComplete を送っても中断は起きない。
                    nudged = True
                    self._send({"clientContent": {"turnComplete": True}})
                    continue
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
        self._awaiting_response = False
        self._drain()
        audio_path = None
        if len(audio) < OUTPUT_RATE * 2 // 10:
            # 0.1 秒未満の音声は発話ではない(3.8 Live は数十サンプルだけの音声を返すことがある)。
            # 電話帯域に落とすと長さ 0 になり、音声指標の採点が止まるので捨てる。
            audio.clear()
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
        """通話後の応対記録を同じ Live セッションで書かせる。"""

        messages = payload.get("messages")
        if not isinstance(messages, list) or any(not isinstance(row, dict) for row in messages):
            raise ValueError("Gemini Live post-call adapter requires an object message array")
        instruction = "\n".join(str(row.get("content") or "") for row in messages)
        instruction += (
            f"\n通話は終了しました。{STRUCTURED_OUTPUT_TOOL_NAME} を一度だけ呼び、ticket_json に応対記録の"
            "JSON 全体を文字列で渡してください。話さないでください。次の形式を厳守してください: "
            + json.dumps(payload.get("response_format"), ensure_ascii=False, sort_keys=True)
        )
        response = self.request_response(force_instruction=instruction)
        if response["audio_path"] is not None:
            # 通話後の発話は採点対象ではない。
            response["audio_path"].unlink(missing_ok=True)
        content = response["message"]["content"]
        for call in response["message"]["tool_calls"]:
            if call["function"]["name"] == STRUCTURED_OUTPUT_TOOL_NAME:
                arguments = call["function"]["arguments"]
                try:
                    content = json.loads(arguments).get("ticket_json") or arguments
                except (json.JSONDecodeError, AttributeError):
                    content = arguments
                break
        extracted = _extract_json_block(content)
        return {
            "model": self.model,
            "choices": [{"message": {"role": "assistant", "content": extracted or content}}],
        }

    def _drain(self) -> None:
        """ターン終了後に残っているメッセージを読み捨てる。

        残したまま次の入力を送ると、割り込みとして扱われて応答が欠ける。
        """

        deadline = time.monotonic() + 5.0
        hard_stop = time.monotonic() + DRAIN_MAX_SEC
        while time.monotonic() < min(deadline, hard_stop):
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
            if '"sessionResumptionUpdate"' in raw or '"goAway"' in raw:
                # 読み捨てる中にも再開用のハンドルと切断の予告は拾う。
                message = json.loads(raw)
                self._log_event("recv", message)
                update = message.get("sessionResumptionUpdate") or {}
                if update.get("resumable") and update.get("newHandle"):
                    self._resume_handle = update["newHandle"]
                if "goAway" in message and getattr(self, "_resume_handle", None):
                    self._reconnect()
                    return

    def close(self) -> None:
        try:
            self._event_log.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001 - 切断の失敗は評価に影響しない
            pass


def _peak(pcm16: bytes) -> int:
    """PCM16 チャンクの最大振幅。"""

    import array

    samples = array.array("h", pcm16[: len(pcm16) // 2 * 2])
    return max((abs(v) for v in samples), default=0)


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
