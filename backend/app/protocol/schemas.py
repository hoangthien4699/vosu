"""Pydantic schema cho envelope + payload từng event type (Task F3).

Đặc tả §4.3: event thiếu `session_id`/`utterance_id`/`sequence`/`timestamp`
phải bị reject ở tầng validate — không để lọt ra WebSocket.

Đặc tả §4.4: output LLM rút gọn — KHÔNG có trường `meaning` cho từng reply
(mỗi reply thêm bản dịch sẽ tăng token → tăng latency).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .events import UTTERANCE_SCOPED, Event, EventType


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Payload theo từng event type
# --------------------------------------------------------------------------- #


class SessionStartedData(_Strict):
    platform: str
    max_concurrent_sessions: int
    sample_rate: int
    tts_enabled: bool


class SessionEndedData(_Strict):
    reason: str = "client_disconnect"
    utterances: int = 0


class AudioStartedData(_Strict):
    sample_rate: int
    channels: int


class SttPartialData(_Strict):
    text: str
    language: str | None = None
    window_s: float = Field(ge=0.0)


class SttFinalData(_Strict):
    text: str
    language: str | None = None
    duration_s: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    #: Vị trí bắt đầu câu, tính bằng giây kể từ byte audio đầu tiên của phiên.
    #: Client đếm cùng một dòng byte nên cắt được đúng đoạn audio GỐC của câu.
    start_s: float = Field(default=0.0, ge=0.0)


class CopilotStartedData(_Strict):
    source_text: str
    language: str | None = None


class CopilotDoneData(_Strict):
    ttft_ms: float | None = Field(default=None, ge=0.0)
    total_ms: float = Field(ge=0.0)
    tokens: int = Field(default=0, ge=0)
    truncated: bool = False


class TranslationDeltaData(_Strict):
    """Delta cộng dồn của bản dịch. `text` là phần MỚI, `full` là toàn bộ tới giờ."""

    text: str
    full: str


class ReplyReadyData(_Strict):
    index: int = Field(ge=0)
    text: str
    #: Bản dịch tiếng Việt của `text`. Rỗng nếu llm.reply_meaning tắt.
    meaning: str = ""


class TtsStartedData(_Strict):
    utterance_field: Literal["translation", "reply"]
    text: str
    voice: str
    sample_rate: int


class TtsAudioChunkData(_Strict):
    """Metadata đi kèm; PCM thật gửi qua binary frame riêng."""

    chunk_index: int = Field(ge=0)
    bytes: int = Field(ge=0)
    sample_rate: int


class TtsDoneData(_Strict):
    chunks: int = Field(ge=0)
    synthesis_ms: float = Field(ge=0.0)


class TtsCancelledData(_Strict):
    reason: Literal["barge_in", "new_utterance", "session_end", "client_request"]
    #: §2.4.1 / B8 — thời gian từ lúc nhận tín hiệu tới lúc thực sự dừng phát.
    response_ms: float = Field(ge=0.0)
    chunks_sent: int = Field(default=0, ge=0)


class TtsErrorData(_Strict):
    message: str
    stage: Literal["synthesis", "playback"] = "synthesis"


class ErrorData(_Strict):
    message: str
    code: str = "internal_error"
    recoverable: bool = True


PAYLOAD_SCHEMAS: dict[EventType, type[BaseModel]] = {
    EventType.SESSION_STARTED: SessionStartedData,
    EventType.SESSION_ENDED: SessionEndedData,
    EventType.AUDIO_STARTED: AudioStartedData,
    EventType.STT_PARTIAL: SttPartialData,
    EventType.STT_FINAL: SttFinalData,
    EventType.COPILOT_STARTED: CopilotStartedData,
    EventType.COPILOT_DONE: CopilotDoneData,
    EventType.TRANSLATION_DELTA: TranslationDeltaData,
    EventType.REPLY_READY: ReplyReadyData,
    EventType.TTS_STARTED: TtsStartedData,
    EventType.TTS_AUDIO_CHUNK: TtsAudioChunkData,
    EventType.TTS_DONE: TtsDoneData,
    EventType.TTS_CANCELLED: TtsCancelledData,
    EventType.TTS_ERROR: TtsErrorData,
    EventType.ERROR: ErrorData,
}


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #


class EventEnvelope(_Strict):
    session_id: str = Field(min_length=1)
    utterance_id: str | None = None
    sequence: int = Field(ge=0)
    type: EventType
    timestamp: str = Field(min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> EventEnvelope:
        if self.type in UTTERANCE_SCOPED and not self.utterance_id:
            raise ValueError(
                f"event {self.type.value!r} thuộc phạm vi utterance nhưng thiếu utterance_id"
            )
        schema = PAYLOAD_SCHEMAS.get(self.type)
        if schema is not None:
            schema.model_validate(self.data)
        return self


class SchemaValidationError(ValueError):
    """Event không hợp lệ — không được gửi ra client."""


def validate_event(event: Event) -> dict[str, Any]:
    """Validate và trả về dict JSON sẵn sàng gửi. Raise nếu không hợp lệ."""
    try:
        envelope = EventEnvelope.model_validate(event.to_json_dict())
    except Exception as exc:  # pydantic.ValidationError hoặc ValueError
        raise SchemaValidationError(
            f"Event {event.type.value!r} (seq={event.sequence}) không hợp lệ: {exc}"
        ) from exc
    return envelope.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Output LLM — §4.4 (KHÔNG có trường `meaning`)
# --------------------------------------------------------------------------- #


class CopilotOutput(_Strict):
    translation: str = ""
    replies: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Message client -> server (control frame dạng JSON)
# --------------------------------------------------------------------------- #


class ClientControl(_Strict):
    action: Literal["speak_reply", "speak", "set_tts_mode", "cancel_tts", "reset", "ping"]
    reply_index: int | None = None
    text: str | None = None
    #: `speak`: đọc phần nào (đổi giọng và gắn nhãn cho đúng)
    field: Literal["translation", "reply"] | None = None
    #: `speak`: ép dùng giọng cụ thể ("vi" hoặc "en"). None = theo config.
    voice: Literal["vi", "en"] | None = None
    #: `set_tts_mode`: "auto" = server tự đọc translation (mặc định, §2.4.1);
    #: "manual" = server không tự đọc gì, client tự yêu cầu từng phần theo
    #: thứ tự mình muốn. Dùng cho chế độ nghe lại từng câu.
    mode: Literal["auto", "manual"] | None = None
