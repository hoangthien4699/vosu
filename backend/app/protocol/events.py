"""Event types + Event Bus nội bộ (Task F1, F2).

Đặc tả §4.3: mọi message BẮT BUỘC có `session_id`, `utterance_id`, `sequence`,
`timestamp`, `type`, `data`.

Đặc tả §2.3 (review v2.1): Event Bus ở MVP chỉ là `asyncio.Queue` — cố ý KHÔNG
dùng message broker thật (Kafka/RabbitMQ/Redis).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    # --- vòng đời session ---
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    AUDIO_STARTED = "audio_started"

    # --- STT ---
    #: VAD vừa chốt xong một câu — phát NGAY, trước khi STT chạy.
    #:
    #: Client phát file cần mốc này để dừng đúng chỗ câu vừa dứt. Nếu đợi
    #: `stt_final` thì file đã chạy quá thêm ~1.9s (thời gian Whisper nghe),
    #: tức là đã phát lấn sang câu sau rồi mới dừng.
    UTTERANCE_ENDPOINT = "utterance_endpoint"
    #: Nghe ra câu còn dở — đang chờ người ta nói tiếp để ghép.
    #:
    #: Client phát file PHẢI phát tiếp khi nhận cái này: nó đã dừng file ở
    #: `utterance_endpoint`, không báo thì audio cần để quyết định sẽ không
    #: bao giờ tới và file dừng vĩnh viễn.
    UTTERANCE_CONTINUED = "utterance_continued"
    STT_PARTIAL = "stt_partial"
    STT_FINAL = "stt_final"

    # --- Copilot: lifecycle ---
    COPILOT_STARTED = "copilot_started"
    COPILOT_DONE = "copilot_done"

    # --- Copilot: semantic events (§4.4) ---
    # Frontend CHỈ được thấy các event này, không bao giờ thấy JSON thô dở dang
    # đang được LLM sinh.
    TRANSLATION_DELTA = "translation_delta"

    # --- TTS (§2.4.1) ---
    TTS_STARTED = "tts_started"
    TTS_AUDIO_CHUNK = "tts_audio_chunk"
    TTS_DONE = "tts_done"
    TTS_CANCELLED = "tts_cancelled"
    TTS_ERROR = "tts_error"

    # --- khác ---
    #: Câu bị bỏ vì xử lý không kịp. Người dùng PHẢI biết mình vừa mất một câu,
    #: chứ không phải im lặng như thể đối phương không nói gì.
    UTTERANCE_DROPPED = "utterance_dropped"
    ERROR = "error"


#: Event mang payload nhị phân — gửi qua WebSocket binary frame, không JSON.
BINARY_EVENTS = frozenset({EventType.TTS_AUDIO_CHUNK})

#: Event thuộc vòng đời một utterance cụ thể (bắt buộc có utterance_id).
UTTERANCE_SCOPED = frozenset(
    {
        EventType.UTTERANCE_ENDPOINT,
        EventType.UTTERANCE_CONTINUED,
        EventType.STT_PARTIAL,
        EventType.STT_FINAL,
        EventType.COPILOT_STARTED,
        EventType.COPILOT_DONE,
        EventType.TRANSLATION_DELTA,
        EventType.TTS_STARTED,
        EventType.TTS_AUDIO_CHUNK,
        EventType.TTS_DONE,
        EventType.TTS_CANCELLED,
        EventType.TTS_ERROR,
        EventType.UTTERANCE_DROPPED,
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


@dataclass
class Event:
    type: EventType
    session_id: str
    utterance_id: str | None = None
    sequence: int = 0
    timestamp: str = field(default_factory=_utc_now)
    data: dict[str, Any] = field(default_factory=dict)
    #: payload nhị phân cho TTS_AUDIO_CHUNK — không serialize vào JSON.
    binary: bytes | None = field(default=None, repr=False)
    #: mốc thời gian đơn điệu để đo latency nội bộ (không gửi cho client).
    monotonic_ns: int = field(default_factory=time.monotonic_ns, repr=False)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "utterance_id": self.utterance_id,
            "sequence": self.sequence,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class EventBus:
    """Hàng đợi event một chiều cho MỘT session.

    `sequence` được cấp phát ở đây, không phải ở nơi tạo event — đảm bảo thứ tự
    message trong một session là liên tục và duy nhất (Task F2).
    """

    def __init__(self, session_id: str, maxsize: int = 512) -> None:
        self.session_id = session_id
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=maxsize)
        self._counter = itertools.count()
        self._closed = False
        self._dropped = 0

    @property
    def dropped(self) -> int:
        return self._dropped

    def next_sequence(self) -> int:
        return next(self._counter)

    async def publish(self, event: Event) -> None:
        if self._closed:
            logger.debug("Bỏ qua event %s: bus đã đóng", event.type.value)
            return
        event.sequence = self.next_sequence()
        await self._queue.put(event)

    def publish_nowait(self, event: Event) -> bool:
        """Publish không chờ. Trả về False nếu hàng đợi đầy (backpressure)."""
        if self._closed:
            return False
        event.sequence = self.next_sequence()
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning(
                "Event bus đầy, drop %s (tổng đã drop: %d)",
                event.type.value,
                self._dropped,
            )
            return False

    async def emit(
        self,
        type: EventType,
        *,
        utterance_id: str | None = None,
        data: dict[str, Any] | None = None,
        binary: bytes | None = None,
    ) -> None:
        await self.publish(
            Event(
                type=type,
                session_id=self.session_id,
                utterance_id=utterance_id,
                data=data or {},
                binary=binary,
            )
        )

    async def get(self) -> Event | None:
        return await self._queue.get()

    async def close(self) -> None:
        """Đóng bus, đánh thức consumer đang chờ bằng sentinel None."""
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)

    def __aiter__(self) -> EventBus:
        return self

    async def __anext__(self) -> Event:
        event = await self._queue.get()
        if event is None:
            raise StopAsyncIteration
        return event
