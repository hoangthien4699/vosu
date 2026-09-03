"""Orchestrator: token stream thô -> semantic events (Task E3, E9).

Đặc tả §4.4 (review v4.1) — contract bắt buộc:

    LLM output (token stream) và application event (semantic event) là HAI
    abstraction khác nhau. Frontend không bao giờ được nhận mảnh JSON đang
    được LLM sinh dở. Nếu nhận, frontend buộc phải hiểu cách Qwen cấu trúc
    JSON — vỡ ngay khi đổi model hoặc đổi prompt format.

        llama.cpp -> token stream -> LLM output parser (BACKEND)
                  -> semantic events -> WebSocket -> frontend

MVP scope đọc tự động (§2.4.1, review v4.1):
    - translation -> AUTO
    - intent      -> chỉ hiển thị UI, TTS tùy chọn (mặc định tắt)
    - reply       -> chỉ đọc khi người dùng chọn thủ công
Lý do: người đối diện nói liên tục nhiều câu, đọc hết mọi gợi ý sẽ gây audio
overload — giọng AI chồng lấp lên hội thoại thật đang diễn ra.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field

from .json_stream import (
    ContainerDone,
    IncrementalJsonParser,
    ParseEvent,
    StringDelta,
    ValueDone,
)

logger = logging.getLogger(__name__)

#: Rác cấu trúc JSON lọt vào CUỐI một giá trị chuỗi.
#
# Sinh có grammar đảm bảo cấu trúc đúng, nhưng `{`, `}`, `[`, `]` là ký tự hợp
# lệ BÊN TRONG chuỗi JSON nên grammar không cấm được. Quan sát thật với
# Gemma 3 4B: model muốn đóng object và mở object mới, nhưng token nó chọn đặt
# `},{` vào trong chuỗi trước rồi mới đóng:
#
#     "meaning":"Được rồi, chúng ta tạm dừng lại đây.},{"
#
# JSON vẫn hợp lệ, chỉ là người dùng đọc thấy rác. Đã thử cấm bằng `pattern`
# trong JSON Schema — llama.cpp bỏ qua.
_TRAILING_JSON_NOISE = re.compile(r'[\s]*[}\]][\s,{\[\]"]*$')


def clean_value(text: str) -> str:
    """Cắt rác cấu trúc ở cuối một giá trị chuỗi do model sinh ra.

    Cố ý hẹp: chỉ cắt khi phần đuôi CÓ dấu đóng `}` hoặc `]`. Một câu tiếng
    Việt không bao giờ kết thúc bằng những ký tự đó, còn dấu ngoặc kép hay dấu
    phẩy đứng một mình thì để nguyên.
    """
    cleaned = _TRAILING_JSON_NOISE.sub("", text)
    return cleaned if cleaned.strip() else text


#: Model đôi khi lờ prompt và dùng tên trường của baseline v1.
_TRANSLATION_KEYS = {"translation", "trans"}
_INTENT_KEYS = {"intent", "cultural_intent"}
_REPLY_KEYS = {"replies", "suggested_replies"}


@dataclass(frozen=True)
class TranslationDelta:
    text: str
    full: str


@dataclass(frozen=True)
class IntentDone:
    intent: str


@dataclass(frozen=True)
class ReplyReady:
    index: int
    text: str
    #: Bản dịch tiếng Việt của `text` — để người dùng biết mình sắp nói gì.
    #: Rỗng nếu model không sinh ra hoặc `llm.reply_meaning` đang tắt.
    meaning: str = ""


SemanticEvent = TranslationDelta | IntentDone | ReplyReady


@dataclass
class CopilotResult:
    translation: str = ""
    intent: str = ""
    replies: list[str] = field(default_factory=list)
    malformed: bool = False

    @property
    def is_useful(self) -> bool:
        """Có ít nhất một thứ đáng hiển thị cho người dùng.

        Dùng để chốt mốc "first useful result" khi đo E2E (§7).
        """
        return bool(self.translation.strip() or self.intent.strip() or self.replies)


class SemanticEventParser:
    """Ánh xạ sự kiện JSON theo path thành semantic event.

    Tách khỏi `IncrementalJsonParser` để chỗ nào cần đổi schema output của LLM
    thì chỉ sửa ánh xạ, không đụng vào bộ parse JSON.
    """

    def __init__(self) -> None:
        self._json = IncrementalJsonParser()
        self.result = CopilotResult()
        self._emitted_replies: set[int] = set()
        self._intent_emitted = False
        #: reply đang gom dở: index -> {"text": ..., "meaning": ...}
        self._pending_replies: dict[int, dict[str, str]] = {}

    @property
    def malformed(self) -> bool:
        return self._json.malformed

    def feed(self, text: str) -> list[SemanticEvent]:
        return self._map(self._json.feed(text))

    def finish(self) -> list[SemanticEvent]:
        events = self._map(self._json.finish())
        # JSON bị cắt giữa chừng vẫn phải giao reply đã gom được — thà thiếu
        # `purpose` còn hơn mất luôn câu gợi ý.
        for index in sorted(self._pending_replies):
            emitted = self._flush_reply(index)
            if emitted is not None:
                events.append(emitted)
        self.result.malformed = self._json.malformed
        return events

    # ------------------------------------------------------------------ #

    def _map(self, parse_events: Iterable[ParseEvent]) -> list[SemanticEvent]:
        out: list[SemanticEvent] = []
        for event in parse_events:
            if isinstance(event, StringDelta):
                mapped = self._map_delta(event)
            elif isinstance(event, ContainerDone):
                mapped = self._map_container(event)
            else:
                mapped = self._map_done(event)
            if mapped is not None:
                out.append(mapped)
        return out

    def _map_container(self, event: ContainerDone) -> SemanticEvent | None:
        """Một reply dạng object đã đủ cả `text` lẫn `purpose`."""
        path = event.path
        if len(path) == 2 and path[0] in _REPLY_KEYS and isinstance(path[1], int):
            return self._flush_reply(path[1])
        return None

    def _flush_reply(self, index: int) -> SemanticEvent | None:
        fields = self._pending_replies.pop(index, None)
        if fields is None or index in self._emitted_replies:
            return None
        text = clean_value(fields.get("text", "")).strip()
        if not text:
            return None
        self._emitted_replies.add(index)
        self._store_reply(index, text)
        # Chấp nhận cả `purpose`: model đôi khi bám theo từ khóa cũ nếu prompt
        # được sửa mà cache prompt phía server chưa kịp đổi.
        meaning = clean_value(fields.get("meaning") or fields.get("purpose") or "").strip()
        return ReplyReady(index=index, text=text, meaning=meaning)

    def _store_reply(self, index: int, text: str) -> None:
        while len(self.result.replies) <= index:
            self.result.replies.append("")
        self.result.replies[index] = text

    def _map_delta(self, event: StringDelta) -> SemanticEvent | None:
        # Chỉ translation cần streaming theo ký tự: đó là thứ người dùng đọc
        # (và nghe) sớm nhất. intent/reply chỉ có nghĩa khi đã hoàn chỉnh.
        if len(event.path) == 1 and event.path[0] in _TRANSLATION_KEYS:
            self.result.translation += event.text
            return TranslationDelta(text=event.text, full=self.result.translation)
        return None

    def _map_done(self, event: ValueDone) -> SemanticEvent | None:
        path = event.path
        if not path:
            return None

        head = path[0]

        if len(path) == 1 and head in _TRANSLATION_KEYS:
            # Nội dung đã ghép dần qua delta. Nếu việc dọn rác làm đổi kết quả
            # thì phát thêm một delta sửa lại — nếu không, UI sẽ giữ nguyên
            # phần rác đã hiện ra.
            cleaned = clean_value(str(event.value))
            changed = cleaned != self.result.translation
            self.result.translation = cleaned
            if changed:
                return TranslationDelta(text="", full=cleaned)
            return None

        if len(path) == 1 and head in _INTENT_KEYS and not self._intent_emitted:
            intent = clean_value(str(event.value)).strip()
            self.result.intent = intent
            self._intent_emitted = True
            return IntentDone(intent=intent)

        if head in _REPLY_KEYS:
            return self._map_reply(path, event.value)

        return None

    def _map_reply(self, path, value) -> SemanticEvent | None:
        # ("replies", 0)                -> mảng chuỗi: phát ngay
        # ("replies", 0, "text")        -> mảng object: gom, chờ ContainerDone
        # ("replies", 0, "meaning")     -> bản dịch tiếng Việt của câu đó
        if len(path) == 2 and isinstance(path[1], int):
            index = path[1]
            if index in self._emitted_replies:
                return None
            text = clean_value(str(value)).strip()
            if not text:
                return None
            self._emitted_replies.add(index)
            self._store_reply(index, text)
            return ReplyReady(index=index, text=text)

        if len(path) == 3 and isinstance(path[1], int) and path[2] in ("text", "meaning", "purpose"):
            # Gom lại, chỉ phát khi object đóng — lúc đó mới đủ cả hai trường.
            self._pending_replies.setdefault(path[1], {})[str(path[2])] = str(value)
            return None

        return None


async def parse_stream(tokens: AsyncIterator[str]) -> AsyncIterator[SemanticEvent]:
    """Bọc một token stream thành luồng semantic event."""
    parser = SemanticEventParser()
    async for token in tokens:
        for event in parser.feed(token):
            yield event
    for event in parser.finish():
        yield event
