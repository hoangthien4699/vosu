"""Orchestrator: token stream thô -> semantic events (Task E3).

Đặc tả §4.4 (review v4.1) — contract bắt buộc:

    LLM output (token stream) và application event (semantic event) là HAI
    abstraction khác nhau. Frontend không bao giờ được nhận mảnh JSON đang
    được LLM sinh dở. Nếu nhận, frontend buộc phải hiểu cách model đang cấu
    trúc JSON — rất dễ vỡ khi đổi model hoặc đổi prompt format.

        llama.cpp -> token stream -> LLM output parser (BACKEND)
                  -> semantic events -> WebSocket -> frontend

Output giờ chỉ còn MỘT trường `translation`. Cả `intent` (§4.4) lẫn `replies`
đều đã bỏ theo yêu cầu sản phẩm: máy không nghĩ hộ câu trả lời nữa — người
dùng tự nói bằng tiếng mình, máy dịch sang tiếng đối phương và đọc chậm để họ
nói theo.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

from .json_stream import IncrementalJsonParser, ParseEvent, StringDelta, ValueDone

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
#: Dấu ngoặc kép cong cũng phải tính: model sinh `”}”}”}...` chứ không phải
#: `"}"}`. Bộ ký tự chỉ có dấu thẳng sẽ trượt hoàn toàn.
_QUOTES = '"\u201c\u201d\u2018\u2019'
_TRAILING_JSON_NOISE = re.compile(
    r'[\s]*[}\]][\s,{\[\]' + _QUOTES + r']*$'
)


#: Chuỗi kẹt vòng lặp: cùng một cụm ngắn lặp lại nhiều lần liên tiếp ở cuối.
_LOOPED_TAIL = re.compile(r'(.{1,6}?)\1{4,}\s*$')


def clean_value(text: str) -> str:
    """Cắt rác cấu trúc ở cuối một giá trị chuỗi do model sinh ra.

    Cố ý hẹp: chỉ cắt khi phần đuôi CÓ dấu đóng `}` hoặc `]`. Một câu tiếng
    Việt không bao giờ kết thúc bằng những ký tự đó, còn dấu ngoặc kép hay dấu
    phẩy đứng một mình thì để nguyên.
    """
    # Cắt vòng lặp trước, rồi mới cắt rác cấu trúc — vòng lặp thường KẾT THÚC
    # bằng rác cấu trúc nên làm ngược thứ tự sẽ chỉ gỡ được một mắt xích.
    cleaned = _LOOPED_TAIL.sub("", text)
    cleaned = _TRAILING_JSON_NOISE.sub("", cleaned)
    return cleaned if cleaned.strip() else text


#: Model đôi khi lờ prompt và dùng tên trường cũ.
_TRANSLATION_KEYS = {"translation", "trans"}


@dataclass(frozen=True)
class TranslationDelta:
    text: str
    full: str


SemanticEvent = TranslationDelta


@dataclass
class CopilotResult:
    translation: str = ""
    malformed: bool = False

    @property
    def is_useful(self) -> bool:
        """Có ít nhất một thứ đáng hiển thị cho người dùng.

        Dùng để chốt mốc "first useful result" khi đo E2E (§7).
        """
        return bool(self.translation.strip())


class SemanticEventParser:
    """Ánh xạ sự kiện JSON theo path thành semantic event.

    Tách khỏi `IncrementalJsonParser` để chỗ nào cần đổi schema output của LLM
    thì chỉ sửa ánh xạ, không đụng vào bộ parse JSON.
    """

    def __init__(self) -> None:
        self._json = IncrementalJsonParser()
        self.result = CopilotResult()

    @property
    def malformed(self) -> bool:
        return self._json.malformed

    def feed(self, text: str) -> list[SemanticEvent]:
        return self._map(self._json.feed(text))

    def finish(self) -> list[SemanticEvent]:
        events = self._map(self._json.finish())
        self.result.malformed = self._json.malformed
        return events

    # ------------------------------------------------------------------ #

    def _map(self, parse_events: Iterable[ParseEvent]) -> list[SemanticEvent]:
        out: list[SemanticEvent] = []
        for event in parse_events:
            if isinstance(event, StringDelta):
                mapped = self._map_delta(event)
            else:
                mapped = self._map_done(event)
            if mapped is not None:
                out.append(mapped)
        return out

    def _map_delta(self, event: StringDelta) -> SemanticEvent | None:
        # Chỉ translation cần streaming theo ký tự: đó là thứ người dùng đọc
        # (và nghe) sớm nhất.
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



        return None


async def parse_stream(tokens: AsyncIterator[str]) -> AsyncIterator[SemanticEvent]:
    """Bọc một token stream thành luồng semantic event."""
    parser = SemanticEventParser()
    async for token in tokens:
        for event in parser.feed(token):
            yield event
    for event in parser.finish():
        yield event
