"""Parser JSON tăng dần theo ký tự — nền cho semantic events (§4.4).

LLM sinh JSON theo token. Không thể `json.loads()` cho tới khi sinh xong, mà
đợi sinh xong thì mất hết lợi ích của streaming. Module này parse JSON *trong
lúc nó đang được sinh* và phát sự kiện theo đường dẫn (path):

    {"translation": "Tôi nghĩ...     -> StringDelta(path=("translation",), "Tôi nghĩ")
    ..."}                            -> ValueDone(path=("translation",), "Tôi nghĩ...")
    "replies": ["a", "b"]            -> ValueDone(("replies", 0), "a") ...

Cố ý KHÔNG dùng thư viện JSON tăng dần bên ngoài: cần kiểm soát chính xác
hành vi khi gặp output méo (LLM 3B lượng tử hóa 4-bit sinh JSON hỏng là
chuyện thường), và cần tolerant hơn json.loads.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

Path = tuple[str | int, ...]

_WS = " \t\r\n"
_HEX = "0123456789abcdefABCDEF"
_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


@dataclass(frozen=True)
class StringDelta:
    """Phần ký tự MỚI của một chuỗi đang được sinh dở."""

    path: Path
    text: str


@dataclass(frozen=True)
class ValueDone:
    """Một giá trị đã hoàn tất (chuỗi đóng ngoặc kép, số kết thúc, ...)."""

    path: Path
    value: Any


ParseEvent = StringDelta | ValueDone


class _S(Enum):
    PREFIX = auto()      # bỏ qua rác trước `{` (markdown fence, lời dẫn của model)
    VALUE = auto()
    OBJ_KEY = auto()
    OBJ_COLON = auto()
    OBJ_NEXT = auto()
    ARR_NEXT = auto()
    STRING = auto()
    LITERAL = auto()
    DONE = auto()


@dataclass
class _Frame:
    kind: str            # "object" | "array"
    key: str | None = None
    index: int = -1

    def path_component(self) -> str | int:
        return self.key if self.kind == "object" else self.index


class IncrementalJsonParser:
    """Nạp từng mẩu text; sinh StringDelta / ValueDone."""

    def __init__(self, *, emit_string_deltas: bool = True) -> None:
        self.emit_string_deltas = emit_string_deltas
        self._state = _S.PREFIX
        self._stack: list[_Frame] = []
        self._buf: list[str] = []          # nội dung chuỗi/literal đang tích lũy
        self._is_key = False
        self._escape = False
        #: None = không đang thu thập \uXXXX; list = các chữ số hex đã nhận
        self._unicode: list[str] | None = None
        #: nửa cao của surrogate pair đang chờ nửa thấp (\ud83d\ude00)
        self._high_surrogate: int | None = None
        self.malformed = False

    # ------------------------------------------------------------------ #

    @property
    def finished(self) -> bool:
        return self._state is _S.DONE

    def _path(self) -> Path:
        return tuple(f.path_component() for f in self._stack)

    def _child_path(self) -> Path:
        return self._path()

    def feed(self, text: str) -> Iterator[ParseEvent]:
        for ch in text:
            if self._state is _S.DONE:
                return
            yield from self._step(ch)

    def finish(self) -> Iterator[ParseEvent]:
        """Đóng parser. Chuỗi còn dở vẫn được phát ra như giá trị hoàn tất.

        Cần thiết khi LLM bị cắt vì `n_predict` — thà giao bản dịch cụt còn hơn
        vứt toàn bộ.
        """
        if self._state is _S.STRING and not self._is_key:
            self.malformed = True
            yield ValueDone(self._child_path(), "".join(self._buf))
            self._buf.clear()
        elif self._state is _S.LITERAL:
            yield from self._finish_literal()
        elif self._state not in (_S.DONE, _S.PREFIX):
            self.malformed = True
        self._state = _S.DONE

    # ------------------------------------------------------------------ #

    def _step(self, ch: str) -> Iterator[ParseEvent]:
        state = self._state

        if state is _S.PREFIX:
            if ch == "{":
                self._stack.append(_Frame("object"))
                self._state = _S.OBJ_KEY
            elif ch == "[":
                self._stack.append(_Frame("array", index=0))
                self._state = _S.VALUE
            return

        if state is _S.STRING:
            yield from self._step_string(ch)
            return

        if state is _S.LITERAL:
            if ch in ",}]" or ch in _WS:
                yield from self._finish_literal()
                yield from self._step(ch)      # ký tự này thuộc về ngữ cảnh cha
            else:
                self._buf.append(ch)
            return

        if ch in _WS:
            return

        if state is _S.OBJ_KEY:
            if ch == '"':
                self._is_key = True
                self._buf.clear()
                self._state = _S.STRING
            elif ch == "}":
                yield from self._close_frame()
            else:
                self.malformed = True
            return

        if state is _S.OBJ_COLON:
            if ch == ":":
                self._state = _S.VALUE
            else:
                self.malformed = True
            return

        if state is _S.VALUE:
            yield from self._step_value(ch)
            return

        if state is _S.OBJ_NEXT:
            if ch == ",":
                self._state = _S.OBJ_KEY
            elif ch == "}":
                yield from self._close_frame()
            else:
                self.malformed = True
            return

        if state is _S.ARR_NEXT:
            if ch == ",":
                self._stack[-1].index += 1
                self._state = _S.VALUE
            elif ch == "]":
                yield from self._close_frame()
            else:
                self.malformed = True
            return

    def _step_value(self, ch: str) -> Iterator[ParseEvent]:
        if ch == '"':
            self._is_key = False
            self._buf.clear()
            self._state = _S.STRING
        elif ch == "{":
            self._stack.append(_Frame("object"))
            self._state = _S.OBJ_KEY
        elif ch == "[":
            self._stack.append(_Frame("array", index=0))
            self._state = _S.VALUE
        elif ch in "]}":
            # mảng/đối tượng rỗng: `[]` hoặc `{}` — ký tự đóng đến ngay sau khi mở
            yield from self._close_frame()
        else:
            self._buf.clear()
            self._buf.append(ch)
            self._state = _S.LITERAL
        return
        yield  # pragma: no cover - giữ hàm là generator

    def _step_string(self, ch: str) -> Iterator[ParseEvent]:
        # --- đang thu thập 4 chữ số hex của \uXXXX ---
        if self._unicode is not None:
            if ch not in _HEX:
                self.malformed = True
                self._unicode = None
                self._high_surrogate = None
                return
            self._unicode.append(ch)
            if len(self._unicode) < 4:
                return
            code = int("".join(self._unicode), 16)
            self._unicode = None
            yield from self._emit_code_point(code)
            return

        if self._escape:
            self._escape = False
            if ch == "u":
                self._unicode = []
                return
            decoded = _ESCAPES.get(ch, ch)
            yield from self._append(decoded)
            return

        if ch == "\\":
            self._escape = True
            return

        if ch == '"':
            self._flush_orphan_surrogate()
            text = "".join(self._buf)
            self._buf.clear()
            if self._is_key:
                self._stack[-1].key = text
                self._is_key = False
                self._state = _S.OBJ_COLON
            else:
                yield ValueDone(self._child_path(), text)
                yield from self._after_value()
            return

        yield from self._append(ch)

    def _emit_code_point(self, code: int) -> Iterator[ParseEvent]:
        if 0xD800 <= code <= 0xDBFF:          # nửa cao — chờ nửa thấp
            self._flush_orphan_surrogate()
            self._high_surrogate = code
            return
        if 0xDC00 <= code <= 0xDFFF:          # nửa thấp
            if self._high_surrogate is None:
                self.malformed = True
                return
            combined = 0x10000 + ((self._high_surrogate - 0xD800) << 10) + (code - 0xDC00)
            self._high_surrogate = None
            yield from self._append(chr(combined))
            return
        self._flush_orphan_surrogate()
        yield from self._append(chr(code))

    def _flush_orphan_surrogate(self) -> None:
        """Nửa cao không có nửa thấp đi kèm — bỏ và đánh dấu output méo."""
        if self._high_surrogate is not None:
            self._high_surrogate = None
            self.malformed = True

    def _append(self, text: str) -> Iterator[ParseEvent]:
        self._buf.append(text)
        if self.emit_string_deltas and not self._is_key:
            yield StringDelta(self._child_path(), text)

    def _finish_literal(self) -> Iterator[ParseEvent]:
        raw = "".join(self._buf).strip()
        self._buf.clear()
        value: Any
        if raw == "true":
            value = True
        elif raw == "false":
            value = False
        elif raw == "null":
            value = None
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    self.malformed = True
                    value = raw
        yield ValueDone(self._child_path(), value)
        yield from self._after_value()

    def _after_value(self) -> Iterator[ParseEvent]:
        if not self._stack:
            self._state = _S.DONE
            return
        frame = self._stack[-1]
        self._state = _S.OBJ_NEXT if frame.kind == "object" else _S.ARR_NEXT
        return
        yield  # pragma: no cover

    def _close_frame(self) -> Iterator[ParseEvent]:
        if not self._stack:
            self._state = _S.DONE
            return
        self._stack.pop()
        if not self._stack:
            self._state = _S.DONE
        else:
            yield from self._after_value()
