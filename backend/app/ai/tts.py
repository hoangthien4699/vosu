"""Piper TTS trên CPU + Barge-in state machine (Task E5, E7, E8).

Hai quyết định kiến trúc đã chốt:

1. **CPU, không GPU** (§2.4). Whisper + Qwen đã chiếm ~5.1GB/6GB VRAM. Thêm TTS
   lên GPU sẽ vượt trần, hoặc tạo tranh chấp SM 3 chiều — nghiêm trọng hơn
   nhiều so với tranh chấp 2 chiều đã cảnh báo ở v3.0.

2. **Worker/process riêng** (§2.4, review v4.1). "async" trên danh nghĩa là chưa
   đủ: gọi hàm blocking thẳng trong async route vẫn chặn event loop. Piper chạy
   như subprocess, đọc stdout không đồng bộ.

Barge-in (§2.4.1): khi VAD phát hiện speech mới lúc state = PLAYING, phải dừng
phát trong < 200ms. Ta không cần AEC thật ở MVP — chỉ cần phát hiện + hủy kịp
thời để Whisper không nhận nhầm audio TTS làm speech input mới.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class TtsState(str, Enum):
    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    PLAYING = "playing"
    INTERRUPTED = "interrupted"
    DONE = "done"
    ERROR = "error"


_ALLOWED: dict[TtsState, frozenset[TtsState]] = {
    TtsState.IDLE: frozenset({TtsState.SYNTHESIZING, TtsState.ERROR}),
    TtsState.SYNTHESIZING: frozenset(
        {TtsState.PLAYING, TtsState.INTERRUPTED, TtsState.DONE, TtsState.ERROR}
    ),
    TtsState.PLAYING: frozenset({TtsState.INTERRUPTED, TtsState.DONE, TtsState.ERROR}),
    TtsState.INTERRUPTED: frozenset({TtsState.IDLE, TtsState.SYNTHESIZING}),
    TtsState.DONE: frozenset({TtsState.IDLE, TtsState.SYNTHESIZING}),
    TtsState.ERROR: frozenset({TtsState.IDLE, TtsState.SYNTHESIZING}),
}


class TtsUnavailable(RuntimeError):
    """Không tìm thấy binary hoặc voice model của Piper."""


@dataclass
class TtsJob:
    utterance_id: str
    field: str            # translation | intent | reply
    text: str
    voice: str
    started_at: float


@dataclass
class CancelResult:
    cancelled: bool
    #: §2.4.1 / B8 — từ lúc nhận tín hiệu tới lúc thực sự ngừng phát
    response_ms: float
    chunks_sent: int
    reason: str


class PiperTts:
    """Một engine TTS cho một session. Mỗi lúc chỉ đọc một job."""

    def __init__(self, config) -> None:
        self._config = config
        self._state = TtsState.IDLE
        self._job: TtsJob | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._cancel_event = asyncio.Event()
        self._chunks_sent = 0
        self._lock = asyncio.Lock()

    # -- trạng thái ------------------------------------------------------- #

    @property
    def state(self) -> TtsState:
        return self._state

    @property
    def current_job(self) -> TtsJob | None:
        return self._job

    @property
    def is_active(self) -> bool:
        return self._state in (TtsState.SYNTHESIZING, TtsState.PLAYING)

    def _transition(self, target: TtsState) -> None:
        if target is self._state:
            return
        if target not in _ALLOWED.get(self._state, frozenset()):
            raise RuntimeError(
                f"TTS: {self._state.value} -> {target.value} không hợp lệ"
            )
        logger.debug("TTS %s -> %s", self._state.value, target.value)
        self._state = target

    # -- kiểm tra khả dụng ------------------------------------------------ #

    def resolve_voice(self, voice: str | None = None) -> Path:
        name = voice or self._config.tts.voice
        key = "piper_voice_en" if name.startswith("en") else "piper_voice_vi"
        return self._config.paths.resolve(key)

    def preflight(self, voice: str | None = None) -> None:
        """Kiểm tra binary + voice trước khi nhận job. Raise nếu thiếu."""
        binary = self._config.paths.piper_bin
        if shutil.which(binary) is None and not Path(binary).exists():
            raise TtsUnavailable(
                f"Không tìm thấy binary Piper: {binary!r}. "
                "Cài `brew install piper` (macOS) hoặc `pip install piper-tts`, "
                "hoặc đặt tts.enabled=false."
            )
        model = self.resolve_voice(voice)
        if not model.exists():
            raise TtsUnavailable(
                f"Không tìm thấy voice model: {model}. Chạy scripts/download_models.sh."
            )

    # -- tổng hợp --------------------------------------------------------- #

    async def synthesize(
        self,
        utterance_id: str,
        text: str,
        *,
        field: str = "translation",
        voice: str | None = None,
        on_chunk: Callable[[bytes, int], None] | None = None,
    ) -> AsyncIterator[bytes]:
        """Sinh audio PCM theo chunk. Dừng ngay khi `cancel()` được gọi."""
        text = text.strip()
        if not text:
            return

        async with self._lock:
            self.preflight(voice)
            self._cancel_event.clear()
            self._chunks_sent = 0
            self._job = TtsJob(
                utterance_id=utterance_id,
                field=field,
                text=text,
                voice=voice or self._config.tts.voice,
                started_at=time.monotonic(),
            )
            if self._state in (TtsState.DONE, TtsState.INTERRUPTED, TtsState.ERROR):
                self._transition(TtsState.IDLE)
            self._transition(TtsState.SYNTHESIZING)

        model = self.resolve_voice(voice)
        cmd = [
            self._config.paths.piper_bin,
            "--model", str(model),
            "--output-raw",
            "--length-scale", str(self._config.tts.length_scale),
        ]

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            self._transition(TtsState.ERROR)
            raise TtsUnavailable(f"Không khởi động được Piper: {exc}") from exc

        process = self._process
        assert process.stdin is not None and process.stdout is not None

        # bytes cho mỗi chunk: PCM16 mono
        chunk_bytes = max(
            2, int(self._config.tts.sample_rate * self._config.tts.chunk_ms / 1000) * 2
        )

        try:
            process.stdin.write(text.encode("utf-8") + b"\n")
            await process.stdin.drain()
            process.stdin.close()

            first = True
            while True:
                if self._cancel_event.is_set():
                    break
                read_task = asyncio.ensure_future(process.stdout.read(chunk_bytes))
                cancel_task = asyncio.ensure_future(self._cancel_event.wait())
                done, pending = await asyncio.wait(
                    {read_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                if cancel_task in done:
                    break

                data = read_task.result()
                if not data:
                    break

                if first:
                    first = False
                    async with self._lock:
                        if self._state is TtsState.SYNTHESIZING:
                            self._transition(TtsState.PLAYING)

                self._chunks_sent += 1
                if on_chunk is not None:
                    on_chunk(data, self._chunks_sent)
                yield data

        finally:
            await self._terminate_process()
            async with self._lock:
                if self._state in (TtsState.SYNTHESIZING, TtsState.PLAYING):
                    self._transition(TtsState.DONE)

    # -- Barge-in --------------------------------------------------------- #

    async def cancel(self, reason: str = "barge_in") -> CancelResult:
        """Hủy job hiện tại. Đây là contract < 200ms của §2.4.1.

        Đặt cờ TRƯỚC khi kill process: vòng lặp đọc đang chờ trên cờ này qua
        `asyncio.wait`, nên nó thoát ngay lập tức mà không phải đợi OS thu hồi
        tiến trình. Việc kill diễn ra sau, không nằm trên đường tới hạn.
        """
        signalled_at = time.monotonic()

        if not self.is_active:
            return CancelResult(False, 0.0, self._chunks_sent, reason)

        self._cancel_event.set()

        async with self._lock:
            if self._state in (TtsState.SYNTHESIZING, TtsState.PLAYING):
                self._transition(TtsState.INTERRUPTED)

        response_ms = (time.monotonic() - signalled_at) * 1000.0
        logger.info(
            "TTS bị hủy (%s) sau %.1fms, đã gửi %d chunk",
            reason, response_ms, self._chunks_sent,
        )
        return CancelResult(True, response_ms, self._chunks_sent, reason)

    async def _terminate_process(self) -> None:
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=1.0)

    async def close(self) -> None:
        self._cancel_event.set()
        await self._terminate_process()


# --------------------------------------------------------------------------- #
# Cắt câu để streaming TTS (Task E6)
# --------------------------------------------------------------------------- #

_SENTENCE_END = ".!?…。！？"


class SentenceSplitter:
    """Gom text delta thành câu/cụm hoàn chỉnh để bắt đầu đọc sớm (§2.4).

    Đợi cả JSON hoàn tất rồi mới tổng hợp giọng nói là bỏ phí toàn bộ lợi ích
    của LLM streaming — time-to-first-audio sẽ bằng tổng thời gian sinh.
    """

    def __init__(self, min_chars: int = 12) -> None:
        self.min_chars = min_chars
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        out: list[str] = []
        search_from = 0
        while True:
            index = self._first_boundary(self._buffer, search_from)
            if index is None:
                break
            candidate = self._buffer[: index + 1].strip()
            if len(candidate) < self.min_chars:
                # Câu quá ngắn ("Vâng.") — gộp với câu kế tiếp thay vì cắt vụn
                # giọng đọc. Tìm tiếp ranh giới SAU, không dừng: dừng ở đây sẽ
                # khiến câu ngắn kẹt lại vĩnh viễn và không bao giờ được đọc.
                search_from = index + 1
                continue
            out.append(candidate)
            self._buffer = self._buffer[index + 1 :]
            search_from = 0
        return out

    def _first_boundary(self, text: str, start: int = 0) -> int | None:
        for i in range(start, len(text)):
            ch = text[i]
            if ch not in _SENTENCE_END:
                continue
            # "3.5" hay "v.v." — dấu chấm giữa chữ số/chữ cái không phải hết câu
            if ch == "." and i + 1 < len(text) and text[i + 1].isalnum():
                continue
            return i
        return None

    def flush(self) -> str:
        text, self._buffer = self._buffer.strip(), ""
        return text
