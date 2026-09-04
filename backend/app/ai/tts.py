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


def _absolute_binary(program: str) -> str:
    """Trả về đường dẫn TUYỆT ĐỐI của binary.

    Không phải để cho đẹp: CPython chỉ dùng `posix_spawn()` thay cho
    `fork()+exec()` khi đường dẫn có thành phần thư mục, `close_fds=False`,
    không `start_new_session`, và không `preexec_fn`.

    Vì sao phải tránh fork(): faster-whisper kéo theo OpenMP/OpenBLAS, thư
    viện này cài `pthread_atfork` handler. Khi fork() xảy ra lúc Whisper đang
    transcribe, handler đó gọi `pthread_join` lên các worker đang tính toán và
    treo vĩnh viễn — cả tiến trình chết đứng, không lỗi, không timeout.

    Đây KHÔNG phải tình huống hiếm: pipeline thường xuyên chạy final STT của
    câu này trong lúc TTS của câu trước bắt đầu. Đã tái hiện được bằng B6.
    Stack lúc treo: fork -> _pthread_atfork_prepare_handlers -> _pthread_join
    -> __ulock_wait.

    Đánh đổi của `close_fds=False`: tiến trình con kế thừa fd đang mở. Chấp
    nhận được với một tiến trình cục bộ, ngắn hạn, do ta kiểm soát hoàn toàn.
    """
    resolved = shutil.which(program)
    if resolved:
        return str(Path(resolved).resolve())
    path = Path(program)
    if path.exists():
        return str(path.resolve())
    raise TtsUnavailable(f"Không tìm thấy binary: {program!r}")


@dataclass
class TtsJob:
    utterance_id: str
    field: str            # translation | coach
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

    #: Piper đổi được tốc độ đọc qua --length-scale. VieNeu thì không — router
    #: dựa vào cờ này để chọn engine cho chiều đọc chậm.
    supports_length_scale = True
    #: Piper không có model thường trú để nạp trước. Hâm nóng tiến trình dự
    #: phòng là việc làm SAU mỗi lượt đọc, không phải lúc mở session.
    needs_preload = False

    def __init__(self, config) -> None:
        self._config = config
        self._state = TtsState.IDLE
        self._job: TtsJob | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._cancel_event = asyncio.Event()
        self._chunks_sent = 0
        self._lock = asyncio.Lock()
        #: Một tiến trình Piper đã spawn sẵn và nạp xong model, đang chờ text.
        #:
        #: Piper nạp model NGAY LÚC KHỞI ĐỘNG, không đợi có input. Đo thật:
        #: spawn rồi gửi ngay mất 617ms tới byte đầu; spawn trước rồi mới gửi
        #: chỉ 120ms. Tức mỗi câu đang mất ~500ms chỉ để nạp lại đúng cái model
        #: vừa dùng xong.
        #:
        #: Gắn với (model, tốc độ đọc) vì cả hai là tham số dòng lệnh, không
        #: đổi được sau khi đã spawn.
        self._standby: tuple[tuple[str, float], asyncio.subprocess.Process] | None = None
        self._standby_task: asyncio.Task | None = None
        #: Lượt đọc gần nhất có dùng được tiến trình hâm nóng sẵn không.
        #: Đưa ra event để quan sát được — nếu nó luôn False thì việc hâm nóng
        #: đang vô ích mà không có gì báo.
        self.used_standby = False

    # -- trạng thái ------------------------------------------------------- #

    @property
    def sample_rate(self) -> int:
        return self._config.tts.sample_rate

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
        length_scale: float | None = None,
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
        scale = (
            length_scale if length_scale is not None
            else self._config.tts.length_scale
        )
        cmd = self._command(model, scale)

        key = (str(model), scale)
        self.used_standby = False
        try:
            self._process = self._take_standby(key) or await self._spawn(cmd)
        except OSError as exc:
            self._transition(TtsState.ERROR)
            raise TtsUnavailable(f"Không khởi động được Piper: {exc}") from exc

        process = self._process
        assert process.stdin is not None and process.stdout is not None

        # bytes cho mỗi chunk: PCM16 mono
        chunk_bytes = max(
            2, int(self._config.tts.sample_rate * self._config.tts.chunk_ms / 1000) * 2
        )

        pacing = self._config.tts.realtime_pacing
        lead_s = self._config.tts.pacing_lead_ms / 1000.0
        bytes_per_second = self._config.tts.sample_rate * 2   # PCM16 mono
        emitted_bytes = 0
        stream_started: float | None = None

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
                    stream_started = time.monotonic()
                    async with self._lock:
                        if self._state is TtsState.SYNTHESIZING:
                            self._transition(TtsState.PLAYING)

                self._chunks_sent += 1
                if on_chunk is not None:
                    on_chunk(data, self._chunks_sent)
                yield data
                emitted_bytes += len(data)

                if pacing and stream_started is not None:
                    # Giữ lượng audio đã đẩy đi không vượt thời gian thực quá
                    # `lead_s`. Chờ trên cờ hủy chứ không sleep suông, để
                    # Barge-in vẫn cắt được ngay giữa nhịp chờ.
                    audio_s = emitted_bytes / bytes_per_second
                    ahead = audio_s - (time.monotonic() - stream_started) - lead_s
                    if ahead > 0:
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(
                                self._cancel_event.wait(), timeout=ahead
                            )
                        if self._cancel_event.is_set():
                            break

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

    # -- tiến trình Piper hâm nóng sẵn ------------------------------------ #

    def _command(self, model: Path, scale: float) -> list[str]:
        return [
            _absolute_binary(self._config.paths.piper_bin),
            "--model", str(model),
            "--output-raw",
            # Số càng lớn đọc càng chậm. Chiều dịch ngược dùng giá trị lớn hơn
            # vì người dùng phải nói theo, không chỉ nghe hiểu.
            "--length-scale", str(scale),
        ]

    async def _spawn(self, cmd: list[str]) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # BẮT BUỘC — xem _absolute_binary(): ép CPython dùng
            # posix_spawn() thay vì fork()+exec().
            close_fds=False,
        )

    def _take_standby(self, key: tuple[str, float]) -> asyncio.subprocess.Process | None:
        """Lấy tiến trình đã hâm nóng nếu nó đúng giọng và đúng tốc độ."""
        standby, self._standby = self._standby, None
        if standby is None:
            return None
        standby_key, process = standby
        if standby_key != key or process.returncode is not None:
            # Sai giọng/tốc độ, hoặc nó đã chết. Dọn đi, spawn cái mới.
            self._kill_quietly(process)
            return None
        self.used_standby = True
        logger.debug("TTS dùng tiến trình đã hâm nóng sẵn")
        return process

    def prewarm(self, voice: str | None = None, length_scale: float | None = None) -> None:
        """Spawn sẵn tiến trình Piper cho lượt đọc kế tiếp.

        Gọi sau khi đọc xong: lượt sau gần như luôn cùng giọng và cùng tốc độ,
        nên model đã nằm sẵn trong bộ nhớ và byte đầu ra nhanh hơn ~500ms.
        Không chờ — spawn chạy nền, hỏng thì lượt sau chỉ quay về đường cũ.
        """
        if self._standby is not None or self._standby_task is not None:
            return
        try:
            model = self.resolve_voice(voice)
        except Exception:
            return
        scale = (
            length_scale if length_scale is not None
            else self._config.tts.length_scale
        )
        key = (str(model), scale)

        async def run() -> None:
            try:
                process = await self._spawn(self._command(model, scale))
            except OSError:
                return
            finally:
                self._standby_task = None
            self._standby = (key, process)

        self._standby_task = asyncio.create_task(run(), name="piper-prewarm")

    def _kill_quietly(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, Exception):
            process.kill()

    def _drop_standby(self) -> None:
        if self._standby_task is not None and not self._standby_task.done():
            self._standby_task.cancel()
        self._standby_task = None
        standby, self._standby = self._standby, None
        if standby is not None:
            self._kill_quietly(standby[1])

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

        # Đóng transport ngay thay vì để bộ thu gom rác lo.
        #
        # Ta thoát vòng đọc giữa chừng khi Barge-in, nên pipe stdout vẫn mở.
        # asyncio chỉ đóng nó lúc transport bị GC — và nếu điều đó xảy ra sau
        # khi event loop đã đóng thì __del__ ném "RuntimeError: Event loop is
        # closed". Vô hại, nhưng đủ ồn để che một lỗi thật ở lần chạy sau.
        #
        # Không có API công khai cho việc này; `_transport` được bọc trong
        # getattr + suppress để bản Python nào không có nó thì chỉ quay lại
        # hành vi cũ chứ không hỏng.
        transport = getattr(process, "_transport", None)
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.close()
        await asyncio.sleep(0)

    async def close(self) -> None:
        self._cancel_event.set()
        self._drop_standby()
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
