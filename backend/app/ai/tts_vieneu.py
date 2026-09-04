"""Engine TTS thứ hai: VieNeu-TTS v3 Turbo, chạy qua TIẾN TRÌNH PHỤ.

VÌ SAO THÊM ENGINE: Piper nhanh nhưng giọng máy móc và chỉ có một giọng. Đo
trên chính máy này, cùng ba câu:

    tiếng đầu (ấm)     Piper 120ms   VieNeu  91ms
    tiếng đầu (nguội)  Piper 617ms   VieNeu  91ms
    RTF                Piper 0.047   VieNeu 0.164  (6.1x thời gian thực)
    giọng              1             20 (Bắc/Trung/Nam, có phong cách kể chuyện)
    giấy phép          MIT           Apache-2.0

VÌ SAO TIẾN TRÌNH PHỤ, KHÔNG IMPORT THẲNG: gói `vieneu` phụ thuộc CỨNG vào
`gradio` và `librosa`. Cài chung vào venv của server sẽ nâng cấp FastAPI
0.115.6 -> 0.141.1 và thêm 53 gói — làm rung chính khung đang chạy. Dự án vốn
đã chạy `llama-server` và `piper` như tiến trình ngoài; đây là cùng một khuôn,
chỉ khác là tiến trình này SỐNG LÂU nên chỉ nạp model một lần.

GIỚI HẠN ĐÃ BIẾT: không có tham số tốc độ đọc. Chiều dịch ngược (đọc chậm để
người dùng nói theo) vì vậy vẫn phải đi Piper — xem `tts_router.py`.

Model là loại lấy mẫu (temperature/top_k/top_p) nên cùng một câu đọc hai lần
không giống hệt nhau. Đừng dùng nó cho phép đo cần lặp lại được.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from .tts import _ALLOWED, CancelResult, TtsJob, TtsState, TtsUnavailable

logger = logging.getLogger(__name__)

#: Model sinh ở tần số này, không đổi được.
SAMPLE_RATE = 48_000
#: Nạp model mất vài giây; đo được 5.8s trên Mac.
_START_TIMEOUT_S = 90.0


class VieNeuTts:
    """Cùng bề mặt với `PiperTts` để hai engine thay nhau được."""

    supports_length_scale = False
    #: Nạp model mất ~8.8s — phải bắt đầu ngay từ lúc mở session.
    needs_preload = True

    def __init__(self, config) -> None:
        self._config = config
        self._state = TtsState.IDLE
        self._job: TtsJob | None = None
        self._cancel_event = asyncio.Event()
        self._chunks_sent = 0
        self._lock = asyncio.Lock()
        self._proc: asyncio.subprocess.Process | None = None
        #: Bị hủy giữa chừng thì tiến trình phụ vẫn sinh nốt — phải đọc bỏ tới
        #: `END` trước khi gửi câu sau, nếu không audio hai câu sẽ dính nhau.
        self._needs_drain = False
        #: Có ý nghĩa với Piper (tiến trình hâm nóng sẵn). Ở đây model luôn
        #: nằm sẵn nên coi như luôn ấm.
        self.used_standby = True

    # -- trạng thái ------------------------------------------------------- #

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

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
            raise RuntimeError(f"TTS: {self._state.value} -> {target.value} không hợp lệ")
        self._state = target

    # -- tiến trình phụ --------------------------------------------------- #

    def _interpreter(self) -> str:
        exe = self._config.paths.vieneu_python
        found = shutil.which(exe) or (exe if Path(exe).exists() else None)
        if found is None:
            raise TtsUnavailable(
                f"Không tìm thấy Python của VieNeu: {exe!r}. "
                "Chạy scripts/setup_vieneu.sh, hoặc đặt tts.engine=piper."
            )
        return found

    def _sidecar(self) -> Path:
        script = Path(self._config.paths.vieneu_sidecar)
        if not script.is_absolute():
            from ..core.config import REPO_ROOT

            script = REPO_ROOT / script
        if not script.exists():
            raise TtsUnavailable(f"Không tìm thấy script tiến trình phụ: {script}")
        return script

    def preflight(self, voice: str | None = None) -> None:
        """Kiểm có đủ thứ để chạy chưa. KHÔNG nạp model ở đây."""
        self._interpreter()
        self._sidecar()

    def resolve_voice(self, voice: str | None = None) -> str:
        """`voice` từ tầng trên là mã ngôn ngữ ("vi"/"en") giống Piper, nên
        ánh xạ về giọng VieNeu đã cấu hình."""
        return self._config.tts.vieneu_voice

    async def _ensure_proc(self) -> asyncio.subprocess.Process:
        if self._proc is not None and self._proc.returncode is None:
            return self._proc
        started = time.perf_counter()
        self._proc = await asyncio.create_subprocess_exec(
            self._interpreter(), str(self._sidecar()),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            close_fds=False,
        )
        line = await asyncio.wait_for(
            self._proc.stdout.readline(), timeout=_START_TIMEOUT_S
        )
        if line.strip() != b"READY":
            raise TtsUnavailable(f"Tiến trình VieNeu không khởi động được: {line!r}")
        logger.info("VieNeu-TTS sẵn sàng sau %.0fms", (time.perf_counter() - started) * 1000)
        return self._proc

    async def _drain(self, proc: asyncio.subprocess.Process) -> None:
        """Đọc bỏ phần còn lại của câu trước, tới `END`."""
        while True:
            line = await proc.stdout.readline()
            if not line or line.startswith(b"END"):
                return
            if line.startswith(b"CHUNK "):
                await proc.stdout.readexactly(int(line.split()[1]))

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
        text = text.strip()
        if not text:
            return

        proc = await self._ensure_proc()
        if self._needs_drain:
            self._needs_drain = False
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._drain(proc), timeout=30.0)

        name = self.resolve_voice(voice)
        async with self._lock:
            self._cancel_event.clear()
            self._chunks_sent = 0
            self._job = TtsJob(
                utterance_id=utterance_id, field=field, text=text,
                voice=name, started_at=time.monotonic(),
            )
            if self._state in (TtsState.DONE, TtsState.INTERRUPTED, TtsState.ERROR):
                self._transition(TtsState.IDLE)
            self._transition(TtsState.SYNTHESIZING)

        payload = json.dumps({"text": text, "voice": name}, ensure_ascii=False)
        proc.stdin.write(payload.encode("utf-8") + b"\n")
        await proc.stdin.drain()

        pacing = self._config.tts.realtime_pacing
        lead_s = self._config.tts.pacing_lead_ms / 1000.0
        bytes_per_second = SAMPLE_RATE * 2
        emitted = 0
        started_at: float | None = None
        chunk_bytes = max(2, int(SAMPLE_RATE * self._config.tts.chunk_ms / 1000) * 2)
        buffer = bytearray()
        drained = False

        try:
            while True:
                if self._cancel_event.is_set():
                    self._needs_drain = True
                    break

                read = asyncio.ensure_future(proc.stdout.readline())
                stop = asyncio.ensure_future(self._cancel_event.wait())
                done, pending = await asyncio.wait(
                    {read, stop}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                if stop in done:
                    self._needs_drain = True
                    break

                line = read.result()
                if not line or line.startswith(b"END"):
                    drained = True
                    if buffer:
                        data = bytes(buffer)
                        buffer.clear()
                        self._chunks_sent += 1
                        if on_chunk is not None:
                            on_chunk(data, self._chunks_sent)
                        yield data
                    break
                if line.startswith(b"ERR"):
                    self._transition(TtsState.ERROR)
                    raise TtsUnavailable(line.decode("utf-8", "replace").strip())
                if not line.startswith(b"CHUNK "):
                    continue

                block = await proc.stdout.readexactly(int(line.split()[1]))
                buffer += block
                # Tiến trình phụ trả mẩu ~320ms; cắt lại theo `chunk_ms` cho
                # ngang Piper. Mẩu to thì lúc hủy vẫn phải phát nốt cả mẩu.
                while len(buffer) >= chunk_bytes or (drained and buffer):
                    data = bytes(buffer[:chunk_bytes])
                    del buffer[:chunk_bytes]

                    if started_at is None:
                        started_at = time.monotonic()
                        async with self._lock:
                            if self._state is TtsState.SYNTHESIZING:
                                self._transition(TtsState.PLAYING)

                    self._chunks_sent += 1
                    if on_chunk is not None:
                        on_chunk(data, self._chunks_sent)
                    yield data
                    emitted += len(data)

                    if pacing:
                        # Giữ lượng audio đã đẩy đi không vượt thời gian thực
                        # quá `lead_s`. Chờ trên cờ hủy chứ không sleep suông.
                        ahead = (emitted / bytes_per_second) - (time.monotonic() - started_at) - lead_s
                        if ahead > 0:
                            with contextlib.suppress(asyncio.TimeoutError):
                                await asyncio.wait_for(self._cancel_event.wait(), timeout=ahead)
                            if self._cancel_event.is_set():
                                self._needs_drain = True
                                break
                if self._cancel_event.is_set():
                    self._needs_drain = True
                    break
        finally:
            async with self._lock:
                if self._state in (TtsState.SYNTHESIZING, TtsState.PLAYING):
                    self._transition(TtsState.DONE)

    # -- hủy -------------------------------------------------------------- #

    async def cancel(self, reason: str = "barge_in") -> CancelResult:
        signalled_at = time.monotonic()
        if not self.is_active:
            return CancelResult(False, 0.0, self._chunks_sent, reason)
        self._cancel_event.set()
        async with self._lock:
            if self._state in (TtsState.SYNTHESIZING, TtsState.PLAYING):
                self._transition(TtsState.INTERRUPTED)
        response_ms = (time.monotonic() - signalled_at) * 1000.0
        logger.info("TTS bị hủy (%s) sau %.1fms, đã gửi %d chunk",
                    reason, response_ms, self._chunks_sent)
        return CancelResult(True, response_ms, self._chunks_sent, reason)

    def prewarm(self, voice: str | None = None, length_scale: float | None = None) -> None:
        """Khởi động tiến trình phụ ở NỀN, không chờ.

        Nạp model mất ~8.8s. Để nó xảy ra lúc câu đầu tiên cần đọc thì câu đó
        trễ thêm đúng ngần ấy — đã đo thấy: câu đầu 11.8s, các câu sau ~3s.
        Gọi lúc mở session thì tới lúc bản dịch đầu tiên xong (~6s) nó đã sẵn.
        """
        if self._proc is not None and self._proc.returncode is None:
            return
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_running_loop()
            loop.create_task(self._warm())

    async def _warm(self) -> None:
        with contextlib.suppress(Exception):
            await self._ensure_proc()

    async def close(self) -> None:
        self._cancel_event.set()
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(Exception):
            proc.stdin.close()
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(proc.wait(), timeout=3.0)
