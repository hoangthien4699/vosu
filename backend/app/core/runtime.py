"""Model Runtime — hạ tầng dùng chung (Task C3).

Mental model ĐÚNG theo §3.1 (review v4.1):

    Application
       ├── Model Runtime  (shared infra: Whisper, LLM, TTS)
       └── Sessions       (consumer)

KHÔNG phải `Session -> Model Runtime` như baseline v1, nơi
`active_clients == 0` thì kill model. Cách đó tạo coupling "vòng đời client =
vòng đời model", và hỏng ngay khi Client A ngắt kết nối trong lúc Client B còn
đang xử lý.

CONTRACT BẮT BUỘC:
    "Model runtime không được terminate khi vẫn còn inference job đang dùng
     model."

Lưu ý: contract này phải đúng NGAY CẢ KHI `max_concurrent_sessions = 1`. MVP
không cần orchestration phức tạp, nhưng phải đúng mental model để mở rộng sau
này không phải viết lại từ đầu.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ..ai.llm import LlmClient
from ..ai.stt import SttEngine
from .vram_manager import LlamaServerManager, VramGuard, VramSnapshot

logger = logging.getLogger(__name__)


class SessionRejected(RuntimeError):
    """Vượt `max_concurrent_sessions` (§10 — 6GB chỉ đủ ~1 client)."""


@dataclass
class RuntimeStats:
    started_at: float | None = None
    stt_load_ms: float | None = None
    llm_start_ms: float | None = None
    sessions_served: int = 0
    inference_jobs: int = 0
    peak_concurrent_jobs: int = 0
    vram_peak_gb: float = 0.0
    notes: list[str] = field(default_factory=list)


class ModelRuntime:
    """Sở hữu model. Session chỉ là consumer, không điều khiển vòng đời."""

    def __init__(self, config) -> None:
        self._config = config
        self.stt = SttEngine(config)
        self.llm = LlmClient(config)
        self.llama = LlamaServerManager(config)
        self.vram = VramGuard(config)
        self.stats = RuntimeStats()

        self._ready = False
        self._lock = asyncio.Lock()
        self._active_sessions: set[str] = set()
        #: Số inference job đang chạy. Đây là thứ chặn shutdown, KHÔNG phải
        #: số session — một session có thể ngắt kết nối khi job còn dang dở.
        self._active_jobs = 0
        self._jobs_idle = asyncio.Event()
        self._jobs_idle.set()

    # -- trạng thái ------------------------------------------------------- #

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def active_sessions(self) -> int:
        return len(self._active_sessions)

    @property
    def active_jobs(self) -> int:
        return self._active_jobs

    def vram_snapshot(self) -> VramSnapshot:
        snapshot = self.vram.sample()
        self.stats.vram_peak_gb = self.vram.peak_gb
        return snapshot

    # -- vòng đời --------------------------------------------------------- #

    async def start(self) -> None:
        async with self._lock:
            if self._ready:
                return

            device = self._config.device
            logger.info("Khởi động Model Runtime — %s", device.describe())
            for note in device.notes:
                logger.info("  · %s", note)

            started = time.monotonic()

            llm_started = time.monotonic()
            await self.llama.start()
            await self.llm.start()
            self.stats.llm_start_ms = (time.monotonic() - llm_started) * 1000.0

            await self.stt.load()
            self.stats.stt_load_ms = self.stt.load_ms

            self.stats.started_at = started
            self._ready = True

            snapshot = self.vram_snapshot()
            logger.info(
                "Model Runtime sẵn sàng sau %.1fs — %s",
                time.monotonic() - started,
                snapshot.describe(),
            )

    async def shutdown(self, *, drain_timeout: float = 30.0) -> None:
        """Dừng runtime. CHỜ mọi inference job xong trước khi hạ model.

        Đây là chỗ contract của §3.1 được thực thi. Hạ model trong lúc còn job
        đang chạy sẽ gây segfault trong CTranslate2 hoặc trả về kết quả rác.
        """
        if self._active_jobs:
            logger.info("Chờ %d inference job hoàn tất trước khi hạ model...", self._active_jobs)
            try:
                await asyncio.wait_for(self._jobs_idle.wait(), timeout=drain_timeout)
            except asyncio.TimeoutError:
                logger.error(
                    "Còn %d job sau %.0fs — buộc phải hạ model.",
                    self._active_jobs, drain_timeout,
                )

        async with self._lock:
            if not self._ready:
                return
            self._ready = False
            await self.llm.close()
            await self.llama.stop()
            self.stt.unload()
            logger.info(
                "Model Runtime đã dừng (đã phục vụ %d session, %d job).",
                self.stats.sessions_served, self.stats.inference_jobs,
            )

    # -- session (consumer) ----------------------------------------------- #

    @contextlib.asynccontextmanager
    async def session(self, session_id: str) -> AsyncIterator[ModelRuntime]:
        """Đăng ký một session. KHÔNG hạ model khi thoát."""
        async with self._lock:
            limit = self._config.session.max_concurrent_sessions
            if len(self._active_sessions) >= limit:
                raise SessionRejected(
                    f"Đã đạt giới hạn {limit} session đồng thời. "
                    "6GB VRAM chỉ đủ phục vụ tốt cho 1 client (§10)."
                )
            self._active_sessions.add(session_id)
            self.stats.sessions_served += 1

        logger.info("Session %s bắt đầu (%d đang hoạt động)", session_id, self.active_sessions)
        try:
            yield self
        finally:
            async with self._lock:
                self._active_sessions.discard(session_id)
            logger.info(
                "Session %s kết thúc (%d đang hoạt động, %d job còn chạy)",
                session_id, self.active_sessions, self._active_jobs,
            )

    @contextlib.asynccontextmanager
    async def job(self, name: str = "inference") -> AsyncIterator[None]:
        """Đánh dấu một inference job đang dùng model.

        Model sẽ không bị hạ chừng nào còn job nào chưa thoát khối này.
        """
        self._active_jobs += 1
        self._jobs_idle.clear()
        self.stats.inference_jobs += 1
        self.stats.peak_concurrent_jobs = max(
            self.stats.peak_concurrent_jobs, self._active_jobs
        )
        try:
            yield
        finally:
            self._active_jobs -= 1
            if self._active_jobs <= 0:
                self._active_jobs = 0
                self._jobs_idle.set()

    async def health(self) -> dict:
        snapshot = self.vram_snapshot()
        return {
            "ready": self._ready,
            "platform": self._config.device.platform.value,
            "llama_server": self.llama.is_running,
            "stt_loaded": self.stt.is_loaded,
            "active_sessions": self.active_sessions,
            "active_jobs": self._active_jobs,
            "vram": {
                "available": snapshot.available,
                "used_gb": round(snapshot.used_gb, 3),
                "total_gb": round(snapshot.total_gb, 3),
                "peak_gb": round(self.vram.peak_gb, 3),
                "hard_ceiling_gb": self._config.vram.hard_ceiling_gb,
                "note": snapshot.reason,
            },
        }
