"""Wrapper Faster-Whisper (Task E1).

Đặc tả §2.2 (review v2.1): Faster-Whisper KHÔNG phải streaming stateful ASR.
Cái ta làm là **pseudo-streaming / sliding-window incremental inference** —
transcribe lại một cửa sổ audio đang mở rộng dần. Tên gọi này quan trọng để
developer không hiểu nhầm khả năng của thư viện.

Đặc tả §6 (review v4.1): inference là blocking C++ call. Gọi thẳng trong async
route sẽ chặn event loop. Vì vậy mọi lời gọi đi qua một executor MỘT thread:
một thread duy nhất còn để tuần tự hóa truy cập GPU, tránh cộng thêm tranh
chấp SM vào phần Compute Contention vốn đã có với llama-server (§3.2).
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    language_probability: float
    is_final: bool
    #: thời lượng audio đã đưa vào (giây)
    audio_s: float
    #: thời gian inference thực tế (ms) — số so với target < 400ms của §6
    latency_ms: float

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class SttEngine:
    """Bao quanh `WhisperModel`, chạy trong thread riêng."""

    def __init__(self, config) -> None:
        self._config = config
        self._model = None
        self._executor: ThreadPoolExecutor | None = None
        self._load_ms: float | None = None

    # ------------------------------------------------------------------ #

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_ms(self) -> float | None:
        return self._load_ms

    def load_sync(self) -> None:
        """Nạp model. Blocking — gọi từ thread khác hoặc lúc khởi động."""
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Chưa cài faster-whisper. "
                "Dùng requirements/cuda.txt hoặc requirements/macos.txt."
            ) from exc

        device = self._config.device
        started = time.perf_counter()
        logger.info(
            "Nạp Whisper %r trên %s/%s ...",
            self._config.paths.whisper_model,
            device.stt_device,
            device.stt_compute_type,
        )
        kwargs = {}
        if self._config.stt.cpu_threads:
            kwargs["cpu_threads"] = self._config.stt.cpu_threads
        self._model = WhisperModel(
            self._config.paths.whisper_model,
            device=device.stt_device,
            device_index=device.stt_device_index,
            compute_type=device.stt_compute_type,
            **kwargs,
        )
        self._load_ms = (time.perf_counter() - started) * 1000.0
        logger.info("Whisper sẵn sàng sau %.0fms", self._load_ms)

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._ensure_executor(), self.load_sync)

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            # max_workers=1: tuần tự hóa truy cập GPU (xem docstring module)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        return self._executor

    # ------------------------------------------------------------------ #

    def transcribe_sync(self, pcm: np.ndarray, *, is_final: bool = True) -> Transcript:
        if self._model is None:
            raise RuntimeError("SttEngine chưa được nạp — gọi load() trước.")

        cfg = self._config.stt
        started = time.perf_counter()
        segments, info = self._model.transcribe(
            pcm.astype(np.float32),
            beam_size=cfg.beam_size if is_final else cfg.partial_beam_size,
            language=cfg.language,
            # VAD đã làm ở tầng audio/ — bật lại ở đây là tính hai lần và
            # có thể cắt mất phần đầu đoạn đã được pre-roll giữ lại.
            vad_filter=cfg.vad_filter,
            condition_on_previous_text=cfg.condition_on_previous_text,
        )
        text = " ".join(segment.text for segment in segments).strip()
        latency_ms = (time.perf_counter() - started) * 1000.0

        return Transcript(
            text=text,
            language=getattr(info, "language", None),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            is_final=is_final,
            audio_s=pcm.size / self._config.audio.sample_rate,
            latency_ms=latency_ms,
        )

    async def transcribe(self, pcm: np.ndarray, *, is_final: bool = True) -> Transcript:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._ensure_executor(), lambda: self.transcribe_sync(pcm, is_final=is_final)
        )

    # ------------------------------------------------------------------ #

    def unload(self) -> None:
        """Giải phóng model + cache CUDA (§3.1)."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._model is not None:
            del self._model
            self._model = None
        _empty_cuda_cache()


def _empty_cuda_cache() -> None:
    """Dọn cache CUDA nếu có torch. Không bắt buộc — faster-whisper dùng
    CTranslate2, không cần torch. Chỉ dọn khi torch tình cờ có mặt."""
    import gc

    gc.collect()
    try:
        import torch  # type: ignore
    except ImportError:
        return
    if torch.cuda.is_available():  # pragma: no cover - phụ thuộc phần cứng
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
