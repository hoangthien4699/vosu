"""Silero VAD (Task D1).

Phát hiện `speech_started` / `speech_ended` (VAD endpoint) và cung cấp
`speech_probability` để:
  - `audio/chunker.py` biết khi nào speech probability bắt đầu suy giảm
    (dấu hiệu sắp dứt câu — §2.2 review v3.0), và
  - `ai/tts.py` nhận tín hiệu Barge-in (§2.4.1).

Hai backend:
  - `silero`  : ONNX, chính xác, cần models/silero_vad.onnx
  - `energy`  : fallback RMS thuần numpy, không cần model — dùng cho unit test
                và cho máy chưa tải model. KHÔNG dùng để chốt benchmark.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

logger = logging.getLogger(__name__)

INT16_SCALE = 32768.0


class VadEventType(str, Enum):
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"


@dataclass(frozen=True)
class VadEvent:
    type: VadEventType
    #: vị trí trong stream, tính bằng giây kể từ khi bắt đầu session
    timestamp_s: float
    probability: float
    #: chỉ có với SPEECH_ENDED — độ dài đoạn speech vừa kết thúc
    speech_duration_s: float = 0.0
    #: chi phí tính toán để ra quyết định này (đo cho Benchmark Gate mục 1)
    compute_ms: float = 0.0


@dataclass(frozen=True)
class VadFrame:
    probability: float
    is_speech: bool
    timestamp_s: float
    compute_ms: float


class VadBackend(Protocol):
    frame_samples: int

    def probability(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class SileroVadBackend:
    """Silero VAD qua onnxruntime. Tự thích ứng cả checkpoint v4 và v5."""

    def __init__(self, model_path: Path, sample_rate: int = 16_000) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Cần onnxruntime cho Silero VAD: pip install onnxruntime"
            ) from exc

        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model Silero VAD tại {model_path}. "
                "Chạy `scripts/download_models.sh` hoặc đặt vad.backend=energy."
            )

        opts = ort.SessionOptions()
        # VAD chạy trên từng frame 32ms — nhiều thread chỉ tổ tạo overhead và
        # tranh chấp CPU với TTS (§2.4, CPU contention).
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        self.sample_rate = sample_rate
        self.frame_samples = 512 if sample_rate == 16_000 else 256

        input_names = {i.name for i in self._session.get_inputs()}
        # v5: input/state/sr ; v4: input/sr/h/c
        self._is_v5 = "state" in input_names
        self._state_dim = 128 if self._is_v5 else 64
        self.reset()
        logger.info(
            "Silero VAD đã nạp (%s, frame=%d sample)",
            "v5" if self._is_v5 else "v4",
            self.frame_samples,
        )

    def reset(self) -> None:
        shape = (2, 1, self._state_dim)
        if self._is_v5:
            self._state = np.zeros(shape, dtype=np.float32)
        else:
            self._h = np.zeros(shape, dtype=np.float32)
            self._c = np.zeros(shape, dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        x = frame.reshape(1, -1).astype(np.float32)
        sr = np.array(self.sample_rate, dtype=np.int64)
        if self._is_v5:
            out, self._state = self._session.run(
                None, {"input": x, "state": self._state, "sr": sr}
            )
        else:
            out, self._h, self._c = self._session.run(
                None, {"input": x, "sr": sr, "h": self._h, "c": self._c}
            )
        return float(np.asarray(out).ravel()[0])


class EnergyVadBackend:
    """Fallback RMS + adaptive noise floor. Không cần model, không dùng cho benchmark."""

    def __init__(self, sample_rate: int = 16_000, frame_samples: int = 512) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.reset()

    def reset(self) -> None:
        self._noise_rms = 1e-3
        self._warm = 0

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame))) + 1e-9)
        # 10 frame đầu (~320ms) dùng để ước lượng nền nhiễu.
        if self._warm < 10:
            self._warm += 1
            self._noise_rms = 0.7 * self._noise_rms + 0.3 * rms
            return 0.0
        snr = rms / max(self._noise_rms, 1e-5)
        if snr < 2.0:  # coi là im lặng -> cập nhật nền nhiễu chậm
            self._noise_rms = 0.98 * self._noise_rms + 0.02 * rms
        # map SNR 2..8 -> 0..1
        return float(np.clip((snr - 2.0) / 6.0, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Processor: probability -> speech_started / speech_ended
# --------------------------------------------------------------------------- #


class VadProcessor:
    """Biến chuỗi probability theo frame thành sự kiện đầu/cuối câu.

    `min_silence_ms` là khoảng im lặng cần quan sát trước khi khẳng định người
    nói đã dứt câu. Đặc tả §7: 300–500ms, và khoảng này **cộng thẳng vào E2E**.
    Đặt quá ngắn (150ms) sẽ chốt câu nhầm khi người nói ngập ngừng ("uhm").
    """

    def __init__(
        self,
        backend: VadBackend,
        *,
        sample_rate: int = 16_000,
        threshold: float = 0.5,
        min_silence_ms: int = 400,
        min_speech_ms: int = 200,
        decay_ratio: float = 0.6,
    ) -> None:
        self._backend = backend
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.frame_samples = backend.frame_samples
        self.frame_ms = 1000.0 * self.frame_samples / sample_rate
        self._min_silence_frames = max(1, round(min_silence_ms / self.frame_ms))
        self._min_speech_frames = max(1, round(min_speech_ms / self.frame_ms))
        self._decay_ratio = decay_ratio
        self._tail = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._backend.reset()
        self._triggered = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._samples_seen = 0
        self._speech_start_s = 0.0
        self._peak_probability = 0.0
        self._last_probability = 0.0
        self._tail = np.zeros(0, dtype=np.float32)

    # -- thuộc tính quan sát được từ ngoài -------------------------------- #

    @property
    def is_speaking(self) -> bool:
        return self._triggered

    @property
    def last_probability(self) -> float:
        return self._last_probability

    @property
    def is_decaying(self) -> bool:
        """Speech probability đang suy giảm — dấu hiệu sắp dứt câu (§2.2)."""
        if not self._triggered or self._peak_probability <= 0:
            return False
        return self._last_probability < self._peak_probability * self._decay_ratio

    @property
    def current_speech_duration_s(self) -> float:
        if not self._triggered:
            return 0.0
        return self._samples_seen / self.sample_rate - self._speech_start_s

    # -- nạp dữ liệu ------------------------------------------------------ #

    def feed(self, pcm: np.ndarray) -> Iterator[VadEvent]:
        """Nạp PCM float32 [-1,1] độ dài bất kỳ; sinh sự kiện VAD."""
        buf = np.concatenate([self._tail, pcm]) if self._tail.size else pcm
        n = self.frame_samples
        total = (buf.size // n) * n
        for offset in range(0, total, n):
            yield from self._process_frame(buf[offset : offset + n])
        self._tail = buf[total:].copy()

    def feed_bytes(self, pcm16: bytes) -> Iterator[VadEvent]:
        return self.feed(pcm16_to_float32(pcm16))

    def iter_frames(
        self, pcm: np.ndarray
    ) -> Iterator[tuple[np.ndarray, list[VadEvent]]]:
        """Như `feed` nhưng trả kèm frame sinh ra event.

        `audio/chunker.py` cần biết event thuộc frame nào để cắt audio đúng
        biên — `feed()` gộp hết event của cả buffer nên không đủ thông tin.
        """
        buf = np.concatenate([self._tail, pcm]) if self._tail.size else pcm
        n = self.frame_samples
        total = (buf.size // n) * n
        for offset in range(0, total, n):
            frame = buf[offset : offset + n]
            yield frame, list(self._process_frame(frame))
        self._tail = buf[total:].copy()

    def _process_frame(self, frame: np.ndarray) -> Iterator[VadEvent]:
        t0 = time.perf_counter()
        probability = self._backend.probability(frame)
        compute_ms = (time.perf_counter() - t0) * 1000.0

        self._samples_seen += frame.size
        now_s = self._samples_seen / self.sample_rate
        self._last_probability = probability

        if not self._triggered:
            if probability >= self.threshold:
                self._speech_frames += 1
                if self._speech_frames >= self._min_speech_frames:
                    self._triggered = True
                    self._silence_frames = 0
                    self._peak_probability = probability
                    # lùi lại đúng số frame đã tích lũy để không mất đầu câu
                    self._speech_start_s = now_s - self._speech_frames * self.frame_ms / 1000.0
                    yield VadEvent(
                        VadEventType.SPEECH_STARTED,
                        timestamp_s=self._speech_start_s,
                        probability=probability,
                        compute_ms=compute_ms,
                    )
            else:
                self._speech_frames = 0
            return

        # đang trong speech
        self._peak_probability = max(self._peak_probability, probability)
        if probability >= self.threshold:
            self._silence_frames = 0
            return

        self._silence_frames += 1
        if self._silence_frames < self._min_silence_frames:
            return

        # --- VAD endpoint: điều kiện BẮT BUỘC để chạy final STT (§2.2) ---
        silence_s = self._silence_frames * self.frame_ms / 1000.0
        duration = now_s - silence_s - self._speech_start_s
        yield VadEvent(
            VadEventType.SPEECH_ENDED,
            timestamp_s=now_s,
            probability=probability,
            speech_duration_s=max(0.0, duration),
            compute_ms=compute_ms,
        )
        self._triggered = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._peak_probability = 0.0


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """PCM 16-bit little-endian -> float32 trong [-1, 1]."""
    if len(data) % 2:
        data = data[:-1]  # bỏ byte lẻ ở biên chunk
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / INT16_SCALE


def float32_to_pcm16(data: np.ndarray) -> bytes:
    clipped = np.clip(data, -1.0, 1.0)
    return (clipped * (INT16_SCALE - 1)).astype("<i2").tobytes()


def build_vad(config) -> VadProcessor:
    """Tạo VadProcessor từ `Config`. Tự fallback sang energy nếu thiếu model."""
    backend: VadBackend
    if config.vad.backend == "silero":
        try:
            backend = SileroVadBackend(
                config.paths.resolve("silero_vad_onnx"), config.audio.sample_rate
            )
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Không nạp được Silero VAD (%s) — dùng backend energy.", exc)
            backend = EnergyVadBackend(config.audio.sample_rate, config.audio.frame_samples)
    elif config.vad.backend == "energy":
        backend = EnergyVadBackend(config.audio.sample_rate, config.audio.frame_samples)
    else:
        raise ValueError(f"vad.backend không hợp lệ: {config.vad.backend!r}")

    return VadProcessor(
        backend,
        sample_rate=config.audio.sample_rate,
        threshold=config.vad.threshold,
        min_silence_ms=config.vad.min_silence_ms,
        min_speech_ms=config.vad.min_speech_ms,
        decay_ratio=config.vad.decay_ratio,
    )
