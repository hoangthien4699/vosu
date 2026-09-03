"""Audio mẫu cho benchmark.

Ưu tiên file thật trong `benchmarks/audio/` (chất lượng đo tốt hơn nhiều).
Nếu chưa có thì sinh tín hiệu tổng hợp để script vẫn chạy được ngay — nhưng
kết quả STT trên tín hiệu tổng hợp KHÔNG dùng để đánh giá WER.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 16_000
AUDIO_DIR = Path(__file__).resolve().parent / "audio"


def list_samples() -> list[Path]:
    if not AUDIO_DIR.exists():
        return []
    return sorted(p for p in AUDIO_DIR.glob("*.wav"))


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SR or handle.getnchannels() != 1:
            raise ValueError(
                f"{path.name}: cần WAV 16kHz mono, đang là "
                f"{handle.getframerate()}Hz {handle.getnchannels()}ch"
            )
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def synthetic_speech(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """Tín hiệu giống giọng nói: sóng hài + đường bao formant + nhiễu."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    f0 = 130.0 + 25.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    signal = sum(0.36 / (k + 1) * np.sin((k + 1) * phase) for k in range(5))
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 3.1 * t) ** 2
    return ((signal * envelope) + 0.02 * rng.normal(0, 1, t.size)).astype(np.float32)


def get_samples(count: int, seconds: float = 2.0) -> list[tuple[str, np.ndarray]]:
    real = list_samples()
    if real:
        out = []
        for i in range(count):
            path = real[i % len(real)]
            out.append((path.name, load_wav(path)))
        return out
    return [
        (f"synthetic_{i:02d}", synthetic_speech(seconds, seed=i)) for i in range(count)
    ]


def uses_synthetic() -> bool:
    return not list_samples()


SYNTHETIC_WARNING = (
    "Đang dùng audio TỔNG HỢP (chưa có file nào trong benchmarks/audio/). "
    "Số đo latency vẫn hợp lệ, nhưng transcript/WER thì KHÔNG. "
    "Đặt file WAV 16kHz mono vào benchmarks/audio/ trước khi chốt Gate."
)
