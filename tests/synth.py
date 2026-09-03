"""Sinh audio tổng hợp cho test — không cần file mẫu ngoài repo."""
from __future__ import annotations

import numpy as np

SR = 16_000


def silence(seconds: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0, 0.001, int(SR * seconds)).astype(np.float32)


def speech(seconds: float, rng: np.random.Generator, f0: float = 180.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    # sóng hài đơn giản + nhiễu -> đủ để VAD năng lượng coi là speech
    wave = 0.30 * np.sin(2 * np.pi * f0 * t) + 0.12 * np.sin(2 * np.pi * 2 * f0 * t)
    return (wave + 0.05 * rng.normal(0, 1, t.size)).astype(np.float32)
