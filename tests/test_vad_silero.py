"""Test hồi quy cho tích hợp Silero VAD.

Bỏ qua nếu chưa có models/silero_vad.onnx (CI không tải model). Nhưng trên máy
có model thì các test này BẮT BUỘC phải xanh — bug được khóa ở đây từng khiến
toàn bộ sản phẩm không chạy mà không ném ra lỗi nào.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from app.audio.vad import SileroVadBackend, VadEventType, VadProcessor
from app.core.config import REPO_ROOT

MODEL = REPO_ROOT / "models" / "silero_vad.onnx"
AUDIO_DIR = REPO_ROOT / "benchmarks" / "audio"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason="chưa có models/silero_vad.onnx — chạy scripts/download_models.sh",
)


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def speech_samples() -> list[Path]:
    return sorted(AUDIO_DIR.glob("*.wav")) if AUDIO_DIR.exists() else []


def test_silero_nhan_dien_duoc_giong_noi_that():
    """HỒI QUY: Silero v5 cần 64 sample context ghép vào đầu mỗi frame.

    Thiếu bước đó, model vẫn chạy và vẫn trả về số — nhưng probability luôn
    ~0.001 kể cả với giọng nói rõ ràng. Không exception, không cảnh báo: VAD
    im lặng không bao giờ kích hoạt, nên STT không bao giờ chạy, nên toàn bộ
    sản phẩm chết mà log vẫn sạch.
    """
    samples = speech_samples()
    if not samples:
        pytest.skip("chưa có audio mẫu — chạy scripts/make_test_audio.sh")

    backend = SileroVadBackend(MODEL)
    pcm = load_wav(samples[-1])          # mẫu dài nhất
    probabilities = [
        backend.probability(pcm[i : i + 512])
        for i in range(0, pcm.size - 512, 512)
    ]

    assert max(probabilities) > 0.9, (
        f"probability đỉnh chỉ {max(probabilities):.4f} trên giọng nói rõ ràng — "
        "gần như chắc chắn là lỗi context window của Silero v5"
    )
    voiced = sum(p > 0.5 for p in probabilities) / len(probabilities)
    assert voiced > 0.5, f"chỉ {voiced:.0%} số frame được coi là speech"


def test_reset_xoa_sach_context_giua_hai_utterance():
    samples = speech_samples()
    if not samples:
        pytest.skip("chưa có audio mẫu")

    backend = SileroVadBackend(MODEL)
    pcm = load_wav(samples[-1])[:512 * 20]

    first = [backend.probability(pcm[i : i + 512]) for i in range(0, pcm.size - 512, 512)]
    backend.reset()
    second = [backend.probability(pcm[i : i + 512]) for i in range(0, pcm.size - 512, 512)]

    assert first == pytest.approx(second, abs=1e-5), "reset() không đưa về trạng thái ban đầu"


def test_cau_ngan_that_van_co_vad_endpoint():
    """Contract §2.2 với Silero thật, không phải backend năng lượng."""
    samples = speech_samples()
    if not samples:
        pytest.skip("chưa có audio mẫu")

    shortest = min(samples, key=lambda p: p.stat().st_size)
    pcm = load_wav(shortest)
    duration = pcm.size / 16000
    assert duration < 1.5, f"{shortest.name} dài {duration:.2f}s — cần mẫu ngắn hơn 1.5s"

    rng = np.random.default_rng(0)

    def silence(seconds: float) -> np.ndarray:
        return rng.normal(0, 0.0005, int(16000 * seconds)).astype(np.float32)

    stream = np.concatenate([silence(0.5), pcm, silence(1.2)])

    processor = VadProcessor(SileroVadBackend(MODEL), min_silence_ms=400, min_speech_ms=200)
    events = [e for i in range(0, len(stream), 1600)
              for e in processor.feed(stream[i : i + 1600])]

    kinds = [e.type for e in events]
    assert VadEventType.SPEECH_STARTED in kinds
    assert VadEventType.SPEECH_ENDED in kinds, (
        f"câu {duration:.2f}s (< ngưỡng 1.5s) không có VAD endpoint — "
        "final STT sẽ không bao giờ chạy cho câu ngắn"
    )
