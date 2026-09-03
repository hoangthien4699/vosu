"""Khóa contract §2.2 — phần dễ code sai nhất của dự án.

Nếu một trong các test này đỏ, pipeline đã quay về mô hình "buffer cố định"
mà 5 vòng review đã loại bỏ.
"""
from __future__ import annotations

import numpy as np

from app.audio.chunker import AudioChunker, SegmentKind
from app.audio.vad import EnergyVadBackend, VadProcessor
from tests.synth import SR, silence, speech


def make_chunker(**kwargs) -> AudioChunker:
    vad = VadProcessor(EnergyVadBackend(), min_silence_ms=400, min_speech_ms=200)
    return AudioChunker(vad, **kwargs)


def feed_streaming(chunker: AudioChunker, stream: np.ndarray, chunk_ms: int = 100):
    """Nạp theo chunk 100ms giống hệt client thật gửi qua WebSocket (§4.1)."""
    n = int(SR * chunk_ms / 1000)
    out = []
    for i in range(0, len(stream), n):
        out.extend(chunker.feed(stream[i : i + n]))
    return out


def test_cau_ngan_duoi_nguong_van_co_final_stt():
    """"Yes." (0.6s < 1.5s) BẮT BUỘC vẫn phải có final STT qua VAD endpoint.

    Đây chính là trade-off ẩn mà review v4.1 yêu cầu làm rõ: nếu final STT phụ
    thuộc ngưỡng 1.5s thì câu ngắn không bao giờ được transcribe.
    """
    rng = np.random.default_rng(1)
    stream = np.concatenate([silence(0.5, rng), speech(0.6, rng), silence(1.0, rng)])

    segments = feed_streaming(make_chunker(), stream)
    finals = [s for s in segments if s.is_final]

    assert len(finals) == 1, "câu ngắn phải sinh đúng 1 final segment"
    assert finals[0].trigger == "vad_endpoint"
    assert not [s for s in segments if s.kind is SegmentKind.PARTIAL], (
        "câu 0.6s chưa đạt cửa sổ tối thiểu nên KHÔNG được có partial"
    )


def test_final_luon_do_vad_endpoint_kich_hoat_khong_phai_nguong_1_5s():
    rng = np.random.default_rng(2)
    for duration in (0.4, 0.9, 1.4, 1.6, 3.0):
        stream = np.concatenate(
            [silence(0.4, rng), speech(duration, rng), silence(1.0, rng)]
        )
        finals = [s for s in feed_streaming(make_chunker(), stream) if s.is_final]
        assert len(finals) == 1, f"độ dài {duration}s không sinh đúng 1 final"
        assert finals[0].trigger == "vad_endpoint"


def test_cau_dai_co_partial_truoc_final():
    rng = np.random.default_rng(3)
    stream = np.concatenate([silence(0.4, rng), speech(4.0, rng), silence(1.0, rng)])

    segments = feed_streaming(make_chunker(), stream)
    partials = [s for s in segments if s.kind is SegmentKind.PARTIAL]
    finals = [s for s in segments if s.is_final]

    assert partials, "câu 4s phải có ít nhất một partial"
    assert len(finals) == 1
    kinds = [s.kind for s in segments]
    assert kinds[-1] is SegmentKind.FINAL, "final phải là segment cuối cùng"
    # partial phải lớn dần (sliding window mở rộng), không phải cửa sổ cố định
    sizes = [s.duration_s for s in partials]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_cooldown_chan_sliding_window_qua_day():
    """§2.2 review v3.0 — transcribe lại 3-5 lần/giây sẽ đẩy GPU 6GB lên 100%."""
    rng = np.random.default_rng(4)
    stream = np.concatenate([silence(0.4, rng), speech(6.0, rng), silence(1.0, rng)])

    segments = feed_streaming(make_chunker(partial_cooldown_s=0.8), stream)
    partials = [s for s in segments if s.kind is SegmentKind.PARTIAL]

    assert len(partials) >= 2
    # 6s speech, cooldown 0.8s -> tối đa ~6 partial. Nếu vượt xa là cooldown hỏng.
    assert len(partials) <= 7, f"quá nhiều partial ({len(partials)}) — cooldown không hiệu lực"


def test_nhieu_utterance_lien_tiep_khong_dinh_nhau():
    rng = np.random.default_rng(5)
    stream = np.concatenate([
        silence(0.4, rng), speech(0.8, rng),
        silence(1.0, rng), speech(0.7, rng),
        silence(1.0, rng), speech(0.9, rng),
        silence(1.0, rng),
    ])
    finals = [s for s in feed_streaming(make_chunker(), stream) if s.is_final]
    assert len(finals) == 3, f"kỳ vọng 3 utterance, nhận {len(finals)}"


def test_preroll_giu_duoc_dau_cau():
    """Không có pre-roll thì phụ âm đầu bị cắt mất -> Whisper nghe hụt."""
    rng = np.random.default_rng(6)
    speech_s = 1.0
    stream = np.concatenate([silence(0.5, rng), speech(speech_s, rng), silence(1.0, rng)])

    finals = [s for s in feed_streaming(make_chunker(), stream) if s.is_final]
    assert finals[0].duration_s >= speech_s, (
        f"segment {finals[0].duration_s:.2f}s ngắn hơn speech {speech_s}s — mất đầu câu"
    )


def test_enable_partial_false_van_co_final():
    rng = np.random.default_rng(7)
    stream = np.concatenate([silence(0.4, rng), speech(4.0, rng), silence(1.0, rng)])

    segments = feed_streaming(make_chunker(enable_partial=False), stream)
    assert not [s for s in segments if s.kind is SegmentKind.PARTIAL]
    assert len([s for s in segments if s.is_final]) == 1


def test_max_utterance_cuong_buc_final_khi_vad_khong_chot_cau():
    rng = np.random.default_rng(8)
    stream = np.concatenate([silence(0.4, rng), speech(8.0, rng)])

    finals = [s for s in feed_streaming(make_chunker(max_utterance_s=3.0), stream) if s.is_final]
    assert finals and finals[0].trigger == "max_duration"
