"""Chọn engine TTS: lượt nào cần đọc chậm thì phải đi Piper.

Chiều dịch ngược đọc chậm (`coach_length_scale`, mặc định 1.35) để người dùng
NÓI THEO — yêu cầu có thật của sản phẩm. VieNeu-TTS cho giọng hay hơn nhiều
nhưng `infer_stream()` KHÔNG có tham số tốc độ, nên thay thẳng là mất tính
năng đó mà không có gì báo.
"""
from __future__ import annotations

from app.ai.tts import PiperTts
from app.ai.tts_router import TtsRouter, create_tts
from app.core.config import load_config


class FakeExpressive:
    """Đứng thay VieNeu: giọng hay, không đổi được tốc độ."""

    supports_length_scale = False
    used_standby = True

    def __init__(self):
        self.sample_rate = 48_000
        self.is_active = False
        self.calls = []

    def synthesize(self, *a, **kw):
        self.calls.append(kw.get("length_scale"))
        return iter(())

    def prewarm(self, *a, **kw):
        pass


def _router():
    config = load_config()
    return TtsRouter(config, primary=FakeExpressive(), fallback=PiperTts(config))


def test_toc_do_binh_thuong_di_engine_chinh():
    r = _router()
    r.synthesize("utt_001", "xin chào", length_scale=None)
    assert isinstance(r._active, FakeExpressive)
    r.synthesize("utt_002", "xin chào", length_scale=1.0)
    assert isinstance(r._active, FakeExpressive)


def test_can_doc_cham_thi_quay_ve_piper():
    r = _router()
    r.synthesize("utt_003", "I think we should wait", length_scale=1.35)
    assert isinstance(r._active, PiperTts), (
        "đọc chậm mà đi engine không đổi được tốc độ = mất tính năng nói theo"
    )


def test_sample_rate_theo_engine_dang_dung():
    """Hai engine khác tần số (48k và 22.05k). Báo sai thì client phát méo."""
    r = _router()
    r.synthesize("utt_004", "xin chào", length_scale=None)
    assert r.sample_rate == 48_000
    r.synthesize("utt_005", "slow please", length_scale=1.35)
    assert r.sample_rate == load_config().tts.sample_rate


def test_mac_dinh_van_la_piper_tran():
    """Không bật `vieneu` thì không được kéo theo gói nặng hay lớp bọc thừa."""
    config = load_config()
    config.tts.engine = "piper"
    assert type(create_tts(config)) is PiperTts
