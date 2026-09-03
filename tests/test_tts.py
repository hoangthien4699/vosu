from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from app.ai.tts import PiperTts, SentenceSplitter, TtsState, TtsUnavailable
from app.core.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def tts(tmp_path, monkeypatch):
    voice = tmp_path / "voice.onnx"
    voice.write_bytes(b"stub")

    cfg = load_config(env={})
    cfg.paths.piper_voice_vi = str(voice)
    cfg.paths.piper_voice_en = str(voice)
    cfg.paths.piper_bin = sys.executable
    cfg.tts.chunk_ms = 50

    engine = PiperTts(cfg)

    # thay lệnh chạy: python fake_piper.py --model ...
    real_exec = asyncio.create_subprocess_exec

    async def patched(program, *args, **kwargs):
        return await real_exec(
            program, str(FIXTURES / "fake_piper.py"), *args, **kwargs
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", patched)
    monkeypatch.setenv("VOSU_FAKE_PIPER_DELAY", "0.02")
    monkeypatch.setenv("VOSU_FAKE_PIPER_CHUNKS", "50")
    return engine


async def test_trang_thai_di_dung_vong_doi(tts, monkeypatch):
    monkeypatch.setenv("VOSU_FAKE_PIPER_CHUNKS", "3")
    monkeypatch.setenv("VOSU_FAKE_PIPER_DELAY", "0.001")

    assert tts.state is TtsState.IDLE
    seen = []
    async for _chunk in tts.synthesize("utt_001", "Xin chào các bạn"):
        seen.append(tts.state)
    assert TtsState.PLAYING in seen
    assert tts.state is TtsState.DONE


async def test_barge_in_dung_duoi_200ms(tts):
    """Contract §2.4.1 / benchmark B8."""
    chunks = []

    async def consume():
        async for chunk in tts.synthesize("utt_001", "Một câu dài để còn kịp ngắt lời"):
            chunks.append(chunk)

    task = asyncio.create_task(consume())
    # đợi tới khi thực sự đang phát
    for _ in range(200):
        if tts.state is TtsState.PLAYING:
            break
        await asyncio.sleep(0.005)
    assert tts.state is TtsState.PLAYING, "TTS chưa vào trạng thái PLAYING"

    result = await tts.cancel(reason="barge_in")
    await asyncio.wait_for(task, timeout=2.0)

    assert result.cancelled
    assert result.response_ms < 200.0, f"Barge-in mất {result.response_ms:.1f}ms (target <200ms)"
    assert tts.state is TtsState.INTERRUPTED
    assert result.chunks_sent > 0


async def test_cancel_khi_khong_phat_thi_khong_lam_gi(tts):
    result = await tts.cancel()
    assert not result.cancelled
    assert tts.state is TtsState.IDLE


async def test_co_the_doc_tiep_sau_khi_bi_ngat(tts, monkeypatch):
    async def consume():
        async for _ in tts.synthesize("utt_001", "câu thứ nhất dài dòng"):
            pass

    task = asyncio.create_task(consume())
    for _ in range(200):
        if tts.state is TtsState.PLAYING:
            break
        await asyncio.sleep(0.005)
    await tts.cancel()
    await asyncio.wait_for(task, timeout=2.0)
    assert tts.state is TtsState.INTERRUPTED

    monkeypatch.setenv("VOSU_FAKE_PIPER_CHUNKS", "2")
    monkeypatch.setenv("VOSU_FAKE_PIPER_DELAY", "0.001")
    got = [c async for c in tts.synthesize("utt_002", "câu thứ hai")]
    assert got, "utterance mới phải đọc được sau Barge-in"
    assert tts.state is TtsState.DONE


async def test_text_rong_khong_khoi_dong_tien_trinh(tts):
    got = [c async for c in tts.synthesize("utt_001", "   ")]
    assert got == []
    assert tts.state is TtsState.IDLE


async def test_thieu_voice_model_bao_loi_ro_rang(tmp_path):
    cfg = load_config(env={})
    cfg.paths.piper_bin = sys.executable
    cfg.paths.piper_voice_vi = str(tmp_path / "khong-ton-tai.onnx")
    engine = PiperTts(cfg)
    with pytest.raises(TtsUnavailable, match="voice model"):
        [c async for c in engine.synthesize("utt_001", "xin chào")]


# --------------------------- SentenceSplitter ---------------------------- #

def test_splitter_cat_theo_cau():
    s = SentenceSplitter(min_chars=10)
    assert s.feed("Tôi nghĩ chúng ta nên dừng. ") == ["Tôi nghĩ chúng ta nên dừng."]
    assert s.feed("Còn câu thứ hai nữa đây! ") == ["Còn câu thứ hai nữa đây!"]


def test_splitter_khong_cat_o_so_thap_phan():
    s = SentenceSplitter(min_chars=5)
    assert s.feed("Giá là 3.5 triệu đồng nhé") == []
    assert s.feed(".") == ["Giá là 3.5 triệu đồng nhé."]


def test_splitter_gop_cau_qua_ngan():
    s = SentenceSplitter(min_chars=20)
    assert s.feed("Vâng. ") == []          # quá ngắn -> chờ thêm
    out = s.feed("Tôi hiểu ý anh rồi ạ. ")
    assert out == ["Vâng. Tôi hiểu ý anh rồi ạ."]


def test_splitter_flush_tra_phan_con_lai():
    s = SentenceSplitter(min_chars=5)
    s.feed("chưa có dấu chấm")
    assert s.flush() == "chưa có dấu chấm"
    assert s.flush() == ""
