"""E2E qua WebSocket thật với STT/LLM giả lập (Task F4).

Kiểm tra hợp đồng ở ranh giới hệ thống — thứ mà unit test từng module không
bắt được: thứ tự event, tính đầy đủ của envelope, và cam kết §4.4 rằng frontend
không bao giờ thấy JSON thô của LLM.
"""
from __future__ import annotations

import contextlib
import json
import sys

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ai.llm import CHATML, build_prompt
from app.ai.stt import Transcript
from app.api.websocket import router
from app.core.config import load_config
from tests.synth import SR, silence, speech

LLM_OUTPUT = json.dumps(
    {
        "translation": "Tôi nghĩ chúng ta nên tạm gác lại việc này.",
        "intent": "muốn hoãn thảo luận",
        "replies": ["Understood. When can we revisit?", "Is there a blocker?"],
    },
    ensure_ascii=False,
)


class FakeStt:
    def __init__(self):
        self.calls = []

    async def transcribe(self, pcm, *, is_final=True):
        self.calls.append(("final" if is_final else "partial", pcm.size))
        return Transcript(
            text="I think we should table this for now.",
            language="en",
            language_probability=0.98,
            is_final=is_final,
            audio_s=pcm.size / SR,
            latency_ms=120.0,
        )


class FakeLlm:
    def __init__(self, output: str = LLM_OUTPUT, chunk: int = 4):
        self.output = output
        self.chunk = chunk
        self.template = CHATML
        self.prompts: list[str] = []

    def build_prompt(self, text, language):
        prompt = build_prompt(text, language, self.template)
        self.prompts.append(prompt)
        return prompt

    async def stream(self, prompt, *, stats=None, n_predict=None):
        for i in range(0, len(self.output), self.chunk):
            token = self.output[i : i + self.chunk]
            if stats is not None:
                if stats.ttft_ms is None:
                    stats.ttft_ms = 42.0
                stats.tokens += 1
            yield token


class FakeRuntime:
    def __init__(self):
        self.stt = FakeStt()
        self.llm = FakeLlm()

    @contextlib.asynccontextmanager
    async def session(self, session_id):
        yield self

    @contextlib.asynccontextmanager
    async def job(self, name="inference"):
        yield


@pytest.fixture
def client():
    cfg = load_config(env={})
    cfg.tts.enabled = False           # TTS có test riêng; ở đây tách biệt
    cfg.vad.backend = "energy"        # không phụ thuộc file model
    cfg.chunker.enable_partial = False

    app = FastAPI()
    app.include_router(router)
    app.state.config = cfg
    app.state.runtime = FakeRuntime()
    return TestClient(app)


def utterance_stream(seconds: float = 1.2) -> bytes:
    rng = np.random.default_rng(11)
    stream = np.concatenate([silence(0.4, rng), speech(seconds, rng), silence(1.2, rng)])
    return (np.clip(stream, -1, 1) * 32767).astype("<i2").tobytes()


def drain(ws, stop_type: str, limit: int = 200) -> list[dict]:
    events = []
    for _ in range(limit):
        message = ws.receive()
        if "text" not in message or message["text"] is None:
            continue
        event = json.loads(message["text"])
        events.append(event)
        if event["type"] == stop_type:
            break
    return events


def test_luong_e2e_day_du(client):
    with client.websocket_connect("/ws/copilot") as ws:
        started = json.loads(ws.receive()["text"])
        assert started["type"] == "session_started"

        payload = utterance_stream()
        for i in range(0, len(payload), 3200):     # chunk 100ms như §4.1
            ws.send_bytes(payload[i : i + 3200])

        events = drain(ws, "copilot_done")

    types = [e["type"] for e in events]
    assert "audio_started" in types
    assert "stt_final" in types
    assert "copilot_started" in types
    assert "translation_delta" in types
    assert "intent_done" in types
    assert "reply_ready" in types
    assert types[-1] == "copilot_done"


def test_moi_event_du_truong_bat_buoc(client):
    """§4.3: session_id + utterance_id + sequence + timestamp trên mọi event."""
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        payload = utterance_stream()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])
        events = drain(ws, "copilot_done")

    utterance_scoped = {
        "stt_final", "copilot_started", "copilot_done",
        "translation_delta", "intent_done", "reply_ready",
    }
    for event in events:
        assert event["session_id"].startswith("sess_")
        assert isinstance(event["sequence"], int)
        assert event["timestamp"]
        if event["type"] in utterance_scoped:
            assert event["utterance_id"], f"{event['type']} thiếu utterance_id"


def test_sequence_lien_tuc_va_tang_dan(client):
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        payload = utterance_stream()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])
        events = drain(ws, "copilot_done")

    sequences = [e["sequence"] for e in events]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences), "sequence bị trùng"


def test_khong_lo_json_tho_cua_llm_ra_frontend(client):
    """Contract §4.4 — đây là lý do tồn tại của lớp parser ở backend."""
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        payload = utterance_stream()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])
        events = drain(ws, "copilot_done")

    deltas = [e for e in events if e["type"] == "translation_delta"]
    assert deltas
    full = deltas[-1]["data"]["full"]
    assert full == "Tôi nghĩ chúng ta nên tạm gác lại việc này."
    for event in deltas:
        text = event["data"]["text"]
        for forbidden in ("{", "}", '"translation"', '"replies"', '":'):
            assert forbidden not in text, f"JSON thô lọt ra: {forbidden!r}"


def test_nhieu_utterance_co_utterance_id_khac_nhau(client):
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        seen = set()
        for _ in range(2):
            payload = utterance_stream(0.9)
            for i in range(0, len(payload), 3200):
                ws.send_bytes(payload[i : i + 3200])
            events = drain(ws, "copilot_done")
            ids = {e["utterance_id"] for e in events if e["utterance_id"]}
            seen |= ids
    assert len(seen) == 2, f"kỳ vọng 2 utterance_id riêng biệt, nhận {seen}"


def test_stt_rong_khong_goi_llm(client):
    client.app.state.runtime.stt.transcribe = _empty_transcribe
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        payload = utterance_stream(0.8)
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])
        ws.send_text(json.dumps({"action": "ping"}))
        # không có copilot_started -> chỉ nhận audio_started rồi im
        events = []
        for _ in range(3):
            message = ws.receive()
            if "text" in message and message["text"]:
                events.append(json.loads(message["text"])["type"])
            if len(events) >= 1:
                break
    assert "copilot_started" not in events


async def _empty_transcribe(pcm, *, is_final=True):
    return Transcript("", "en", 0.0, is_final, pcm.size / SR, 10.0)


def test_control_frame_sai_bi_bao_loi_khong_lam_sap(client):
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        ws.send_text("{không phải json}")
        event = json.loads(ws.receive()["text"])
        assert event["type"] == "error"
        assert event["data"]["recoverable"] is True
        # kết nối vẫn sống
        ws.send_text(json.dumps({"action": "ping"}))


# --------------------------------------------------------------------------- #
# TTS trong pipeline (§2.4 — TTS là nhánh song song, không chặn E2E của text)
# --------------------------------------------------------------------------- #

@pytest.fixture
def tts_client(monkeypatch, tmp_path):
    """Như `client` nhưng bật TTS, dùng script Piper giả lập."""
    import asyncio
    from pathlib import Path

    voice = tmp_path / "voice.onnx"
    voice.write_bytes(b"stub")

    cfg = load_config(env={})
    cfg.tts.enabled = True
    cfg.tts.stream_by_sentence = True
    cfg.tts.min_sentence_chars = 8
    cfg.paths.piper_bin = sys.executable
    cfg.paths.piper_voice_vi = str(voice)
    cfg.paths.piper_voice_en = str(voice)
    cfg.vad.backend = "energy"
    cfg.chunker.enable_partial = False

    fixtures = Path(__file__).parent / "fixtures"
    real_exec = asyncio.create_subprocess_exec

    async def patched(program, *args, **kwargs):
        return await real_exec(program, str(fixtures / "fake_piper.py"), *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", patched)
    monkeypatch.setenv("VOSU_FAKE_PIPER_CHUNKS", "4")
    monkeypatch.setenv("VOSU_FAKE_PIPER_DELAY", "0.01")

    app = FastAPI()
    app.include_router(router)
    app.state.config = cfg
    app.state.runtime = FakeRuntime()
    return TestClient(app)


def test_text_hoan_tat_truoc_khi_tts_doc_xong(tts_client):
    """§2.4: text phải hiển thị ngay, KHÔNG đợi TTS hoàn tất.

    Nếu TTS được await ngay trong vòng lặp token của LLM thì copilot_done sẽ
    bị đẩy ra sau tts_done — nghĩa là token stream đã bị treo trong lúc phát
    tiếng, và toàn bộ lợi ích của LLM streaming mất sạch.
    """
    with tts_client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        payload = utterance_stream()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])
        events = drain(ws, "copilot_done", limit=400)

    types = [e["type"] for e in events]
    assert "copilot_done" in types
    assert types.index("copilot_done") == len(types) - 1
    # Phải có ít nhất một reply_ready TRƯỚC copilot_done: nghĩa là LLM stream
    # đã chạy hết trong khi TTS còn đang đọc ở nhánh khác.
    assert "reply_ready" in types
    assert types.index("reply_ready") < types.index("copilot_done")


def test_tts_phat_su_kien_va_gui_binary(tts_client):
    with tts_client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        payload = utterance_stream()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])

        seen_types, binary_frames = [], 0
        for _ in range(400):
            message = ws.receive()
            if message.get("bytes") is not None:
                binary_frames += 1
                continue
            if not message.get("text"):
                continue
            event = json.loads(message["text"])
            seen_types.append(event["type"])
            if event["type"] == "tts_done":
                break

    assert "tts_started" in seen_types
    assert "tts_audio_chunk" in seen_types
    assert "tts_done" in seen_types
    assert binary_frames > 0, "không có frame PCM nào được gửi"


def test_cancel_tts_tu_client_dung_va_don_hang_doi(tts_client):
    with tts_client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        payload = utterance_stream()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])

        # đợi tới khi TTS bắt đầu rồi mới hủy
        for _ in range(400):
            message = ws.receive()
            if message.get("text") and json.loads(message["text"])["type"] == "tts_started":
                break
        ws.send_text(json.dumps({"action": "cancel_tts"}))

        cancelled = None
        for _ in range(200):
            message = ws.receive()
            if not message.get("text"):
                continue
            event = json.loads(message["text"])
            if event["type"] == "tts_cancelled":
                cancelled = event
                break

    assert cancelled is not None, "không nhận được tts_cancelled"
    assert cancelled["data"]["reason"] == "client_request"
    assert cancelled["data"]["response_ms"] < 200.0
    assert cancelled["utterance_id"], "tts_cancelled phải nói rõ hủy utterance nào"
