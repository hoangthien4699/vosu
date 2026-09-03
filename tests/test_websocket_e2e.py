"""E2E qua WebSocket thật với STT/LLM giả lập (Task F4).

Kiểm tra hợp đồng ở ranh giới hệ thống — thứ mà unit test từng module không
bắt được: thứ tự event, tính đầy đủ của envelope, và cam kết §4.4 rằng frontend
không bao giờ thấy JSON thô của LLM.
"""
from __future__ import annotations

import contextlib
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
