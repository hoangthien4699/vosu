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
    {"translation": "Tôi nghĩ chúng ta nên tạm gác lại việc này."}, ensure_ascii=False
)
LLM_OUTPUT_EN = json.dumps(
    {"translation": "I think we should hold off on this."}, ensure_ascii=False
)


class FakeStt:
    """STT giả lập. `language`/`text` đổi được để mô phỏng cả hai chiều."""

    def __init__(self, language: str = "en", text: str | None = None):
        self.calls = []
        self.language = language
        self.text = text or "I think we should table this for now."

    async def transcribe(self, pcm, *, is_final=True):
        self.calls.append(("final" if is_final else "partial", pcm.size))
        return Transcript(
            text=self.text,
            language=self.language,
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
        self.histories: list[str] = []
        self.directions: list = []
        self.retry_hints: list[str] = []

    def build_prompt(self, text, language, *, direction=None, counterpart_language=None,
                     history="", retry_hint=""):
        from app.ai.direction import Direction

        direction = direction or Direction.TO_USER
        prompt = build_prompt(
            text, language, self.template,
            direction=direction,
            counterpart_language=counterpart_language or "en",
            history=history,
        )
        self.prompts.append(prompt)
        self.histories.append(history)
        self.directions.append(direction)
        self.retry_hints.append(retry_hint)
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
        "stt_final", "copilot_started", "copilot_done", "translation_delta",
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
    assert deltas[-1]["data"]["full"] == "Tôi nghĩ chúng ta nên tạm gác lại việc này."
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
    # translation_delta phải đến TRƯỚC copilot_done: LLM stream chạy hết trong
    # khi TTS còn đang đọc ở nhánh khác.
    assert types.index("translation_delta") < types.index("copilot_done")


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


# --------------------------------------------------------------------------- #
# Tạm dừng giữa chừng (chế độ "từng câu một" của web client)
# --------------------------------------------------------------------------- #

def test_ngung_gui_audio_khong_lam_hong_trang_thai_vad(client):
    """VAD chạy theo FRAME, không theo đồng hồ thực.

    Đây là tính chất khiến nút "Tạm dừng" của web client an toàn: ngừng gửi
    audio thì trạng thái VAD đóng băng nguyên vẹn, gửi tiếp là chạy đúng chỗ
    cũ. Nếu VAD phụ thuộc thời gian thực, một khoảng dừng dài sẽ bị hiểu nhầm
    thành im lặng và cắt câu sai chỗ.
    """
    payload = utterance_stream(1.2)
    chunk = 3200

    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        # Gửi nửa đầu, "tạm dừng" (không gửi gì), rồi gửi nốt.
        half = len(payload) // 2
        for i in range(0, half, chunk):
            ws.send_bytes(payload[i : i + chunk])
        # khoảng dừng: client thật sẽ dừng ở đây hàng giây để chờ TTS đọc xong
        for i in range(half, len(payload), chunk):
            ws.send_bytes(payload[i : i + chunk])
        events = drain(ws, "copilot_done")

    finals = [e for e in events if e["type"] == "stt_final"]
    assert len(finals) == 1, (
        f"kỳ vọng đúng 1 câu, nhận {len(finals)} — khoảng dừng đã cắt nhầm câu"
    )


def test_nhieu_cau_cach_nhau_khong_bi_gop_hay_tach(client):
    """Ba câu cách nhau -> đúng ba utterance_id, không gộp, không tách."""
    import numpy as np

    rng = np.random.default_rng(21)
    pieces = []
    for _ in range(3):
        pieces.extend([speech(1.0, rng), silence(1.2, rng)])
    stream = np.concatenate([silence(0.4, rng)] + pieces)
    payload = (np.clip(stream, -1, 1) * 32767).astype("<i2").tobytes()

    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        seen = set()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])
        for _ in range(600):
            message = ws.receive()
            if not message.get("text"):
                continue
            event = json.loads(message["text"])
            if event["type"] == "stt_final":
                seen.add(event["utterance_id"])
            if len(seen) == 3:
                break

    assert len(seen) == 3, f"kỳ vọng 3 utterance riêng biệt, nhận {sorted(seen)}"


def test_doc_thu_cong_van_chay_sau_khi_utterance_ket_thuc(tts_client):
    """§2.4.1: quick reply CHỈ đọc khi người dùng chọn thủ công.

    Lúc người dùng bấm chọn thì pipeline đã xong từ lâu và utterance đã ở
    trạng thái kết thúc. Nếu worker TTS bỏ qua utterance kết thúc thì tính năng
    này KHÔNG hoạt động — bấm vào không có gì xảy ra, và không có lỗi nào.
    """
    with tts_client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        payload = utterance_stream()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])

        # đợi cả pipeline lẫn TTS tự động xong hẳn
        for _ in range(500):
            message = ws.receive()
            if message.get("text") and json.loads(message["text"])["type"] == "tts_done":
                break

        ws.send_text(json.dumps({"action": "speak", "text": "Understood, thanks."}))

        started = None
        for _ in range(200):
            message = ws.receive()
            if not message.get("text"):
                continue
            event = json.loads(message["text"])
            if event["type"] == "tts_started":
                started = event
                break

    assert started is not None, "yêu cầu đọc thủ công nhưng TTS không chạy"
    assert started["data"]["text"] == "Understood, thanks."


def test_che_do_manual_server_khong_tu_doc(tts_client):
    """Chế độ nghe lại: client tự quyết thứ tự đọc, server không chen vào."""
    with tts_client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        ws.send_text(json.dumps({"action": "set_tts_mode", "mode": "manual"}))

        payload = utterance_stream()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])
        events = drain(ws, "copilot_done", limit=400)

    assert not [e for e in events if e["type"] == "tts_started"], (
        "chế độ manual mà server vẫn tự đọc"
    )


def test_manual_roi_client_yeu_cau_doc_theo_thu_tu(tts_client):
    with tts_client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        ws.send_text(json.dumps({"action": "set_tts_mode", "mode": "manual"}))
        payload = utterance_stream()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])
        drain(ws, "copilot_done", limit=400)

        for field, text in (("translation", "Câu một."),
                            ("translation", "Câu hai."),
                            ("translation", "Câu ba.")):
            ws.send_text(json.dumps({"action": "speak", "field": field, "text": text}))

        seen = []
        for _ in range(600):
            message = ws.receive()
            if not message.get("text"):
                continue
            event = json.loads(message["text"])
            if event["type"] == "tts_started":
                seen.append((event["data"]["utterance_field"], event["data"]["text"]))
            if len(seen) == 3:
                break

    assert seen == [
        ("translation", "Câu một."), ("translation", "Câu hai."),
        ("translation", "Câu ba."),
    ], f"thứ tự đọc sai: {seen}"


def test_stt_final_co_vi_tri_cau_trong_stream(client):
    """Client cần `start_s` để cắt lại đúng đoạn audio GỐC của câu."""
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        payload = utterance_stream()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])
        events = drain(ws, "stt_final")

    final = next(e for e in events if e["type"] == "stt_final")
    start_s = final["data"]["start_s"]
    # client chèn 0.4s im lặng trước khi nói (xem utterance_stream)
    assert 0.1 < start_s < 0.9, f"start_s={start_s} không khớp vị trí thật của câu"


# --------------------------------------------------------------------------- #
# Bộ nhớ hội thoại (§10)
# --------------------------------------------------------------------------- #

def send_utterance(ws, seconds: float = 1.2) -> list[dict]:
    payload = utterance_stream(seconds)
    for i in range(0, len(payload), 3200):
        ws.send_bytes(payload[i : i + 3200])
    return drain(ws, "copilot_done", limit=400)


def test_cau_dau_tien_khong_co_lich_su(client):
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        send_utterance(ws)
    assert client.app.state.runtime.llm.histories[0] == "", (
        "câu đầu tiên mà đã có lịch sử thì model sẽ thấy chính nó lặp lại"
    )


def test_cau_sau_nhin_thay_cau_truoc(client):
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        send_utterance(ws)
        send_utterance(ws)
    histories = client.app.state.runtime.llm.histories
    assert len(histories) == 2
    assert "Them:" in histories[1], f"lượt 2 không thấy lượt 1: {histories[1]!r}"
    # Lịch sử ghi BẢN DỊCH, không phải nguyên văn: model đọc lịch sử bằng một
    # thứ tiếng thì mạch lạc hơn là trộn hai.
    assert "Tôi nghĩ chúng ta nên tạm gác lại" in histories[1]


def test_lich_su_khong_chua_chinh_cau_dang_hoi(client):
    """Câu hiện tại đã nằm ở phần "Now they said" — lặp lại là model tưởng
    người ta nói hai lần."""
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        send_utterance(ws)
        send_utterance(ws)
        prompt = client.app.state.runtime.llm.prompts[-1]

    marker = "Them said:"
    assert prompt.count(marker) == 1
    before, _after = prompt.split(marker)
    # Lịch sử ghi bản dịch còn phần hỏi ghi nguyên văn, nên câu hiện tại chỉ
    # được xuất hiện đúng một lần ở phần hỏi.
    assert before.count("I think we should table this") == 0, (
        "câu hiện tại lọt vào cả khối lịch sử"
    )


def test_reset_xoa_lich_su(client):
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        send_utterance(ws)
        ws.send_text(json.dumps({"action": "reset"}))
        send_utterance(ws)
    assert client.app.state.runtime.llm.histories[-1] == ""


def test_tat_lich_su_thi_prompt_khong_doi(client):
    client.app.state.config.llm.history_turns = 0
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        send_utterance(ws)
        send_utterance(ws)
    assert all(h == "" for h in client.app.state.runtime.llm.histories)


# --------------------------------------------------------------------------- #
# Hai chiều dịch
# --------------------------------------------------------------------------- #

@pytest.fixture
def vi_client(client):
    """Như `client` nhưng STT trả về tiếng Việt — mô phỏng NGƯỜI DÙNG nói."""
    client.app.state.runtime.stt = FakeStt("vi", "Tôi nghĩ chúng ta nên hoãn lại.")
    client.app.state.runtime.llm = FakeLlm(LLM_OUTPUT_EN)
    return client


def test_doi_phuong_noi_thi_chieu_la_to_user(client):
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        events = send_utterance(ws)

    final = next(e for e in events if e["type"] == "stt_final")
    assert final["data"]["direction"] == "to_user"
    delta = [e for e in events if e["type"] == "translation_delta"][-1]
    assert delta["data"]["direction"] == "to_user"
    assert delta["data"]["language"] == "vi", "dịch cho người dùng thì đích là tiếng Việt"


def test_nguoi_dung_noi_thi_chieu_la_to_counterpart(vi_client):
    """Whisper nhận ra tiếng Việt -> hiểu là người dùng đang nói -> dịch ngược."""
    with vi_client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        events = send_utterance(ws)

    final = next(e for e in events if e["type"] == "stt_final")
    assert final["data"]["direction"] == "to_counterpart"
    delta = [e for e in events if e["type"] == "translation_delta"][-1]
    assert delta["data"]["direction"] == "to_counterpart"
    assert delta["data"]["language"] == "en", "dịch ngược thì đích là tiếng đối phương"
    assert delta["data"]["full"] == "I think we should hold off on this."


def test_prompt_dung_dung_chieu(vi_client):
    from app.ai.direction import Direction

    with vi_client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        send_utterance(ws)
    assert vi_client.app.state.runtime.llm.directions[-1] is Direction.TO_COUNTERPART


def test_dich_nguoc_doc_cham_va_dung_giong_doi_phuong(tts_client):
    """Người dùng phải NÓI THEO, không chỉ nghe hiểu."""
    tts_client.app.state.runtime.stt = FakeStt("vi", "Tôi nghĩ chúng ta nên hoãn lại.")
    tts_client.app.state.runtime.llm = FakeLlm(LLM_OUTPUT_EN)
    cfg = tts_client.app.state.config

    with tts_client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        started = None
        payload = utterance_stream()
        for i in range(0, len(payload), 3200):
            ws.send_bytes(payload[i : i + 3200])
        for _ in range(500):
            message = ws.receive()
            if not message.get("text"):
                continue
            event = json.loads(message["text"])
            if event["type"] == "tts_started":
                started = event
                break

    assert started is not None, "chiều ngược không đọc gì"
    assert started["data"]["utterance_field"] == "coach"
    assert started["data"]["voice"] == "en", "phải đọc bằng giọng tiếng đối phương"
    assert cfg.tts.coach_length_scale > cfg.tts.length_scale, (
        "tốc độ đọc chiều ngược phải chậm hơn bình thường"
    )


def test_ngon_ngu_doi_phuong_lay_theo_thuc_te_nghe_duoc(client):
    """Nghe họ nói tiếng Nhật thì chiều ngược phải dịch sang tiếng Nhật,
    không bám mãi vào mặc định trong config."""
    client.app.state.runtime.stt = FakeStt("ja", "その件は社内で確認します。")
    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        send_utterance(ws)
        client.app.state.runtime.stt = FakeStt("vi", "Vâng, tôi hiểu rồi.")
        send_utterance(ws)
        prompt = client.app.state.runtime.llm.prompts[-1]

    assert "into Japanese" in prompt, "không bám theo ngôn ngữ thật của đối phương"


# --------------------------------------------------------------------------- #
# Dịch lại khi bản dịch hỏng
# --------------------------------------------------------------------------- #

class EchoLlm(FakeLlm):
    """Lần đầu chép nguyên văn, lần sau dịch đúng — mô phỏng lỗi thật của
    model 2B (`"Chị cho em hỏi thêm một chút về giá."` bị trả lại y nguyên)."""

    def __init__(self, source: str, good: str):
        super().__init__(json.dumps({"translation": good}, ensure_ascii=False))
        self.echo_output = json.dumps({"translation": source}, ensure_ascii=False)
        self.good_output = self.output
        self.calls = 0

    async def stream(self, prompt, *, stats=None, n_predict=None):
        self.calls += 1
        self.output = self.echo_output if self.calls == 1 else self.good_output
        async for token in super().stream(prompt, stats=stats, n_predict=n_predict):
            yield token


def test_chep_nguyen_van_thi_dich_lai_mot_lan(client):
    source = "Chị cho em hỏi thêm một chút về giá."
    client.app.state.runtime.stt = FakeStt("vi", source)
    client.app.state.runtime.llm = EchoLlm(source, "Can I ask a bit more about the pricing?")

    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        events = send_utterance(ws)

    llm = client.app.state.runtime.llm
    assert llm.calls == 2, f"không dịch lại (gọi {llm.calls} lần)"
    assert any("repeated the input" in h for h in llm.retry_hints), llm.retry_hints

    final = [e for e in events if e["type"] == "translation_delta"][-1]
    assert final["data"]["full"] == "Can I ask a bit more about the pricing?"


def test_ban_dich_tot_thi_khong_dich_lai(client):
    """Dịch lại tốn ~1 giây — không được kích hoạt nhầm."""
    client.app.state.runtime.stt = FakeStt("vi", "Tôi đồng ý.")
    client.app.state.runtime.llm = FakeLlm(LLM_OUTPUT_EN)

    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        send_utterance(ws)

    llm = client.app.state.runtime.llm
    assert llm.prompts and all(h == "" for h in llm.retry_hints), llm.retry_hints


def test_chi_dich_lai_dung_mot_lan(client):
    """Lần hai còn hỏng thì lần ba cũng thế — không lặp vô hạn."""
    source = "Chị cho em hỏi thêm một chút về giá."
    client.app.state.runtime.stt = FakeStt("vi", source)
    # luôn chép, không bao giờ dịch đúng
    client.app.state.runtime.llm = FakeLlm(
        json.dumps({"translation": source}, ensure_ascii=False)
    )

    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        events = send_utterance(ws)

    assert len(client.app.state.runtime.llm.prompts) == 2
    assert any(e["type"] == "copilot_done" for e in events), "pipeline phải kết thúc"


def test_tat_duoc_viec_dich_lai(client):
    client.app.state.config.llm.retry_on_bad_translation = False
    source = "Chị cho em hỏi thêm một chút về giá."
    client.app.state.runtime.stt = FakeStt("vi", source)
    client.app.state.runtime.llm = FakeLlm(
        json.dumps({"translation": source}, ensure_ascii=False)
    )

    with client.websocket_connect("/ws/copilot") as ws:
        ws.receive()
        send_utterance(ws)

    assert len(client.app.state.runtime.llm.prompts) == 1
