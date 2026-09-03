from __future__ import annotations

import json

from app.ai.copilot import (
    IntentDone,
    ReplyReady,
    SemanticEventParser,
    TranslationDelta,
)

SAMPLE = {
    "translation": "Tôi nghĩ chúng ta nên tạm gác lại cuộc thảo luận này.",
    "intent": "muốn hoãn thảo luận",
    "replies": ["Understood. When can we revisit?", "Is there a blocker first?"],
}


def feed_all(text: str, size: int = 3):
    parser = SemanticEventParser()
    events = []
    for i in range(0, len(text), size):
        events.extend(parser.feed(text[i : i + size]))
    events.extend(parser.finish())
    return events, parser


def test_sinh_du_ba_loai_semantic_event():
    events, parser = feed_all(json.dumps(SAMPLE, ensure_ascii=False))

    assert any(isinstance(e, TranslationDelta) for e in events)
    assert [e.intent for e in events if isinstance(e, IntentDone)] == ["muốn hoãn thảo luận"]
    replies = [(e.index, e.text) for e in events if isinstance(e, ReplyReady)]
    assert replies == [(0, SAMPLE["replies"][0]), (1, SAMPLE["replies"][1])]
    assert parser.result.translation == SAMPLE["translation"]
    assert not parser.result.malformed


def test_khong_bao_gio_lo_json_tho_ra_ngoai():
    """Contract §4.4 — frontend không được thấy dấu ngoặc/tên trường của LLM."""
    events, _ = feed_all(json.dumps(SAMPLE, ensure_ascii=False), size=1)

    leaked = "".join(e.text for e in events if isinstance(e, TranslationDelta))
    assert leaked == SAMPLE["translation"]
    for forbidden in ('{', '}', '"translation"', '"replies"', '":'):
        assert forbidden not in leaked, f"JSON thô lọt ra ngoài: {forbidden!r}"


def test_translation_delta_cong_don_khop_full():
    events, _ = feed_all(json.dumps(SAMPLE, ensure_ascii=False), size=2)
    deltas = [e for e in events if isinstance(e, TranslationDelta)]
    assert "".join(e.text for e in deltas) == deltas[-1].full == SAMPLE["translation"]


def test_intent_chi_phat_mot_lan():
    events, _ = feed_all(json.dumps(SAMPLE, ensure_ascii=False), size=1)
    assert len([e for e in events if isinstance(e, IntentDone)]) == 1


def test_reply_khong_bi_lap_khi_finish():
    events, _ = feed_all(json.dumps(SAMPLE, ensure_ascii=False))
    indexes = [e.index for e in events if isinstance(e, ReplyReady)]
    assert indexes == sorted(set(indexes))


def test_chap_nhan_ten_truong_cua_baseline_v1():
    """Model 3B lượng tử hóa hay quay về schema cũ — không được mất dữ liệu."""
    payload = json.dumps(
        {"trans": "Xin chào", "cultural_intent": "chào hỏi",
         "suggested_replies": [{"tone": "Thân mật", "text": "Hi there!"}]},
        ensure_ascii=False,
    )
    events, parser = feed_all(payload)
    assert parser.result.translation == "Xin chào"
    assert [e.intent for e in events if isinstance(e, IntentDone)] == ["chào hỏi"]
    assert [e.text for e in events if isinstance(e, ReplyReady)] == ["Hi there!"]


def test_json_cut_giua_chung_van_dung_duoc_phan_dich():
    events, parser = feed_all('{"translation": "Tôi nghĩ chúng ta nên')
    assert parser.result.translation == "Tôi nghĩ chúng ta nên"
    assert parser.result.is_useful, "bản dịch cụt vẫn hữu ích, không được vứt"
    assert parser.result.malformed


def test_result_rong_thi_khong_useful():
    _, parser = feed_all("hoàn toàn không phải JSON")
    assert not parser.result.is_useful


def test_bo_qua_truong_meaning_thua():
    """§4.4 bỏ `meaning`; nếu model vẫn sinh thì không được coi là reply."""
    payload = json.dumps(
        {"replies": [{"text": "Sure.", "meaning": "Chắc chắn rồi."}]}, ensure_ascii=False
    )
    events, parser = feed_all(payload)
    assert [e.text for e in events if isinstance(e, ReplyReady)] == ["Sure."]
    assert parser.result.replies == ["Sure."]
