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


def test_meaning_di_kem_reply():
    """`meaning` là bản dịch tiếng Việt của reply — người dùng cần biết mình
    sắp nói gì. §4.4 từng bỏ trường này vì latency; đưa lại theo yêu cầu sản
    phẩm, và chi phí đã được đo lại."""
    payload = json.dumps(
        {"replies": [{"text": "Sure, that works.", "meaning": "Được, vậy cũng ổn."}]},
        ensure_ascii=False,
    )
    events, parser = feed_all(payload)
    replies = [e for e in events if isinstance(e, ReplyReady)]
    assert [r.text for r in replies] == ["Sure, that works."]
    assert [r.meaning for r in replies] == ["Được, vậy cũng ổn."]
    assert parser.result.replies == ["Sure, that works."]


def test_van_chap_nhan_tu_khoa_purpose_cu():
    """Model đôi khi bám theo từ khóa cũ — không được mất dữ liệu vì thế."""
    payload = json.dumps(
        {"replies": [{"text": "Okay.", "purpose": "Đồng ý ngắn gọn."}]}, ensure_ascii=False
    )
    events, _ = feed_all(payload)
    assert [e.meaning for e in events if isinstance(e, ReplyReady)] == ["Đồng ý ngắn gọn."]


# --------------------------------------------------------------------------- #
# Dọn rác cấu trúc do sinh có grammar
# --------------------------------------------------------------------------- #

def test_cat_rac_cau_truc_o_cuoi_gia_tri():
    """Grammar đảm bảo JSON đúng cấu trúc, nhưng `{`, `}` là ký tự HỢP LỆ bên
    trong chuỗi JSON nên không cấm được. Quan sát thật với Gemma 3 4B:

        "meaning":"Được rồi, chúng ta tạm dừng lại đây.},{"

    JSON vẫn hợp lệ, chỉ là người dùng đọc thấy rác.
    """
    from app.ai.copilot import clean_value

    assert clean_value("Tạm dừng lại đây.},{") == "Tạm dừng lại đây."
    assert clean_value("Xong.}]") == "Xong."


def test_khong_cat_nham_dau_cau_binh_thuong():
    from app.ai.copilot import clean_value

    for text in (
        "Bình thường không có rác.",
        'Anh ấy nói "được" rồi.',
        "Câu hỏi phải không?",
        "Kết thúc bằng dấu phẩy,",
        "Ba chấm…",
    ):
        assert clean_value(text) == text, text


def test_cat_het_thi_giu_nguyen():
    """Không bao giờ trả về chuỗi rỗng — thà giữ rác còn hơn mất nội dung."""
    from app.ai.copilot import clean_value

    assert clean_value("}{") == "}{"
    assert clean_value("]") == "]"


def test_translation_co_rac_duoc_sua_lai_qua_delta():
    """UI đã hiện phần rác qua delta, nên phải phát thêm một delta sửa lại."""
    payload = '{"translation":"Tôi nghĩ vậy.},{"}'
    events, parser = feed_all(payload, size=4)

    deltas = [e for e in events if isinstance(e, TranslationDelta)]
    assert deltas[-1].full == "Tôi nghĩ vậy.", deltas[-1].full
    assert parser.result.translation == "Tôi nghĩ vậy."


def test_rac_trong_reply_va_meaning_cung_duoc_cat():
    payload = json.dumps(
        {"replies": [{"text": "Okay.},{", "meaning": "Được rồi.},{"}]}, ensure_ascii=False
    )
    events, _ = feed_all(payload)
    reply = next(e for e in events if isinstance(e, ReplyReady))
    assert reply.text == "Okay."
    assert reply.meaning == "Được rồi."
