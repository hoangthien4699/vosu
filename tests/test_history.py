from __future__ import annotations

from app.ai.history import ConversationHistory


def make(n: int) -> ConversationHistory:
    h = ConversationHistory(max_turns=4, max_chars=200)
    for i in range(n):
        h.add(f"utt_{i:03d}", f"Câu số {i}.", "en")
    return h


def test_cua_so_truot_giu_luot_gan_nhat():
    h = make(6)
    assert len(h) == 4
    assert [t.utterance_id for t in h.turns] == ["utt_002", "utt_003", "utt_004", "utt_005"]


def test_render_rong_khi_chua_co_luot_nao():
    assert ConversationHistory().render() == ""


def test_render_bo_dung_luot_dang_xu_ly():
    """Câu hiện tại đã nằm ở phần sau của prompt — đưa vào đây nữa là lặp."""
    h = make(3)
    rendered = h.render(exclude="utt_002")
    assert "Câu số 2." not in rendered
    assert "Câu số 1." in rendered


def test_ghi_ca_hai_phia():
    """Người dùng nói thật nên lượt của họ cũng đi qua STT — lịch sử phản ánh
    đúng hội thoại, không phải suy đoán từ việc bấm nút nào."""
    h = ConversationHistory()
    h.add("u0", "I think we should wait.", "en")
    h.add("u1", "Tôi đồng ý với anh.", "vi", is_user=True)
    rendered = h.render()
    assert 'Them: "I think we should wait."' in rendered
    assert 'You: "Tôi đồng ý với anh."' in rendered


def test_chi_co_luot_doi_phuong_thi_khong_co_dong_you():
    h = make(1)
    assert "You:" not in h.render()


def test_uu_tien_ban_dich_thay_vi_nguyen_van():
    """Model đọc lịch sử bằng một thứ tiếng thì mạch lạc hơn là trộn hai."""
    h = ConversationHistory()
    h.add("u0", "I think we should wait.", "en")
    h.set_translation("u0", "Tôi nghĩ chúng ta nên đợi.")
    assert 'Them: "Tôi nghĩ chúng ta nên đợi."' in h.render()


def test_chua_co_ban_dich_thi_dung_nguyen_van():
    h = ConversationHistory()
    h.add("u0", "Raw text.", "en")
    assert 'Them: "Raw text."' in h.render()


def test_cat_tu_luot_cu_nhat_khi_vuot_tran_ky_tu():
    """Lượt gần nhất giúp phân giải đại từ; lượt xa thì để mất được."""
    h = ConversationHistory(max_turns=10, max_chars=80)
    for i in range(5):
        h.add(f"utt_{i}", "x" * 40, "en")
    rendered = h.render()
    assert len(rendered) <= 90
    assert rendered.count("Them:") < 5


def test_khong_bao_gio_vuot_qua_max_turns():
    h = ConversationHistory(max_turns=3)
    for i in range(20):
        h.add(f"utt_{i}", f"câu {i}", "en")
    assert len(h) == 3


def test_clear_xoa_sach():
    h = make(3)
    h.clear()
    assert len(h) == 0 and h.render() == ""


def test_set_translation_va_reply_cho_luot_khong_ton_tai_khong_no():
    h = make(1)
    h.set_translation("utt_khong_co", "x")
    assert len(h) == 1
