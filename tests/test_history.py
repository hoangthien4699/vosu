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


def test_ghi_nhan_cau_nguoi_dung_da_chon():
    h = make(2)
    h.set_user_reply("utt_000", "Sure, let's do that.")
    rendered = h.render()
    assert 'Them: "Câu số 0."' in rendered
    assert 'You: "Sure, let\'s do that."' in rendered


def test_luot_chua_chon_reply_thi_khong_co_dong_you():
    h = make(1)
    assert "You:" not in h.render()


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
    h.set_user_reply("utt_khong_co", "y")
    assert len(h) == 1
