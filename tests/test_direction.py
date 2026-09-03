from __future__ import annotations

from app.ai.direction import Direction, resolve


def test_doi_phuong_noi_thi_dich_cho_nguoi_dung():
    assert resolve("en", user_language="vi") is Direction.TO_USER
    assert resolve("ja", user_language="vi") is Direction.TO_USER


def test_nguoi_dung_noi_thi_dich_sang_tieng_doi_phuong():
    d = resolve("vi", user_language="vi")
    assert d is Direction.TO_COUNTERPART
    assert d.is_outbound


def test_bo_qua_hoa_thuong_va_hau_to_vung():
    assert resolve("VI", user_language="vi") is Direction.TO_COUNTERPART
    assert resolve("vi-VN", user_language="vi") is Direction.TO_COUNTERPART
    assert resolve("en-US", user_language="vi") is Direction.TO_USER


def test_khong_nhan_dien_duoc_thi_giu_chieu_luot_truoc():
    """LID trên câu ngắn không đáng tin. Đoán bừa thì người dùng nghe bản dịch
    sai hướng và không hiểu chuyện gì đang xảy ra."""
    assert resolve(None, user_language="vi") is Direction.TO_USER
    assert resolve("", user_language="vi",
                   fallback=Direction.TO_COUNTERPART) is Direction.TO_COUNTERPART


def test_nguoi_dung_dung_tieng_khac_van_hoat_dong():
    """user_language không cứng là tiếng Việt."""
    assert resolve("en", user_language="en") is Direction.TO_COUNTERPART
    assert resolve("vi", user_language="en") is Direction.TO_USER
