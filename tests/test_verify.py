"""Phát hiện bản dịch hỏng.

Bắt nhầm tốn một lần dịch lại (~1 giây) và làm bản dịch nhấp nháy trên màn
hình, nên các test "không bắt nhầm" quan trọng ngang các test "bắt được".
"""
from __future__ import annotations

import pytest

from app.ai.verify import failure_reason, retry_hint

VI = "Chị cho em hỏi thêm một chút về giá."


def test_bat_duoc_chep_nguyen_van():
    """Quan sát thật với Qwen3.5-2B, tái hiện được."""
    assert failure_reason(VI, VI, "en") == "echo"


def test_bat_duoc_chep_du_khac_dau_cau():
    assert failure_reason(VI, "Chị cho em hỏi thêm một chút về giá", "en") == "echo"
    assert failure_reason(VI, "  chị cho em hỏi thêm một chút về giá!  ", "en") == "echo"


def test_bat_duoc_sai_ngon_ngu():
    assert failure_reason(VI, "Tôi muốn hỏi thêm về giá cả.", "en") == "wrong_language"


def test_bat_duoc_rong():
    assert failure_reason(VI, "   ", "en") == "empty"


def test_khong_bat_nham_ban_dich_dung():
    assert failure_reason(VI, "Can I ask a bit more about the pricing?", "en") is None
    assert failure_reason("How much slack do we have?",
                          "Chúng ta còn bao nhiêu thời gian?", "vi") is None


def test_khong_dung_dau_hieu_thieu_dau_de_ket_luan():
    """Đích là tiếng Việt mà output không dấu thì KHÔNG kết luận được.

    Câu tiếng Việt ngắn có thể không có dấu nào ("Ok", "Vang"). Dấu hiệu này
    chỉ tin được một chiều.
    """
    assert failure_reason("Okay.", "Ok", "vi") is None


def test_dich_sang_tieng_viet_co_dau_van_hop_le():
    assert failure_reason("Monday works.", "Thứ Hai được nhé.", "vi") is None


def test_ban_dich_trung_lap_mot_phan_khong_bi_bat():
    """Tên riêng hay thuật ngữ giữ nguyên là bình thường, không phải chép."""
    assert failure_reason("Send it to QA.", "Gửi cho QA nhé.", "vi") is None


@pytest.mark.parametrize("reason", ["echo", "wrong_language", "empty"])
def test_loi_nhac_dich_lai_neu_ro_ngon_ngu_dich(reason):
    hint = retry_hint(reason, "English")
    assert "English" in hint
    assert len(hint) < 200, "lời nhắc dài làm hụt ngân sách prompt"
