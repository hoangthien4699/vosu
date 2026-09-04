"""Khóa heuristic "câu đã trọn chưa".

Mọi chuỗi dưới đây là transcript THẬT do Whisper small sinh ra trong lúc đo,
không phải ví dụ bịa. Đó là lý do có những ca trông kỳ như "I think we should."
— Whisper vẫn chấm câu cho mảnh dở, nên chỉ nhìn dấu câu là không đủ.
"""
from __future__ import annotations

import pytest

from app.ai.completeness import looks_complete

# Vế trước chỗ người ta ngập ngừng — PHẢI bị coi là chưa trọn.
CHUA_TRON = [
    "So what I am trying to say is",          # không có dấu kết câu
    "The main problem is that",
    "The reason I am asking is because",
    "I think we should.",                     # có dấu chấm nhưng treo ở "should"
    "Before we sign anything I want to.",
    "They said the delivery date depends on.",
    "Could you walk me through?",
    "Tôi nghĩ là chúng ta nên...",            # dấu ba chấm = bỏ lửng
    "Vấn đề chính ở đây là...",
    "Nếu bên anh giảm ra thì...",
    "Lý do tôi hỏi là vì...",
    "",
    "   ",
]

# Câu nói xong — PHẢI được coi là trọn.
TRON = [
    "I think we should table this discussion for now.",
    "Could you walk me through the pricing structure again?",
    "Is there a major blocker we need to resolve first?",
    "We need a firm commitment on the delivery date.",
    "Their team has already sent the revised contract.",
    "Tôi nghĩ là chưa nên chốt phương án này.",
    "Nếu bên anh giảm giá thì chúng tôi sẽ ký ngay tuần này.",
    "Anh gửi lại bản báo giá mới nhất giúp tôi nhé.",
    "Chúng tôi đồng ý với điều khoản thanh toán.",
]


@pytest.mark.parametrize("text", CHUA_TRON)
def test_cau_con_do_thi_khong_duoc_coi_la_tron(text):
    assert not looks_complete(text)


@pytest.mark.parametrize("text", TRON)
def test_cau_noi_xong_thi_phai_coi_la_tron(text):
    assert looks_complete(text)


def test_tu_chi_dinh_van_ket_thuc_cau_duoc():
    """`này`/`đó`/`this` đứng cuối câu được — từng làm báo nhầm hai câu trọn.

    Trong tiếng Việt chúng đứng SAU danh từ ("phương án này"), khác hẳn mạo từ
    tiếng Anh đứng trước.
    """
    assert looks_complete("Tôi chọn phương án này.")
    assert looks_complete("Chúng tôi đồng ý với điều đó.")
    assert looks_complete("I would rather do this.")


def test_chi_nhin_chu_khong_can_audio():
    """Rẻ tới mức chạy được ngay sau STT mà không cộng gì vào độ trễ."""
    assert looks_complete("Yes.") is True
    assert looks_complete("Ừ.") is True
