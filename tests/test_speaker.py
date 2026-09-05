"""Khóa cách quyết định "có đổi người nói không".

Đây là tín hiệu để CẤM GHÉP hai đoạn thành một câu. Quyết sai theo hướng
"người khác" thì cắt đôi một câu đang nói dở — nên vùng không chắc phải trả về
None chứ không đoán bừa.
"""
from __future__ import annotations

import numpy as np

from app.ai.speaker import SpeakerTracker


def vec(*first, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(first):
        v[i] = x
    return v


def test_cung_mot_giong_thi_cung_mot_nguoi():
    t = SpeakerTracker()
    a = t.assign(vec(1.0, 0.02))
    b = t.assign(vec(1.0, 0.05))
    assert a == b == "spk_1"
    assert t.known == 1


def test_giong_khac_han_thi_la_nguoi_moi():
    t = SpeakerTracker()
    assert t.assign(vec(1.0, 0.0)) == "spk_1"
    assert t.assign(vec(0.0, 1.0)) == "spk_2"
    assert t.known == 2


def test_vung_khong_chac_tra_ve_none_chu_khong_doan_bua():
    """Ở giữa hai ngưỡng thì KHÔNG quyết.

    Đoán nhầm thành "người khác" sẽ cắt đôi câu đang nói dở — tệ hơn hẳn so
    với việc không quyết và để cơ chế cũ xử lý.
    """
    t = SpeakerTracker(same_threshold=0.78, diff_threshold=0.62)
    t.assign(vec(1.0, 0.0))
    # cosine ~0.70, nằm giữa hai ngưỡng
    giua = vec(0.70, 0.714)
    assert t.assign(giua) is None
    assert t.known == 1, "không chắc thì cũng đừng tạo người mới"


def test_tam_cum_troi_theo_giong():
    """Giọng một người đổi theo âm lượng, khoảng cách mic, cảm xúc."""
    t = SpeakerTracker()
    t.assign(vec(1.0, 0.0))
    for _ in range(6):
        t.assign(vec(0.95, 0.30))
    # Sau khi trôi, một mẫu gần vị trí mới vẫn phải là cùng người.
    assert t.assign(vec(0.93, 0.34)) == "spk_1"
    assert t.known == 1


def test_khong_de_so_nguoi_phinh_vo_han():
    t = SpeakerTracker(max_speakers=2)
    t.assign(vec(1.0, 0.0))
    t.assign(vec(0.0, 1.0))
    t.assign(vec(0.0, 0.0, 1.0))          # giọng thứ ba, hết chỗ
    assert t.known == 2


def test_vector_rong_hoac_toan_khong_thi_bo_qua():
    t = SpeakerTracker()
    assert t.assign(np.zeros(8, dtype=np.float32)) is None
    assert t.assign(np.array([], dtype=np.float32)) is None
    assert t.known == 0
