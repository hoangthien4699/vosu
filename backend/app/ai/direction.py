"""Chiều dịch của một utterance.

Sản phẩm nghe HAI người, không phải một:

    Đối phương nói tiếng Anh  ->  dịch sang tiếng Việt cho người dùng ĐỌC/NGHE
    Người dùng nói tiếng Việt ->  dịch sang tiếng Anh và ĐỌC CHẬM để nói theo

Chiều được suy ra từ ngôn ngữ Whisper nhận diện, không phải từ một nút bấm:
người dùng đang trong hội thoại thật, không rảnh để bấm nút mỗi lượt.

Rủi ro đã biết: LID của Whisper trên câu ngắn ("Yes.", "Ừ.") không đáng tin.
Vì vậy `resolve()` nhận cả `fallback` — khi không nhận diện được ngôn ngữ thì
giữ nguyên chiều của lượt trước thay vì đoán bừa.
"""

from __future__ import annotations

from enum import Enum


class Direction(str, Enum):
    #: Đối phương nói -> dịch sang tiếng của người dùng, đọc giọng bình thường.
    TO_USER = "to_user"
    #: Người dùng nói -> dịch sang tiếng đối phương, đọc CHẬM để nói theo.
    TO_COUNTERPART = "to_counterpart"

    @property
    def is_outbound(self) -> bool:
        return self is Direction.TO_COUNTERPART


def resolve(
    detected: str | None,
    *,
    user_language: str,
    fallback: Direction = Direction.TO_USER,
) -> Direction:
    """Suy chiều từ ngôn ngữ Whisper nhận diện.

    `detected` rỗng nghĩa là Whisper không chắc — giữ `fallback` (chiều của
    lượt trước) thay vì mặc định về một chiều cố định. Đoán sai chiều thì người
    dùng nghe bản dịch sai hướng, rất khó hiểu chuyện gì đang xảy ra.
    """
    if not detected:
        return fallback
    normalized = detected.strip().lower().split("-")[0]
    if normalized == user_language.strip().lower().split("-")[0]:
        return Direction.TO_COUNTERPART
    return Direction.TO_USER
