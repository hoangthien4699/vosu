"""Phát hiện bản dịch hỏng để dịch lại một lần.

Model 2B thỉnh thoảng rơi vào chế độ CHÉP NGUYÊN VĂN thay vì dịch — quan sát
thật, tái hiện được với mẫu câu "Chị cho em hỏi thêm một chút về giá.":

    nguồn     Chị cho em hỏi thêm một chút về giá.
    bản dịch  Chị cho em hỏi thêm một chút về giá.

Không phụ thuộc temperature hay grammar; sửa prompt chỉ bớt được chứ không hết.
Người dùng nghe câu tiếng Việt của chính mình đọc lại bằng giọng Anh — vô dụng
và rất khó hiểu chuyện gì xảy ra. Nên có lưới an toàn: phát hiện thì dịch lại
một lần với lời nhắc cứng hơn.

Cố ý CHỈ bắt các dấu hiệu chắc chắn. Dịch lại tốn ~1 giây, nên báo động giả
đắt hơn là bỏ sót một ca hiếm.
"""

from __future__ import annotations

import re
import unicodedata

#: Dấu phụ tiếng Việt — dấu hiệu đáng tin để nhận ra một chuỗi là tiếng Việt.
_VIET_MARKS = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    r"ùúủũụưừứửữựỳýỷỹỵđ]",
    re.I,
)


def _normalize(text: str) -> str:
    """Chuẩn hóa để so sánh: bỏ dấu câu, gộp khoảng trắng, về chữ thường."""
    folded = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"[^\w\s]", "", folded).strip()


def _looks_vietnamese(text: str) -> bool:
    return bool(_VIET_MARKS.search(text))


def failure_reason(
    source: str, translation: str, target_language: str
) -> str | None:
    """Trả lý do nếu bản dịch có vẻ hỏng; None nếu chấp nhận được.

    `target_language` là mã ngôn ngữ ĐÍCH ("vi", "en", ...).
    """
    stripped = translation.strip()
    if not stripped:
        return "empty"

    source_norm = _normalize(source)
    if source_norm and _normalize(stripped) == source_norm:
        return "echo"

    target = (target_language or "").strip().lower().split("-")[0]
    if target != "vi" and _looks_vietnamese(stripped):
        # Đích không phải tiếng Việt mà output đầy dấu tiếng Việt -> chưa dịch.
        # Chỉ tin dấu hiệu này theo MỘT chiều: vắng dấu tiếng Việt KHÔNG chứng
        # minh được điều ngược lại, vì câu tiếng Việt ngắn có thể không dấu.
        return "wrong_language"

    return None


def retry_hint(reason: str, target_language_name: str) -> str:
    """Lời nhắc thêm vào lượt user khi dịch lại."""
    if reason == "echo":
        return (
            f"Your previous answer just repeated the input. That is wrong. "
            f"Write the sentence in {target_language_name} this time."
        )
    if reason == "wrong_language":
        return (
            f"Your previous answer was not in {target_language_name}. "
            f"Write it in {target_language_name} this time."
        )
    return f"Answer again, in {target_language_name}."
