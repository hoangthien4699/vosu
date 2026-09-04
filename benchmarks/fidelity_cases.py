"""Bộ câu đo ĐỘ TRUNG THỰC của bản dịch.

"Nghe có tự nhiên không" thì phải người bản ngữ đọc. Nhưng "có bỏ sót thông
tin không" thì đo được: mỗi câu đi kèm danh sách yếu tố BẮT BUỘC phải còn lại
trong bản dịch — con số, phủ định, từ giảm nhẹ, tên riêng, mốc thời gian.

Đó chính là loại lỗi đã quan sát được:
    "We're thinking about pushing the launch to next quarter."
        -> "Chúng tôi đang nghĩ đến việc đẩy ra cho quý sau."   (mất "launch")
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    lang: str
    text: str
    #: Mỗi phần tử là một yếu tố phải giữ; bên trong là các dạng chấp nhận
    #: được — khớp MỘT dạng là đạt. Viết thường hết, so khớp không phân biệt hoa.
    must_keep: list[list[str]] = field(default_factory=list)
    note: str = ""


#: Chiều THUẬN: đối phương nói, dịch sang tiếng Việt.
INBOUND = [
    Case("en", "We're thinking about pushing the launch to next quarter.",
         [["ra mắt", "phát hành", "khởi chạy"], ["quý sau", "quý tới", "quý tiếp"]],
         "danh từ chính hay bị nuốt"),
    Case("en", "That's a fair concern. How much slack do we have?",
         [["hợp lý", "chính đáng", "xác đáng"], ["bao nhiêu", "còn"]],
         "hai câu, câu đầu hay bị bỏ"),
    Case("en", "I'll send the final scope by three. Does that work?",
         [["ba giờ", "3 giờ"], ["phạm vi"], ["được không", "ổn không", "có được"]],
         "mốc giờ hay bị hiểu thành ngày"),
    Case("en", "We probably won't get legal sign-off before Monday.",
         [["có lẽ", "chắc", "có thể"], ["không", "chưa"], ["pháp lý", "pháp chế"],
          ["thứ hai", "thứ 2"]],
         "từ giảm nhẹ + phủ định + tên riêng"),
    Case("en", "The vendor quoted forty percent more than last year.",
         [["bốn mươi", "40"], ["năm ngoái", "năm trước"], ["nhà cung cấp", "bên bán"]],
         "con số và mốc so sánh"),
    Case("en", "Don't ship it until QA signs off, even if we're late.",
         [["không", "đừng", "chưa"], ["qa", "kiểm thử", "kiểm định"],
          ["muộn", "trễ", "chậm"]],
         "phủ định mệnh lệnh + mệnh đề nhượng bộ"),
]

#: Chiều NGƯỢC: người dùng nói tiếng Việt, dịch sang tiếng đối phương.
OUTBOUND = [
    Case("vi", "Tôi lo là khách hàng sẽ không hài lòng nếu chậm thêm.",
         [["worried", "worry", "concerned", "afraid"], ["customer", "client"],
          ["not", "won't", "unhappy", "n't"]],
         "phủ định phải còn"),
    Case("vi", "Khoảng hai tuần thôi, sau đó là hết hạn hợp đồng.",
         [["two weeks"], ["contract"], ["expire", "end", "run out", "up"]],
         "con số + danh từ chính"),
    Case("vi", "Được, nhưng anh chốt giúp em phạm vi trước chiều nay nhé.",
         [["scope"],
          # "chiều nay" là buổi chiều, KHÔNG phải tối
          ["this afternoon", "afternoon", "today"],
          ["can you", "could you", "please", "would you"]],
         "phải giữ dạng NHỜ, không đảo thành đề nghị giúp"),
    Case("vi", "Nếu bên mình chậm quá hai tuần thì coi như mất hợp đồng.",
         [["two weeks"], ["contract"], ["lose", "lost", "forfeit"]],
         "mệnh đề điều kiện"),
    Case("vi", "Em chưa nhận được bản báo giá mới nhất từ nhà cung cấp.",
         [["not", "haven't", "hasn't", "no "], ["quote", "quotation", "pricing"],
          ["vendor", "supplier"]],
         "phủ định + hai danh từ"),
    Case("vi", "Anh cho em xin thêm hai ngày để rà lại phần kỹ thuật nhé.",
         [["two days", "two more days", "2 days", "another two"],
          ["technical", "engineering", "tech"],
          # "can you give me" cũng là dạng nhờ hợp lệ, không chỉ "can I ask"
          ["can i", "could i", "may i", "can you", "could you", "please", "ask for"]],
         "xin phép, không phải thông báo"),
]

ALL = INBOUND + OUTBOUND


def missing(case: Case, translation: str) -> list[list[str]]:
    """Các yếu tố KHÔNG còn trong bản dịch."""
    lowered = translation.lower()
    return [group for group in case.must_keep
            if not any(form.lower() in lowered for form in group)]
