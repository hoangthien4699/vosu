"""Bộ câu đo chất lượng bản dịch.

Hai thứ đo được bằng máy, và chúng bắt hai loại lỗi khác nhau:

  must_keep  Yếu tố BẮT BUỘC phải còn — con số, phủ định, từ giảm nhẹ, tên
             riêng, mốc giờ. Bắt lỗi BỎ SÓT:
                 "pushing the launch" -> "đẩy ra"        (mất "launch")

  forbid     Chuỗi KHÔNG được xuất hiện. Bắt lỗi ĐẢO NGHĨA — thứ mà kiểm tra
             "có chứa" một mình không thấy, vì "chắc chắn" chứa "chắc" nên nó
             ĐẠT bài kiểm tra cho "probably" trong khi nghĩa thì ngược lại:
                 "We probably won't get sign-off"
                     -> "Chắc chắn là chúng ta sẽ không..."

Còn "nghe có tự nhiên không" thì máy chỉ gợi ý được (STIFF_WORDS, PARTICLES);
phán quyết cuối vẫn phải là người nói tiếng đó.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    lang: str
    text: str
    #: Mỗi phần tử là một yếu tố phải giữ; bên trong là các dạng chấp nhận
    #: được — khớp MỘT dạng là đạt. Viết thường, so khớp không phân biệt hoa.
    must_keep: list[list[str]] = field(default_factory=list)
    #: Các chuỗi không được xuất hiện — dấu hiệu dịch ngược nghĩa.
    forbid: list[str] = field(default_factory=list)
    #: Câu đời thường: bản dịch sang tiếng Việt NÊN có tiểu từ tình thái.
    casual: bool = False
    note: str = ""


#: Từ "dịch máy" — đúng nghĩa nhưng không ai nói thế trong hội thoại.
STIFF_WORDS = [
    "mối quan ngại", "quan ngại", "sự nới lỏng", "cuộc phát hành",
    "vị cung cấp", "nhằm mục đích", "đối với việc", "trong trường hợp",
    "vận chuyển nó", "ngay cả khi", "cho đến khi mà", "thực hiện việc",
]

#: Tiểu từ tình thái — thứ làm câu nghe như người nói thật.
PARTICLES = ["nhé", "nha", "ạ", "thôi", "mà", "nhỉ", "đấy", "đâu", "chứ", "à", "vậy"]


# --------------------------------------------------------------------------- #
# Chiều THUẬN: đối phương nói, dịch sang tiếng Việt
# --------------------------------------------------------------------------- #

INBOUND = [
    Case(
        "en", "We're thinking about pushing the launch to next quarter.",
        must_keep=[["ra mắt", "phát hành", "khởi chạy"],
                   ["quý sau", "quý tới", "quý tiếp"]],
        note="danh từ chính hay bị nuốt",
    ),
    Case(
        "en", "That's a fair concern. How much slack do we have?",
        must_keep=[["hợp lý", "chính đáng", "xác đáng", "đúng"],
                   ["bao nhiêu", "còn"]],
        casual=True,
        note="hai câu, câu đầu hay bị bỏ",
    ),
    Case(
        "en", "I'll send the final scope by three. Does that work?",
        must_keep=[["ba giờ", "3 giờ"], ["phạm vi"],
                   ["được không", "ổn không", "có được"]],
        casual=True,
        note="mốc giờ hay bị hiểu thành ngày; 'scope' hay bị để nguyên",
    ),
    Case(
        "en", "We probably won't get legal sign-off before Monday.",
        must_keep=[["có lẽ", "có thể", "chắc là", "khả năng"],
                   ["không", "chưa"],
                   ["pháp lý", "pháp chế"],
                   ["trước thứ hai", "trước thứ 2"]],
        forbid=["chắc chắn", "nhất định"],
        note="từ giảm nhẹ hay bị lật thành khẳng định",
    ),
    Case(
        "en", "The vendor quoted forty percent more than last year.",
        must_keep=[["bốn mươi", "40"], ["năm ngoái", "năm trước"],
                   ["nhà cung cấp", "bên bán", "nhà thầu"]],
        note="con số và mốc so sánh",
    ),
    Case(
        "en", "Don't ship it until QA signs off, even if we're late.",
        must_keep=[["không", "đừng", "chưa", "khoan", "chớ", "hãy đợi", "chờ"],
                   ["qa", "kiểm thử", "kiểm định"],
                   ["muộn", "trễ", "chậm"]],
        casual=True,
        note="phủ định mệnh lệnh + mệnh đề nhượng bộ",
    ),
]


# --------------------------------------------------------------------------- #
# Chiều NGƯỢC: người dùng nói tiếng Việt, dịch sang tiếng đối phương
# --------------------------------------------------------------------------- #

OUTBOUND = [
    Case(
        "vi", "Tôi lo là khách hàng sẽ không hài lòng nếu chậm thêm.",
        must_keep=[["worried", "worry", "concerned", "afraid"],
                   ["customer", "client"],
                   ["not", "won't", "unhappy", "n't"]],
        note="phủ định phải còn",
    ),
    Case(
        "vi", "Khoảng hai tuần thôi, sau đó là hết hạn hợp đồng.",
        must_keep=[["two weeks"], ["contract"],
                   ["expire", "end", "run out", "up"]],
        note="con số + danh từ chính",
    ),
    Case(
        "vi", "Được, nhưng anh chốt giúp em phạm vi trước chiều nay nhé.",
        must_keep=[["scope"],
                   ["this afternoon", "afternoon", "today"],
                   ["can you", "could you", "please", "would you"]],
        forbid=["you'll help", "you will help"],
        note="phải giữ dạng NHỜ, không đảo thành 'bạn sẽ giúp tôi'",
    ),
    Case(
        "vi", "Nếu bên mình chậm quá hai tuần thì coi như mất hợp đồng.",
        must_keep=[["two weeks"], ["contract"],
                   ["lose", "lost", "forfeit", "void", "gone"]],
        note="mệnh đề điều kiện",
    ),
    Case(
        "vi", "Em chưa nhận được bản báo giá mới nhất từ nhà cung cấp.",
        must_keep=[["not", "haven't", "hasn't", "no "],
                   ["quote", "quotation", "pricing"],
                   ["vendor", "supplier"]],
        forbid=["you haven't", "you have not"],
        note="phủ định + hai danh từ; ngôi thứ nhất hay bị lật thành 'bạn'",
    ),
    Case(
        "vi", "Anh cho em xin thêm hai ngày để rà lại phần kỹ thuật nhé.",
        must_keep=[["two days", "two more days", "two extra days", "2 days"],
                   ["technical", "engineering", "tech"],
                   ["can i", "could i", "may i", "can you", "could you",
                    "please", "ask for"]],
        note="xin phép, không phải thông báo",
    ),
]

ALL = INBOUND + OUTBOUND


def missing(case: Case, translation: str) -> list[list[str]]:
    """Các yếu tố KHÔNG còn trong bản dịch."""
    lowered = translation.lower()
    return [group for group in case.must_keep
            if not any(form.lower() in lowered for form in group)]


def inverted(case: Case, translation: str) -> list[str]:
    """Các dấu hiệu bản dịch bị ĐẢO NGHĨA."""
    lowered = translation.lower()
    return [bad for bad in case.forbid if bad.lower() in lowered]


def stiff_hits(translation: str) -> list[str]:
    lowered = translation.lower()
    return [w for w in STIFF_WORDS if w in lowered]


def has_particle(translation: str) -> bool:
    padded = f" {translation.lower()} "
    return any(f" {p} " in padded or f" {p}." in padded or f" {p}?" in padded
               for p in PARTICLES)
