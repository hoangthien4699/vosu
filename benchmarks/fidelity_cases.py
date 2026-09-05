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


# --------------------------------------------------------------------------- #
# Bộ KHÓ — thêm khi bộ cũ chạm trần
#
# Bộ ban đầu đo ra 34/35 yếu tố, 0 câu đảo nghĩa, 0 từ khô. Chạm trần thì không
# so được cái gì với cái gì nữa: mọi thay đổi đều ra cùng một điểm, và ta lại
# rơi vào cảnh đo nhiễu như hồi chỉnh temperature.
#
# Các loại lỗi ở đây là thứ model 4B thật sự còn vấp:
#   thành ngữ, phạm vi phủ định, tình thái (might/should/must),
#   xưng hô tiếng Việt, câu nhiều mệnh đề, chủ ngữ mơ hồ.
#
# CẢNH BÁO ĐÃ TRẢ GIÁ: lần chạy đầu báo 7 ca hỏng, nhưng 4 trong đó là lỗi của
# CHÍNH BỘ THỬ chứ không phải của model — "tạm bỏ" dịch đúng `table that`,
# "chậm chạp" dịch đúng `dragging their feet`, "reschedule" dịch đúng "lùi
# lịch", và "không thể LÀ không nói" chỉ thừa một chữ làm lệch khớp chuỗi.
#
# `must_keep` phải liệt kê MỌI cách dịch đúng, không phải cách mình nghĩ ra
# đầu tiên. Siết quá thì tối ưu model theo một cái thước cong.
# --------------------------------------------------------------------------- #

HARD_INBOUND = [
    Case(
        "en", "Let's table that and circle back after lunch.",
        must_keep=[["hoãn", "gác lại", "để sau", "tạm dừng", "gác", "tạm bỏ",
                    "bỏ qua", "dừng"],
                   ["quay lại", "bàn lại", "trở lại", "nói tiếp"]],
        forbid=["đặt lên bàn", "đặt bàn", "lên bàn"],
        casual=True,
        note="hai thành ngữ trong một câu; dịch đen là 'đặt lên bàn'",
    ),
    Case(
        "en", "We can't not tell them at this point.",
        must_keep=[["phải nói", "không thể không nói", "không thể là không nói",
                    "buộc phải", "đành phải"]],
        forbid=["không được nói cho họ", "không thể nói cho họ biết"],
        note="phủ định kép — dịch sai thành cấm nói",
    ),
    Case(
        "en", "You might want to double-check that figure before Friday.",
        must_keep=[["nên", "thử"], ["kiểm tra lại", "xem lại", "rà lại"],
                   ["thứ sáu", "thứ 6"]],
        forbid=["bắt buộc phải", "nhất định phải"],
        note="might ≠ must; tình thái hay bị đẩy lên thành mệnh lệnh",
    ),
    Case(
        "en", "Cut it from ninety days down to forty-five.",
        must_keep=[["90", "chín mươi"], ["45", "bốn mươi lăm", "bốn lăm"]],
        forbid=["49", "54", "40 lăm"],
        note="hai số gần nhau, dễ đổi chỗ hoặc gộp",
    ),
    Case(
        "en", "They're dragging their feet on the contract.",
        must_keep=[["chần chừ", "trì hoãn", "kéo dài", "dây dưa", "lần lữa",
                    "chậm trễ", "câu giờ", "chậm chạp", "ì ạch"]],
        forbid=["kéo chân", "lê chân", "kéo lê bàn chân"],
        note="thành ngữ; dịch đen thành động tác chân",
    ),
    Case(
        "en", "If the vendor misses the date again, we escalate to their "
              "director, and I want it in writing this time.",
        must_keep=[["trễ", "lỡ", "chậm", "không kịp"],
                   ["giám đốc"],
                   ["văn bản", "viết", "giấy tờ"],
                   ["lần này"]],
        note="ba mệnh đề; vế cuối hay bị nuốt",
    ),
    Case(
        "en", "If we had signed last month, this wouldn't be a problem.",
        must_keep=[["tháng trước"], ["ký"]],
        forbid=["tháng tới", "tháng sau"],
        note="giả định trái thực tế ở quá khứ",
    ),
    Case(
        "en", "Would you mind holding off until I hear back from them?",
        must_keep=[["chờ", "hoãn", "đợi", "khoan"]],
        forbid=["bạn có phiền không"],
        casual=True,
        note="lời đề nghị lịch sự; dịch đen ra câu hỏi ngớ ngẩn",
    ),
]

HARD_OUTBOUND = [
    Case(
        "vi", "Dạ anh cho em hỏi bên mình có thể lùi lịch một tuần được không ạ?",
        must_keep=[["push", "postpone", "delay", "move", "put off",
                    "reschedule", "push back"],
                   ["a week", "one week"]],
        forbid=["a month", "two weeks"],
        note="xưng hô anh/em không có tương đương; phải ra câu hỏi lịch sự",
    ),
    Case(
        "vi", "Không phải em không muốn làm, mà là không kịp.",
        must_keep=[["not that", "it isn't that", "it's not"],
                   ["time", "enough", "in time", "too late", "make it"]],
        forbid=["i don't want to do it."],
        note="phủ định kép; dịch sai thành từ chối thẳng",
    ),
    Case(
        "vi", "Bên đó bảo là bên mình phải chịu phí vận chuyển.",
        must_keep=[["shipping", "freight", "delivery", "transport"],
                   ["we", "our", "us"]],
        forbid=["they have to pay", "they will cover"],
        note="hai bên trong một câu; dễ đảo ai trả tiền",
    ),
    Case(
        "vi", "Thôi cứ để đấy đi, mai em xử lý nhé.",
        must_keep=[["tomorrow"]],
        forbid=["today", "tonight"],
        note="câu đời thường, nhiều tiểu từ ở nguồn",
    ),
    Case(
        "vi", "Giảm từ chín mươi ngày xuống còn bốn mươi lăm.",
        must_keep=[["90", "ninety"], ["45", "forty-five", "forty five"]],
        forbid=["49", "54"],
        note="đối xứng với ca tiếng Anh, kiểm cả hai chiều",
    ),
    Case(
        "vi", "Nếu tháng trước mình ký rồi thì giờ đã không phải lo.",
        must_keep=[["last month"], ["sign"]],
        forbid=["next month"],
        note="giả định trái thực tế, chiều ngược",
    ),
]

ALL = INBOUND + OUTBOUND + HARD_INBOUND + HARD_OUTBOUND


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
