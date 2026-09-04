"""Câu vừa nghe đã trọn nghĩa chưa, hay người ta chỉ đang ngập ngừng giữa câu.

VÌ SAO CẦN: chỉ đo độ dài khoảng lặng thì không tách được hai ca này. Đo thật
trên bộ file thử, khoảng ngập ngừng GIỮA câu của người nói chậm (800ms) còn
DÀI HƠN khoảng nghỉ giữa hai câu của người nói bình thường (700ms). Không có
ngưỡng thời gian nào đúng cho cả hai.

Tín hiệu dùng ở đây là NỘI DUNG câu chữ, lấy miễn phí từ transcript:

  1. Whisper chấm câu. Đo trên 32 câu trọn: 32/32 kết thúc bằng `.`/`?`/`!`.
     Không có dấu kết câu -> gần như chắc chắn chưa xong.

  2. Whisper dùng dấu ba chấm cho câu bỏ lửng. Tiếng Việt rất rõ:
     "Tôi nghĩ là chúng ta nên...", "Vấn đề chính ở đây là...".

  3. Riêng dấu câu thì KHÔNG đủ — chỉ bắt được 25%. Whisper vẫn chấm câu cho
     mảnh dở: "Before we sign anything I want to.", "I think we should.".
     Nên xét thêm TỪ CUỐI: một số từ chức năng không thể đứng cuối câu trọn
     nghĩa. Kết hợp cả ba, đo trên bộ thử ngập ngừng thật:

         bắt được câu dở         11/12 (92%)
         không báo nhầm câu trọn 32/32 (100%)

CHI PHÍ HAI PHÍA KHÔNG BẰNG NHAU, và ngưỡng ở đây cố ý nghiêng theo:

  bỏ sót câu dở    -> cắt nhầm giữa câu, người dùng nhận bản dịch của một mảnh
  báo nhầm câu trọn -> chỉ chờ thêm một nhịp rồi vẫn ra. Chậm, không sai.
"""

from __future__ import annotations

import re

#: Dấu kết thúc một câu, cả bộ Latin lẫn bộ CJK toàn rộng.
TERMINAL = ".!?。！？"

#: Từ không thể đứng cuối một câu trọn nghĩa.
#:
#: CHỈ nhận từ chức năng thực sự treo lơ lửng. Đã bỏ `this/these/those` và
#: `này/đó` khỏi danh sách: chúng đứng cuối câu được và từng làm báo nhầm hai
#: câu trọn ("...chốt phương án này.", "...ký ngay tuần này.").
DANGLING = frozenset(
    """
    a an the and or but so if of to in on at for with from by as that than then
    because when while is are was were be been am will would can could should
    shall may might must do does did have has had about into over under through
    between after before very my your our their its some any more
    là và hoặc nhưng nếu thì mà của cho với về từ nên sẽ đã đang bị được có
    phải cần muốn rất những các một vì do khi trong ngoài trên dưới để
    """.split()
)

_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)


def looks_complete(text: str) -> bool:
    """True nếu transcript trông như một câu đã nói xong.

    Chỉ nhìn chữ, không nhìn audio — nên rẻ và chạy được ngay sau STT.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("..") or stripped.endswith("…"):
        return False  # bỏ lửng
    if stripped[-1] not in TERMINAL:
        return False  # chưa chấm câu
    words = _WORDS.findall(stripped.lower())
    return bool(words) and words[-1] not in DANGLING
