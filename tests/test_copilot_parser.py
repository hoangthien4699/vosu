"""Parser semantic event.

Output LLM giờ chỉ còn MỘT trường `translation`. Cả `intent` (§4.4) lẫn
`replies` đều đã bỏ: máy không nghĩ hộ câu trả lời nữa — người dùng tự nói
bằng tiếng mình và máy dịch sang tiếng đối phương để họ nói theo.
"""
from __future__ import annotations

import json

import pytest

from app.ai.copilot import SemanticEventParser, TranslationDelta

VI = "Tôi nghĩ chúng ta nên tạm gác lại cuộc thảo luận này."
EN = "I'd like us to hold off on this for now."


def feed_all(text: str, size: int = 3):
    parser = SemanticEventParser()
    events = []
    for i in range(0, len(text), size):
        events.extend(parser.feed(text[i : i + size]))
    events.extend(parser.finish())
    return events, parser


def test_ban_dich_hien_dan_qua_delta():
    events, parser = feed_all(json.dumps({"translation": VI}, ensure_ascii=False))
    deltas = [e for e in events if isinstance(e, TranslationDelta)]
    assert deltas
    assert "".join(e.text for e in deltas) == VI
    assert parser.result.translation == VI
    assert not parser.result.malformed


def test_khong_bao_gio_lo_json_tho_ra_ngoai():
    """Contract §4.4 — lý do tồn tại của lớp parser ở backend."""
    events, _ = feed_all(json.dumps({"translation": VI}, ensure_ascii=False), size=1)
    leaked = "".join(e.text for e in events if isinstance(e, TranslationDelta))
    assert leaked == VI
    for forbidden in ("{", "}", '"translation"', '":'):
        assert forbidden not in leaked, f"JSON thô lọt ra: {forbidden!r}"


def test_delta_phat_ra_truoc_khi_chuoi_dong():
    parser = SemanticEventParser()
    events = list(parser.feed('{"translation": "Tôi nghĩ'))
    assert "".join(e.text for e in events if isinstance(e, TranslationDelta)) == "Tôi nghĩ"


def test_chieu_nguoc_cung_dung_mot_truong():
    """Dịch Việt -> Anh dùng chung schema; chiều do tầng trên quyết định."""
    _, parser = feed_all(json.dumps({"translation": EN}, ensure_ascii=False))
    assert parser.result.translation == EN


def test_bo_qua_markdown_fence():
    events, _ = feed_all('```json\n' + json.dumps({"translation": VI}, ensure_ascii=False) + '\n```')
    assert [e for e in events if isinstance(e, TranslationDelta)][-1].full == VI


def test_chap_nhan_ten_truong_cu():
    _, parser = feed_all(json.dumps({"trans": "Xin chào"}, ensure_ascii=False))
    assert parser.result.translation == "Xin chào"


def test_json_cut_giua_chung_van_dung_duoc():
    _, parser = feed_all('{"translation": "Tôi nghĩ chúng ta nên')
    assert parser.result.translation == "Tôi nghĩ chúng ta nên"
    assert parser.result.is_useful, "bản dịch cụt vẫn hữu ích, không được vứt"
    assert parser.result.malformed


def test_result_rong_thi_khong_useful():
    _, parser = feed_all("hoàn toàn không phải JSON")
    assert not parser.result.is_useful


# --------------------------------------------------------------------------- #
# Dọn rác cấu trúc do sinh có grammar
# --------------------------------------------------------------------------- #

def test_cat_rac_cau_truc_o_cuoi_gia_tri():
    """Grammar đảm bảo JSON đúng cấu trúc, nhưng `{`, `}` là ký tự HỢP LỆ bên
    trong chuỗi JSON nên không cấm được — model từng nhét `},{` vào cuối bản
    dịch rồi mới đóng chuỗi."""
    from app.ai.copilot import clean_value

    assert clean_value("Tôi nghĩ nên hoãn.},{") == "Tôi nghĩ nên hoãn."
    assert clean_value("Hết hạn rồi.”}”}”}”}”}”}") == "Hết hạn rồi."
    assert clean_value("Hai tuần.”}hơn") == "Hai tuần."


@pytest.mark.parametrize(
    "text",
    [
        # Dấu ngoặc KHỚP đúng thứ tự là chữ thật, không phải rác.
        "Bấm phím [ENTER] để tiếp tục.",
        "Bấm phím có nhãn [ENTER]",
        "Giá trị là {x}",
        'Anh ấy nói “được” rồi.',
        # Lặp từ là cách người ta nói, không phải model kẹt vòng lặp.
        "Hihihihihihi",
        "Dạ dạ dạ dạ dạ dạ",
        "Không không không không không",
        "Nó cứ lằng nhằng lằng nhằng.",
        # Câu thường
        "Chúng ta còn bao nhiêu thời gian?",
        "Ừ thì… cũng được…",
        "Một, hai, ba, bốn, năm",
    ],
)
def test_khong_dung_toi_chu_that(text):
    """clean_value ĐỔI NGHĨA nếu cắt nhầm, nên phải hẹp.

    Bản đầu chỉ nhìn ký tự ở đuôi và cắt hỏng bốn ca thật:
        "Bấm phím có nhãn [ENTER]" -> "...[ENTER"
        "Hihihihihihi"             -> "Hi"
        "Dạ dạ dạ dạ dạ dạ"        -> "Dạ"
        "Giá trị là {x}"           -> "Giá trị là {x"

    Dấu hiệu đúng của rác là dấu ĐÓNG KHÔNG CÓ dấu mở khớp, và mẩu lặp phải
    CHỨA ký tự cấu trúc.
    """
    from app.ai.copilot import clean_value

    assert clean_value(text) == text


def test_cat_het_thi_giu_nguyen():
    """Không bao giờ trả về rỗng — thà để lại rác còn hơn nuốt mất nội dung."""
    from app.ai.copilot import clean_value

    assert clean_value("}{") == "}{"


def test_translation_co_rac_duoc_sua_lai_qua_delta():
    """UI đã hiện phần rác qua delta, nên phải phát thêm một delta sửa lại."""
    events, parser = feed_all('{"translation":"Tôi nghĩ vậy.},{"}', size=4)
    deltas = [e for e in events if isinstance(e, TranslationDelta)]
    assert deltas[-1].full == "Tôi nghĩ vậy."
    assert parser.result.translation == "Tôi nghĩ vậy."


def test_cat_vong_lap_o_cuoi():
    """Quan sát thật: model kẹt sinh `”}”}”}...` tới hết n_predict.

    Grammar không cấm được — đó là ký tự hợp lệ BÊN TRONG chuỗi JSON. Và bộ ký
    tự chỉ có dấu ngoặc THẲNG sẽ trượt hoàn toàn, vì model dùng dấu CONG.
    """
    from app.ai.copilot import clean_value

    looped = 'About two weeks, then it’s the contract expiration.”}' * 1
    looped += '”}' * 20
    assert clean_value(looped) == "About two weeks, then it’s the contract expiration."


