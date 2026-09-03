from __future__ import annotations

import json

import pytest

from app.ai.json_stream import (
    IncrementalJsonParser,
    StringDelta,
    ValueDone,
)

SAMPLE = {
    "translation": "Tôi nghĩ chúng ta nên tạm gác lại cuộc thảo luận này.",
    "intent": "delay discussion",
    "replies": ["Understood. When can we revisit?", "Is there a blocker?"],
}


def run(chunks) -> tuple[list, IncrementalJsonParser]:
    parser = IncrementalJsonParser()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    return events, parser


def values(events) -> dict:
    return {e.path: e.value for e in events if isinstance(e, ValueDone)}


def test_parse_nguyen_khoi():
    events, parser = run([json.dumps(SAMPLE, ensure_ascii=False)])
    vals = values(events)
    assert vals[("translation",)] == SAMPLE["translation"]
    assert vals[("intent",)] == "delay discussion"
    assert vals[("replies", 0)] == SAMPLE["replies"][0]
    assert vals[("replies", 1)] == SAMPLE["replies"][1]
    assert not parser.malformed


@pytest.mark.parametrize("size", [1, 2, 3, 5, 13])
def test_ket_qua_khong_phu_thuoc_cach_cat_token(size):
    """LLM cắt token tùy ý — parse phải cho kết quả y hệt."""
    text = json.dumps(SAMPLE, ensure_ascii=False)
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    events, parser = run(chunks)
    assert values(events)[("translation",)] == SAMPLE["translation"]
    assert not parser.malformed


def test_string_delta_cong_don_dung_bang_gia_tri_cuoi():
    text = json.dumps(SAMPLE, ensure_ascii=False)
    events, _ = run([text[i : i + 3] for i in range(0, len(text), 3)])
    accumulated = "".join(
        e.text for e in events if isinstance(e, StringDelta) and e.path == ("translation",)
    )
    assert accumulated == SAMPLE["translation"]


def test_delta_phat_ra_truoc_khi_chuoi_dong():
    """Đây là toàn bộ lý do tồn tại của parser: có text trước khi JSON hoàn tất."""
    parser = IncrementalJsonParser()
    events = list(parser.feed('{"translation": "Tôi nghĩ'))
    deltas = [e for e in events if isinstance(e, StringDelta)]
    assert "".join(e.text for e in deltas) == "Tôi nghĩ"
    assert not [e for e in events if isinstance(e, ValueDone)]


def test_bo_qua_markdown_fence_va_loi_dan():
    """Qwen hay bọc ```json dù prompt đã cấm."""
    text = 'Đây là kết quả:\n```json\n' + json.dumps(SAMPLE, ensure_ascii=False) + '\n```'
    events, _ = run([text])
    assert values(events)[("intent",)] == "delay discussion"


def test_json_bi_cat_giua_chung_van_giao_phan_da_co():
    """`n_predict` cắt ngang — thà giao bản dịch cụt còn hơn vứt hết."""
    events, parser = run(['{"translation": "Tôi nghĩ chúng ta'])
    assert values(events)[("translation",)] == "Tôi nghĩ chúng ta"
    assert parser.malformed


def test_dau_ngoac_kep_escape_khong_lam_dut_chuoi_som():
    events, parser = run([r'{"translation": "Anh ấy nói \"được\" rồi"}'])
    assert values(events)[("translation",)] == 'Anh ấy nói "được" rồi'
    assert not parser.malformed


def test_xuong_dong_va_backslash_escape():
    events, _ = run([r'{"t": "dòng1\ndòng2\\hết"}'])
    assert values(events)[("t",)] == "dòng1\ndòng2\\hết"


def test_mang_rong_va_object_rong():
    events, parser = run(['{"replies": [], "meta": {}}'])
    assert not parser.malformed
    assert parser.finished


def test_replies_dang_object_van_lay_duoc_text():
    """Model đôi khi lờ prompt và sinh lại cấu trúc v1 có `tone`/`meaning`."""
    payload = '{"replies": [{"tone": "Lịch sự", "text": "Understood."}]}'
    events, _ = run([payload])
    assert values(events)[("replies", 0, "text")] == "Understood."


def test_so_va_boolean():
    events, _ = run(['{"n": 42, "f": 1.5, "b": false, "z": null}'])
    vals = values(events)
    assert vals[("n",)] == 42 and vals[("f",)] == 1.5
    assert vals[("b",)] is False and vals[("z",)] is None


def test_object_long_nhau():
    events, _ = run(['{"a": {"b": {"c": "sâu"}}}'])
    assert values(events)[("a", "b", "c")] == "sâu"


def test_khong_no_khi_gap_rac_hoan_toan():
    events, parser = run(["đây không phải JSON gì cả"])
    assert events == [] and not parser.malformed  # chưa từng vào object -> không có gì để hỏng


def test_dung_lai_sau_khi_object_goc_dong():
    """Text sau JSON (model nói thêm) không được làm hỏng kết quả."""
    events, parser = run(['{"intent": "ok"}\nHy vọng giúp được bạn!'])
    assert values(events)[("intent",)] == "ok"
    assert parser.finished and not parser.malformed
