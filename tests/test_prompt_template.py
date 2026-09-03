"""Prompt template theo họ model.

Dùng sai template không ném lỗi — model vẫn sinh chữ, chỉ là chất lượng tệ đi
một cách khó truy vết. Vì vậy phải khóa bằng test.
"""
from __future__ import annotations

import pytest

from app.ai.direction import Direction
from app.ai.llm import CHATML, GEMMA, build_prompt, resolve_template, system_prompt
from app.core.config import load_config


def test_chatml_dung_dinh_dang_cua_qwen():
    prompt = build_prompt("Hello there.", "en", CHATML)
    assert prompt.startswith("<|im_start|>system\n")
    assert "<|im_start|>user\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "live interpreter" in prompt
    assert "Hello there." in prompt


def test_gemma_gop_system_vao_luot_user():
    """Gemma KHÔNG có vai trò `system` — chỉ dẫn phải nằm trong lượt user."""
    prompt = build_prompt("Hello there.", "en", GEMMA)
    assert prompt.startswith("<start_of_turn>user\n")
    assert prompt.endswith("<start_of_turn>model\n")
    assert "system" not in prompt.split("\n")[0]
    assert "live interpreter" in prompt
    assert "Hello there." in prompt


def test_gemma_khong_tu_them_bos():
    """llama-server tokenize với add_special=true nên đã tự chèn BOS.

    Thêm tay nữa thành BOS kép — chất lượng tụt mà không có dấu hiệu gì.
    """
    assert "<bos>" not in build_prompt("x", "en", GEMMA)


def test_stop_token_khac_nhau_giua_hai_ho():
    assert CHATML.stop == ("<|im_end|>", "<|im_start|>")
    assert GEMMA.stop == ("<end_of_turn>", "<start_of_turn>")
    assert not set(CHATML.stop) & set(GEMMA.stop), (
        "hai họ dùng chung stop token thì nhận nhầm template sẽ không lộ ra"
    )


@pytest.mark.parametrize(
    "gguf,expected",
    [
        ("models/gemma-3-4b-it-q4_k_m.gguf", "gemma"),
        ("models/qwen2.5-3b-instruct-q4_k_m.gguf", "chatml"),
        ("/abs/path/GEMMA-3-4B-IT-Q4_K_M.GGUF", "gemma"),
    ],
)
def test_auto_suy_template_tu_ten_file(gguf, expected):
    cfg = load_config(env={})
    cfg.paths.llm_gguf = gguf
    cfg.llm.prompt_template = "auto"
    assert resolve_template(cfg).name == expected


def test_dat_tuong_minh_thang_auto_detect():
    cfg = load_config(env={})
    cfg.paths.llm_gguf = "models/gemma-3-4b-it-q4_k_m.gguf"
    cfg.llm.prompt_template = "chatml"
    assert resolve_template(cfg).name == "chatml"


def test_ten_template_sai_bao_loi_ro():
    cfg = load_config(env={})
    cfg.llm.prompt_template = "llama3"
    with pytest.raises(ValueError, match="prompt_template không hợp lệ"):
        resolve_template(cfg)


def test_model_la_bao_gio_cung_mac_dinh_chatml_kem_canh_bao(caplog):
    cfg = load_config(env={})
    cfg.paths.llm_gguf = "models/mot-model-nao-do.gguf"
    cfg.llm.prompt_template = "auto"
    assert resolve_template(cfg).name == "chatml"
    assert any("Không suy được prompt template" in r.message for r in caplog.records)


def test_prompt_template_dat_duoc_qua_bien_moi_truong():
    cfg = load_config(env={"VOSU_LLM__PROMPT_TEMPLATE": "gemma"})
    assert cfg.llm.prompt_template == "gemma"
    assert resolve_template(cfg).name == "gemma"


# --------------------------------------------------------------------------- #
# Ràng buộc JSON Schema
# --------------------------------------------------------------------------- #

def test_grammar_cam_ky_tu_cau_truc_trong_chuoi():
    """JSON Schema chỉ kiểm soát CẤU TRÚC, không kiểm soát nội dung chuỗi.

    `{`, `}` và dấu ngoặc kép cong đều hợp lệ bên trong chuỗi JSON, nên model
    viết `”}` giữa chuỗi rồi lảm nhảm tiếp mà JSON vẫn "hợp lệ". Đo với prompt
    có lịch sử: json_schema 0/6 sạch, GBNF 6/6.
    """
    from app.ai.llm import response_grammar

    grammar = response_grammar()
    assert "translation" in grammar
    # Lớp ký tự của `char` phải loại trừ mọi ký tự cấu trúc và ngoặc kép cong.
    char_rule = next(ln for ln in grammar.splitlines() if ln.startswith("char"))
    for forbidden in ("{", "}", "[", "]", "u201C", "u201D"):
        assert forbidden in char_rule, f"grammar không cấm {forbidden!r}: {char_rule}"


def test_payload_gui_kem_grammar():
    """Bật cờ mà không gửi lên server thì grammar vô tác dụng."""
    import inspect

    from app.ai.llm import LlmClient

    source = inspect.getsource(LlmClient.stream)
    assert 'payload["grammar"]' in source
    assert "cfg.grammar" in source


# --------------------------------------------------------------------------- #
# Prompt theo chiều dịch
# --------------------------------------------------------------------------- #

def test_chieu_thuan_dich_sang_tieng_nguoi_dung():
    prompt = system_prompt(Direction.TO_USER, user_language="vi")
    assert "into Vietnamese" in prompt
    assert "USER just spoke" not in prompt


def test_chieu_nguoc_dich_sang_tieng_doi_phuong():
    prompt = system_prompt(
        Direction.TO_COUNTERPART, user_language="vi", counterpart_language="en"
    )
    assert "spoke in Vietnamese" in prompt
    assert "into English" in prompt
    assert "say it out loud" in prompt


def test_chieu_nguoc_theo_dung_tieng_doi_phuong_that():
    """Đích lấy theo ngôn ngữ nghe được, không cứng là tiếng Anh."""
    prompt = system_prompt(
        Direction.TO_COUNTERPART, user_language="vi", counterpart_language="ja"
    )
    assert "into Japanese" in prompt


def test_prompt_danh_dau_ai_dang_noi():
    to_user = build_prompt("Hello.", "en", GEMMA, direction=Direction.TO_USER)
    to_them = build_prompt("Xin chào.", "vi", GEMMA, direction=Direction.TO_COUNTERPART)
    assert "Now Them said:" in to_user
    assert "Now You said:" in to_them


def test_lich_su_nam_trong_system_prompt_de_giu_prefix_cache():
    """Lịch sử phải ở cuối system prompt, TRƯỚC câu hiện tại.

    Đặt sau câu hiện tại thì mỗi lượt tiền tố đều đổi và prefix cache vô dụng —
    đo được: 15 token/86ms so với 241 token/610ms.
    """
    prompt = build_prompt(
        "Now what?", "en", GEMMA,
        direction=Direction.TO_USER,
        history='Them: "Chúng ta nên hoãn lại."',
    )
    assert prompt.index("Chúng ta nên hoãn lại") < prompt.index("Now Them said:")


def test_khong_co_lich_su_thi_khong_co_khoi_context():
    prompt = build_prompt("Hello.", "en", GEMMA)
    assert "Earlier in this conversation" not in prompt


def test_ten_ngon_ngu_dich_sang_tieng_anh_cho_model_hieu():
    from app.ai.llm import language_name

    assert language_name("vi") == "Vietnamese"
    assert language_name("en-US") == "English"
    assert language_name("ja") == "Japanese"
    assert language_name(None) == "English"
    assert language_name("xx") == "xx"      # không biết thì giữ nguyên mã
