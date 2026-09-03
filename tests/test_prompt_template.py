"""Prompt template theo họ model.

Dùng sai template không ném lỗi — model vẫn sinh chữ, chỉ là chất lượng tệ đi
một cách khó truy vết. Vì vậy phải khóa bằng test.
"""
from __future__ import annotations

import pytest

from app.ai.llm import CHATML, GEMMA, SYSTEM_PROMPT, build_prompt, resolve_template
from app.core.config import load_config


def test_chatml_dung_dinh_dang_cua_qwen():
    prompt = build_prompt("Hello there.", "en", CHATML)
    assert prompt.startswith("<|im_start|>system\n")
    assert "<|im_start|>user\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert SYSTEM_PROMPT in prompt
    assert "Hello there." in prompt


def test_gemma_gop_system_vao_luot_user():
    """Gemma KHÔNG có vai trò `system` — chỉ dẫn phải nằm trong lượt user."""
    prompt = build_prompt("Hello there.", "en", GEMMA)
    assert prompt.startswith("<start_of_turn>user\n")
    assert prompt.endswith("<start_of_turn>model\n")
    assert "system" not in prompt.split("\n")[0]
    assert SYSTEM_PROMPT in prompt
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
