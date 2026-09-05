"""Bản giả trong test phải KHỚP CHỮ KÝ với lớp thật.

Đã trả giá: `FakeLlm.build_prompt` nhận `retry_hint`, còn `LlmClient.build_prompt`
thì không. Bản giả rộng hơn bản thật nên mọi test đều xanh, trong khi production
ném TypeError mỗi lần lưới an toàn `retry_on_bad_translation` được kích hoạt —
lỗi bị `except Exception` của worker nuốt, câu dịch hỏng chết lặng thay vì được
dịch lại.

Bật mặc định từ đầu mà chưa lần nào chạy được. Chỉ lộ ra khi thử một model yếu
hơn, đủ để sinh bản dịch hỏng thật.
"""
from __future__ import annotations

import inspect

from app.ai.direction import Direction
from app.ai.llm import LlmClient
from app.core.config import load_config


def test_build_prompt_chuyen_tiep_retry_hint():
    """Gợi ý dịch lại phải THẬT SỰ vào được prompt, không chỉ được nhận."""
    client = LlmClient.__new__(LlmClient)
    client._config = load_config()
    client.template = __import__(
        "app.ai.llm", fromlist=["CHATML"]
    ).CHATML

    hint = "DAU_HIEU_DICH_LAI_DUY_NHAT"
    prompt = LlmClient.build_prompt(
        client, "I think we should wait.", "en",
        direction=Direction.TO_USER, retry_hint=hint,
    )
    assert hint in prompt, "retry_hint bị nuốt — lưới an toàn thành vô dụng"

    without = LlmClient.build_prompt(
        client, "I think we should wait.", "en", direction=Direction.TO_USER,
    )
    assert hint not in without


def test_ban_gia_khong_duoc_rong_hon_ban_that():
    """Bản giả nhận tham số mà bản thật không có = test xanh, production hỏng."""
    from tests.test_websocket_e2e import FakeLlm

    real = set(inspect.signature(LlmClient.build_prompt).parameters) - {"self"}
    fake = set(inspect.signature(FakeLlm.build_prompt).parameters) - {"self"}

    thua = fake - real
    assert not thua, (
        f"FakeLlm nhận tham số mà LlmClient không có: {sorted(thua)}. "
        "Test sẽ xanh trong khi production ném TypeError."
    )
