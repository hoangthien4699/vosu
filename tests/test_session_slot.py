"""Nối lại ngay sau khi ngắt thì không được bị từ chối.

Kịch bản thật: phát xong một file, chọn file khác mà KHÔNG tải lại trang ->
"Đã đạt giới hạn 1 session đồng thời". `ws.close()` phía trình duyệt trả về
ngay, còn server thì phải hủy xong các worker mới nhả chỗ — kết nối mới tới
đúng vào khe đó.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from app.core.config import load_config
from app.core.runtime import ModelRuntime, SessionRejected


@pytest.fixture
def runtime(monkeypatch):
    rt = ModelRuntime(load_config())
    rt._ready = True

    async def ready():
        return None

    monkeypatch.setattr(rt, "ensure_ready", ready)
    return rt


def test_cho_cho_session_cu_nha_cho_thay_vi_tu_choi_ngay(runtime):
    runtime._config.session.max_concurrent_sessions = 1
    runtime._config.session.session_slot_wait_s = 3.0

    async def scenario():
        # Chiếm chỗ, và nhả sau 200ms — mô phỏng server đang dọn dẹp.
        holder = await _hold(runtime, 0.2)
        await asyncio.sleep(0.02)
        started = time.monotonic()
        async with runtime.session("moi"):
            waited = time.monotonic() - started
        await holder
        return waited

    waited = asyncio.run(scenario())
    assert 0.1 < waited < 3.0, f"đáng lẽ chờ rồi vào được, thực tế chờ {waited:.2f}s"


async def _hold(runtime, seconds: float):
    async def run():
        async with runtime.session("dang_giu"):
            await asyncio.sleep(seconds)

    return asyncio.create_task(run())


def test_van_tu_choi_khi_that_su_co_client_khac(runtime):
    """Chờ có giới hạn. Hết giờ mà vẫn đầy thì từ chối là câu trả lời đúng."""
    runtime._config.session.max_concurrent_sessions = 1
    runtime._config.session.session_slot_wait_s = 0.2

    async def scenario():
        async with runtime.session("client_khac"):
            with pytest.raises(SessionRejected):
                async with runtime.session("toi"):
                    pass

    asyncio.run(scenario())
