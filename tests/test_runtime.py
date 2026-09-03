"""Khóa contract vòng đời §3.1: model là hạ tầng dùng chung, session là consumer."""
from __future__ import annotations

import asyncio

import pytest

from app.core.config import load_config
from app.core.runtime import ModelRuntime, SessionRejected


@pytest.fixture
def runtime(monkeypatch):
    cfg = load_config(env={})
    rt = ModelRuntime(cfg)

    # thay model thật bằng no-op: test này chỉ quan tâm vòng đời
    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(rt.llama, "start", noop)
    monkeypatch.setattr(rt.llama, "stop", noop)
    monkeypatch.setattr(rt.llm, "start", noop)
    monkeypatch.setattr(rt.llm, "close", noop)
    monkeypatch.setattr(rt.stt, "load", noop)
    monkeypatch.setattr(rt.stt, "unload", lambda: None)
    return rt


async def test_session_ngat_ket_noi_khong_ha_model(runtime):
    await runtime.start()
    async with runtime.session("sess_a"):
        pass
    assert runtime.is_ready, "model bị hạ khi session thoát — sai contract §3.1"
    assert runtime.active_sessions == 0
    await runtime.shutdown()
    assert not runtime.is_ready


async def test_shutdown_cho_inference_job_hoan_tat(runtime):
    """Contract cốt lõi: không terminate khi còn job đang dùng model."""
    await runtime.start()
    finished = []

    async def long_job():
        async with runtime.job("stt"):
            await asyncio.sleep(0.15)
            finished.append("job xong")

    task = asyncio.create_task(long_job())
    await asyncio.sleep(0.02)
    assert runtime.active_jobs == 1

    await runtime.shutdown(drain_timeout=5.0)

    assert finished == ["job xong"], "shutdown hạ model trước khi job kết thúc"
    await task


async def test_job_song_sot_qua_viec_session_thoat(runtime):
    """Client A ngắt kết nối trong lúc job của nó còn chạy — job phải chạy tiếp."""
    await runtime.start()
    done = asyncio.Event()

    async def job_of_a():
        async with runtime.job():
            await asyncio.sleep(0.1)
            done.set()

    async with runtime.session("sess_a"):
        task = asyncio.create_task(job_of_a())
        await asyncio.sleep(0.01)
    # session A đã thoát nhưng job còn chạy
    assert runtime.active_sessions == 0
    assert runtime.active_jobs == 1
    assert runtime.is_ready

    await asyncio.wait_for(done.wait(), timeout=2.0)
    await task
    await runtime.shutdown()


async def test_gioi_han_session_dong_thoi(runtime):
    await runtime.start()
    async with runtime.session("sess_a"):
        with pytest.raises(SessionRejected, match="giới hạn"):
            async with runtime.session("sess_b"):
                pass
    # sau khi A thoát thì B vào được
    async with runtime.session("sess_b"):
        assert runtime.active_sessions == 1
    await runtime.shutdown()


async def test_shutdown_van_ket_thuc_khi_job_treo(runtime):
    await runtime.start()

    async def stuck():
        async with runtime.job():
            await asyncio.sleep(10)

    task = asyncio.create_task(stuck())
    await asyncio.sleep(0.02)
    await runtime.shutdown(drain_timeout=0.1)   # không được treo vĩnh viễn
    assert not runtime.is_ready
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_dem_job_dong_thoi_dat_dinh(runtime):
    await runtime.start()

    async def job():
        async with runtime.job():
            await asyncio.sleep(0.05)

    await asyncio.gather(*(job() for _ in range(4)))
    assert runtime.stats.peak_concurrent_jobs == 4
    assert runtime.active_jobs == 0
    await runtime.shutdown()
