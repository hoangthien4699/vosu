"""B4 — VRAM khi Whisper + Qwen cùng active. Target: < 5.5GB hard ceiling.

§3.1 (review v2): KHÔNG lấy 5.1GB/6GB làm "design contract an toàn". VRAM thực
tế còn phụ thuộc CUDA context, allocator, KV cache, batch size, kiến trúc GPU,
phiên bản CUDA và memory fragmentation.
"""

from __future__ import annotations

import asyncio
import time

from .audio_fixtures import get_samples
from .common import BenchmarkResult, Check, base_parser, run_cli


async def _run(args) -> BenchmarkResult:
    from app.ai.llm import LlmClient, build_prompt
    from app.ai.stt import SttEngine
    from app.core.config import load_config
    from app.core.vram_manager import LlamaServerManager, query_vram

    config = load_config()
    result = BenchmarkResult("B4", "VRAM khi Whisper + LLM cùng active")

    baseline = query_vram()
    if not baseline.available:
        result.skipped = (
            f"{baseline.reason}. Hard ceiling 5.5GB chỉ có nghĩa trên GPU rời — "
            "chạy lại benchmark này trên máy NVIDIA."
        )
        return result

    manager = LlamaServerManager(config)
    client = LlmClient(config)
    engine = SttEngine(config)
    started_here = False

    await client.start()
    if not await client.health():
        try:
            await manager.start()
            started_here = True
        except Exception as exc:
            await client.close()
            result.error = str(exc)
            return result

    after_llm = query_vram()

    try:
        engine.load_sync()
    except Exception as exc:
        await client.close()
        if started_here:
            await manager.stop()
        result.error = f"không nạp được Whisper: {exc}"
        return result

    after_both = query_vram()

    # Đo lúc CẢ HAI đang thực sự chạy inference — đây mới là đỉnh thật, vì
    # KV cache và bộ đệm inference chỉ được cấp phát khi chạy.
    peak = after_both.used_gb
    samples = get_samples(3, seconds=2.0)

    async def hammer_llm() -> None:
        for _ in range(3):
            await client.complete(build_prompt("Let's discuss this later.", "en"))

    llm_task = asyncio.create_task(hammer_llm())
    for _name, pcm in samples:
        engine.transcribe_sync(pcm, is_final=True)
        peak = max(peak, query_vram().used_gb)
    await llm_task
    peak = max(peak, query_vram().used_gb)

    engine.unload()
    await client.close()
    if started_here:
        await manager.stop()
    time.sleep(1.0)
    after_cleanup = query_vram()

    ceiling = config.benchmark.vram_gb
    result.checks = [
        Check("VRAM đỉnh (cả hai active)", peak, ceiling, unit="GB"),
        Check("VRAM sau khi giải phóng", after_cleanup.used_gb, baseline.used_gb + 0.3,
              unit="GB", note="phát hiện rò rỉ"),
    ]
    result.details = {
        "nền (trước khi nạp)": f"{baseline.used_gb:.2f}GB / {baseline.total_gb:.2f}GB",
        "sau khi nạp LLM": f"{after_llm.used_gb:.2f}GB",
        "sau khi nạp cả Whisper": f"{after_both.used_gb:.2f}GB",
        "đỉnh khi đang inference": f"{peak:.2f}GB",
        "sau khi dọn": f"{after_cleanup.used_gb:.2f}GB",
        "expected theo đặc tả": f"{config.vram.expected_gb:.1f}GB",
    }
    return result


def main(args) -> BenchmarkResult:
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(run_cli(main, base_parser(__doc__)))
