"""B3 — TTS riêng lẻ trên CPU. Target: < 300-400ms cho câu <10 từ (§6 mục 4)."""

from __future__ import annotations

import asyncio
import time

from .common import BenchmarkResult, Check, Distribution, base_parser, run_cli

SENTENCES = [
    "Tôi nghĩ chúng ta nên tạm dừng.",
    "Bạn có thể nói rõ hơn không?",
    "Được, tôi hiểu rồi.",
    "Chúng ta bàn lại vào tuần sau nhé.",
    "Vấn đề này cần thêm thời gian.",
]


async def _run(args) -> BenchmarkResult:
    from app.ai.tts import PiperTts, TtsUnavailable
    from app.core.config import load_config

    config = load_config()
    result = BenchmarkResult("B3", "TTS riêng lẻ (Piper, CPU)")

    engine = PiperTts(config)
    try:
        engine.preflight()
    except TtsUnavailable as exc:
        result.skipped = str(exc)
        return result

    runs = args.runs or 10
    first_audio: list[float] = []
    completions: list[float] = []
    total_bytes = 0

    for i in range(runs):
        sentence = SENTENCES[i % len(SENTENCES)]
        started = time.perf_counter()
        first_at: float | None = None
        async for chunk in engine.synthesize(f"bench_{i}", sentence):
            if first_at is None:
                first_at = (time.perf_counter() - started) * 1000.0
            total_bytes += len(chunk)
        completions.append((time.perf_counter() - started) * 1000.0)
        if first_at is not None:
            first_audio.append(first_at)

    await engine.close()

    first_dist = Distribution.of(first_audio)
    full_dist = Distribution.of(completions)
    result.checks = [
        Check("Time-to-first-audio P50", first_dist.p50, config.benchmark.tts_ms),
        Check("Time-to-first-audio P95", first_dist.p95, config.benchmark.tts_ms),
        Check("Tổng tổng hợp P95", full_dist.p95, None, mode="record"),
    ]
    result.details = {
        "time-to-first-audio": first_dist.summary(),
        "tổng tổng hợp": full_dist.summary(),
        "PCM sinh ra": f"{total_bytes / 1024:.0f} KB",
        "ghi chú": "TTS chạy CPU — không tính vào ngân sách VRAM 6GB (§3.1)",
    }
    return result


def main(args) -> BenchmarkResult:
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(run_cli(main, base_parser(__doc__)))
