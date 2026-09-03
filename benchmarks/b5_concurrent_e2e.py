"""B5 — Concurrent Inference E2E. ĐÂY LÀ BENCHMARK QUAN TRỌNG NHẤT.

§7 (review v4.1) — vì sao bỏ metric "STT + LLM cộng dồn < 1.1s":

    Nếu STT và LLM chạy đồng thời và có overlap (STT 600ms chồng lấn một phần
    với LLM 700ms), tổng cộng dồn 600+700=1300ms KHÔNG phản ánh đúng E2E thực
    tế. Metric đúng là khoảng thời gian từ VAD endpoint đến khi có kết quả
    copilot ĐẦU TIÊN HỮU ÍCH.

Target: P50 < 1.0s · P90 < 1.3s · P95 < 1.5s · Max < 2.0s · error rate < 2%

§3.2: trên GPU 6GB (16-28 SMs), Whisper và llama-server chạy chồng lấn sẽ tranh
chấp Streaming Multiprocessors, có thể làm tốc độ mỗi bên tụt 30-50%. So sánh
kết quả ở đây với B1/B2 để lượng hóa mức suy giảm đó.
"""

from __future__ import annotations

import asyncio
import time

from .audio_fixtures import SYNTHETIC_WARNING, get_samples, uses_synthetic
from .common import BenchmarkResult, Check, Distribution, base_parser, run_cli


async def _run(args) -> BenchmarkResult:
    from app.ai.copilot import SemanticEventParser
    from app.ai.llm import GenerationStats, LlmClient
    from app.ai.stt import SttEngine
    from app.core.config import load_config
    from app.core.vram_manager import LlamaServerManager, query_vram

    config = load_config()
    targets = config.benchmark
    result = BenchmarkResult("B5", "Concurrent Inference E2E (VAD endpoint -> kết quả đầu tiên)")

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
            result.skipped = str(exc)
            return result

    try:
        engine.load_sync()
    except Exception as exc:
        await client.close()
        if started_here:
            await manager.stop()
        result.skipped = f"không nạp được Whisper: {exc}"
        return result

    runs = args.runs or targets.e2e_utterances
    samples = get_samples(runs, seconds=2.0)

    e2e_ms: list[float] = []
    stt_ms: list[float] = []
    ttft_ms: list[float] = []
    errors = 0
    peak_vram = 0.0

    async def one_utterance(pcm) -> None:
        """Mô phỏng đúng luồng thật: VAD endpoint -> final STT -> LLM -> kết quả."""
        nonlocal errors, peak_vram
        endpoint_at = time.perf_counter()
        try:
            transcript = await engine.transcribe(pcm, is_final=True)
            stt_ms.append(transcript.latency_ms)
            text = transcript.text or "I think we should table this discussion."

            parser = SemanticEventParser()
            stats = GenerationStats()
            first_useful: float | None = None

            async for token in client.stream(
                client.build_prompt(text, transcript.language), stats=stats
            ):
                for _event in parser.feed(token):
                    if first_useful is None and parser.result.is_useful:
                        first_useful = time.perf_counter()
                if first_useful is not None:
                    break        # đã có kết quả hữu ích đầu tiên — đúng metric §7

            if first_useful is None:
                for _event in parser.finish():
                    pass
                if not parser.result.is_useful:
                    errors += 1
                    return
                first_useful = time.perf_counter()

            if stats.ttft_ms is not None:
                ttft_ms.append(stats.ttft_ms)
            e2e_ms.append((first_useful - endpoint_at) * 1000.0)
        except Exception:
            errors += 1

    try:
        # warm-up không tính vào kết quả
        await one_utterance(samples[0][1])
        e2e_ms.clear()
        stt_ms.clear()
        ttft_ms.clear()
        errors = 0

        # Chồng lấn thật: khởi động utterance kế tiếp khi cái trước còn đang
        # sinh token — đây chính là kịch bản gây Compute Contention.
        pending: list[asyncio.Task] = []
        for _name, pcm in samples:
            task = asyncio.create_task(one_utterance(pcm))
            pending.append(task)
            await asyncio.sleep(0.35)      # người nói câu tiếp theo
            snapshot = query_vram()
            if snapshot.available:
                peak_vram = max(peak_vram, snapshot.used_gb)
        await asyncio.gather(*pending)
    finally:
        engine.unload()
        await client.close()
        if started_here:
            await manager.stop()

    total = len(e2e_ms) + errors
    error_rate = errors / total if total else 1.0
    distribution = Distribution.of(e2e_ms)

    result.checks = [
        Check("E2E P50", distribution.p50, targets.e2e_p50_ms),
        Check("E2E P90", distribution.p90, targets.e2e_p90_ms),
        Check("E2E P95", distribution.p95, targets.e2e_p95_ms),
        Check("E2E Max", distribution.max, targets.e2e_max_ms),
        Check("Error rate", error_rate * 100, targets.e2e_error_rate * 100, unit="%"),
    ]
    if peak_vram:
        result.checks.append(Check("VRAM đỉnh khi chạy", peak_vram, targets.vram_gb, unit="GB"))

    result.details = {
        "E2E": distribution.summary(),
        "STT trong lúc chồng lấn": Distribution.of(stt_ms).summary(),
        "TTFT trong lúc chồng lấn": Distribution.of(ttft_ms).summary(),
        "lỗi": f"{errors}/{total}",
        "so sánh": "Đối chiếu với B1/B2 để lượng hóa suy giảm do Compute Contention (§3.2)",
        "lưu ý": "P95 là con số quan trọng nhất, không phải P50 (§7)",
    }
    if uses_synthetic():
        result.details["cảnh báo"] = SYNTHETIC_WARNING
    return result


def main(args) -> BenchmarkResult:
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(run_cli(main, base_parser(__doc__)))
