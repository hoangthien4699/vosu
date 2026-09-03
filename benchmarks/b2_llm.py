"""B2 — LLM riêng lẻ. Target: TTFT < 200ms, total < 500ms (§6 mục 3).

§7 (review v2.1): TTFT phải đo RIÊNG. Streaming giảm perceived latency chứ
không nhất thiết giảm compute latency tổng — "LLM total < 500ms" không nói lên
được UX có nhanh hay không.
"""

from __future__ import annotations

import asyncio

from .common import BenchmarkResult, Check, Distribution, base_parser, run_cli

PROMPTS = [
    ("en", "I think we should table this discussion for now."),
    ("en", "Could you walk me through the pricing structure again?"),
    ("ja", "その件については社内で確認させていただきます。"),
    ("zh", "这个方案我们需要再讨论一下。"),
    ("en", "Let's circle back on this next week."),
]


async def _run(args) -> BenchmarkResult:
    from app.ai.llm import LlmClient
    from app.core.config import load_config
    from app.core.vram_manager import LlamaServerManager

    config = load_config()
    result = BenchmarkResult("B2", "LLM riêng lẻ (TTFT + tổng thời gian sinh)")

    manager = LlamaServerManager(config)
    client = LlmClient(config)
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

    runs = args.runs or 10
    ttfts: list[float] = []
    totals: list[float] = []
    token_counts: list[int] = []
    sample_output = ""

    try:
        # warm-up: lần đầu phải nạp KV cache + prompt cache
        language, text = PROMPTS[0]
        await client.complete(client.build_prompt(text, language))

        for i in range(runs):
            language, text = PROMPTS[i % len(PROMPTS)]
            output, stats = await client.complete(client.build_prompt(text, language))
            if stats.ttft_ms is not None:
                ttfts.append(stats.ttft_ms)
            totals.append(stats.total_ms)
            token_counts.append(stats.tokens)
            if not sample_output:
                sample_output = output[:120]
    finally:
        await client.close()
        if started_here:
            await manager.stop()

    ttft_dist = Distribution.of(ttfts)
    total_dist = Distribution.of(totals)
    result.checks = [
        Check("TTFT P50", ttft_dist.p50, config.benchmark.llm_ttft_ms),
        Check("TTFT P95", ttft_dist.p95, config.benchmark.llm_ttft_ms),
        Check("Tổng sinh P50", total_dist.p50, config.benchmark.llm_total_ms),
        Check("Tổng sinh P95", total_dist.p95, config.benchmark.llm_total_ms),
    ]
    result.details = {
        "TTFT": ttft_dist.summary(),
        "tổng": total_dist.summary(),
        "token trung bình": f"{sum(token_counts) / max(1, len(token_counts)):.0f}",
        "output mẫu": sample_output or "(rỗng)",
    }
    return result


def main(args) -> BenchmarkResult:
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(run_cli(main, base_parser(__doc__)))
