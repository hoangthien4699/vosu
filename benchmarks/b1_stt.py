"""B1 — STT riêng lẻ. Target: < 400ms (§6 mục 2)."""

from __future__ import annotations

from .audio_fixtures import SYNTHETIC_WARNING, get_samples, uses_synthetic
from .common import BenchmarkResult, Check, Distribution, base_parser, run_cli


def main(args) -> BenchmarkResult:
    from app.core.config import load_config

    config = load_config()
    result = BenchmarkResult("B1", "STT riêng lẻ (Faster-Whisper)")

    try:
        from app.ai.stt import SttEngine
    except Exception as exc:
        result.skipped = f"không import được faster-whisper: {exc}"
        return result

    engine = SttEngine(config)
    try:
        engine.load_sync()
    except Exception as exc:
        result.skipped = str(exc)
        return result

    runs = args.runs or 10
    samples = get_samples(runs, seconds=2.0)

    # Lần đầu luôn chậm bất thường (cấp phát bộ đệm, JIT kernel) — không tính.
    engine.transcribe_sync(samples[0][1], is_final=True)

    latencies: list[float] = []
    transcripts: list[str] = []
    for _name, pcm in samples:
        transcript = engine.transcribe_sync(pcm, is_final=True)
        latencies.append(transcript.latency_ms)
        transcripts.append(transcript.text)

    distribution = Distribution.of(latencies)
    result.checks = [
        Check("STT P50", distribution.p50, config.benchmark.stt_ms),
        Check("STT P95", distribution.p95, config.benchmark.stt_ms),
    ]
    result.details = {
        "phân bố": distribution.summary(),
        "thời gian nạp model": f"{engine.load_ms:.0f}ms",
        "audio mỗi mẫu": f"{samples[0][1].size / 16000:.2f}s",
        "transcript mẫu": (transcripts[0][:80] or "(rỗng)"),
    }
    if uses_synthetic():
        result.details["cảnh báo"] = SYNTHETIC_WARNING

    engine.unload()
    return result


if __name__ == "__main__":
    raise SystemExit(run_cli(main, base_parser(__doc__)))
