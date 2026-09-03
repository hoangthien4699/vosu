"""B6 — CPU stress: STT + LLM + TTS cùng chạy.

§2.4 (review v4.1) phân biệt HAI loại contention, và đây là chỗ đo cả hai:

  - Event-loop contention: gọi hàm blocking thẳng trong async route sẽ chặn
    event loop dù được gọi là "async". Đo bằng event-loop lag.
  - CPU contention: dù không chặn event loop, TTS vẫn cạnh tranh CPU với
    Whisper (phần CPU-side), Python runtime và network I/O.
"""

from __future__ import annotations

import asyncio
import time

from .audio_fixtures import get_samples
from .common import BenchmarkResult, Check, Distribution, base_parser, run_cli

TTS_SENTENCES = [
    "Tôi nghĩ chúng ta nên tạm dừng việc này.",
    "Bạn có thể nói rõ hơn được không?",
    "Được rồi, tôi hiểu ý anh.",
]


class EventLoopMonitor:
    """Đo độ trễ của event loop.

    Lên lịch đánh thức mỗi `interval`; phần vượt quá interval chính là thời
    gian event loop bị một coroutine nào đó giữ không nhả.
    """

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.lags_ms: list[float] = []
        self._running = False
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while self._running:
            expected = time.perf_counter() + self.interval
            await asyncio.sleep(self.interval)
            self.lags_ms.append(max(0.0, (time.perf_counter() - expected) * 1000.0))

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


async def _run(args) -> BenchmarkResult:
    from app.ai.llm import LlmClient
    from app.ai.stt import SttEngine
    from app.ai.tts import PiperTts, TtsUnavailable
    from app.core.config import load_config
    from app.core.vram_manager import LlamaServerManager

    config = load_config()
    result = BenchmarkResult("B6", "CPU stress (STT + LLM + TTS đồng thời)")

    # Tắt pacing: cần đo tranh chấp CPU ở TRƯỜNG HỢP XẤU NHẤT. Pacing chèn
    # khoảng chờ giữa các chunk, tức là tự giảm tải — bật lên sẽ làm nhẹ đi
    # đúng thứ benchmark này muốn phơi ra.
    config.tts.realtime_pacing = False

    try:
        import psutil
    except ImportError:
        psutil = None

    engine = SttEngine(config)
    client = LlmClient(config)
    manager = LlamaServerManager(config)
    tts = PiperTts(config)
    started_here = False
    components: list[str] = []

    try:
        engine.load_sync()
        components.append("STT")
    except Exception as exc:
        result.skipped = f"không nạp được Whisper: {exc}"
        return result

    await client.start()
    if await client.health():
        components.append("LLM")
    else:
        try:
            await manager.start()
            started_here = True
            components.append("LLM")
        except Exception:
            result.details["cảnh báo"] = "LLM không khả dụng — chỉ đo STT + TTS"

    tts_available = True
    try:
        tts.preflight()
        components.append("TTS")
    except TtsUnavailable as exc:
        tts_available = False
        result.details["cảnh báo TTS"] = str(exc)

    runs = args.runs or 8
    samples = get_samples(runs, seconds=2.0)
    tts_latencies: list[float] = []
    monitor = EventLoopMonitor()

    def progress(message: str) -> None:
        # Benchmark này có thể chạy rất lâu dưới tải. Im lặng hoàn toàn khiến
        # không phân biệt được "đang chạy" với "đã treo".
        print(f"    [B6] {message}", flush=True)

    async def stt_load() -> None:
        for index, (_name, pcm) in enumerate(samples, 1):
            started = time.perf_counter()
            await engine.transcribe(pcm, is_final=True)
            progress(f"STT {index}/{len(samples)} — {(time.perf_counter()-started)*1000:.0f}ms")

    async def llm_load() -> None:
        if "LLM" not in components:
            return
        for i in range(runs):
            started = time.perf_counter()
            await client.complete(client.build_prompt(f"Sample utterance number {i}.", "en"))
            progress(f"LLM {i+1}/{runs} — {(time.perf_counter()-started)*1000:.0f}ms")

    async def tts_load() -> None:
        if not tts_available:
            return
        for i in range(runs):
            sentence = TTS_SENTENCES[i % len(TTS_SENTENCES)]
            started = time.perf_counter()
            first = None
            async for _chunk in tts.synthesize(f"stress_{i}", sentence):
                if first is None:
                    first = (time.perf_counter() - started) * 1000.0
            if first is not None:
                tts_latencies.append(first)
            progress(f"TTS {i+1}/{runs} — {(time.perf_counter()-started)*1000:.0f}ms")

    # --- pha 1: chỉ TTS, lấy đường cơ sở ---
    progress("pha 1: chỉ TTS (đường cơ sở)")
    baseline_tts: list[float] = []
    if tts_available:
        monitor.start()
        await tts_load()
        await monitor.stop()
        baseline_tts = list(tts_latencies)
        tts_latencies.clear()
    idle_lag = Distribution.of(monitor.lags_ms)

    # --- pha 2: cả ba cùng chạy ---
    progress("pha 2: STT + LLM + TTS đồng thời")
    if psutil is not None:
        psutil.cpu_percent(interval=None)
    monitor = EventLoopMonitor()
    monitor.start()
    await asyncio.gather(stt_load(), llm_load(), tts_load())
    await monitor.stop()
    cpu_percent = psutil.cpu_percent(interval=None) if psutil is not None else None

    engine.unload()
    await client.close()
    await tts.close()
    if started_here:
        await manager.stop()

    loaded_lag = Distribution.of(monitor.lags_ms)
    baseline_dist = Distribution.of(baseline_tts)
    loaded_dist = Distribution.of(tts_latencies)

    # Ngưỡng 50ms: event loop giữ quá lâu tới mức này thì WebSocket bắt đầu
    # giật thấy được và Barge-in <200ms không còn đáng tin.
    result.checks = [
        Check("Event-loop lag P95 khi tải", loaded_lag.p95, 50.0),
        Check("Event-loop lag Max khi tải", loaded_lag.max, 200.0),
    ]
    if baseline_tts and tts_latencies:
        degradation = (loaded_dist.p95 / baseline_dist.p95 - 1.0) * 100 if baseline_dist.p95 else 0
        result.checks.append(
            Check("TTS P95 tăng thêm khi tải", degradation, 100.0, unit="%")
        )

    result.details = {
        "thành phần chạy": " + ".join(components) or "không có",
        "event-loop lag lúc rảnh": idle_lag.summary(),
        "event-loop lag khi tải": loaded_lag.summary(),
        "TTS lúc chỉ có TTS": baseline_dist.summary(),
        "TTS khi cả ba chạy": loaded_dist.summary(),
        "CPU": f"{cpu_percent:.0f}%" if cpu_percent is not None else "không đo (thiếu psutil)",
    }
    return result


def main(args) -> BenchmarkResult:
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(run_cli(main, base_parser(__doc__)))
