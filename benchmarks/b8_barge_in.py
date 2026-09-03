"""B8 — Barge-in response time. Target: < 200ms (§2.4.1).

Đo phía server: từ lúc phát tín hiệu speech mới tới lúc TTS thực sự ngừng đẩy
audio chunk. Đây là phần hệ thống kiểm soát được. Độ trễ phát lại ở client
(buffer Web Audio) được đo riêng trong web client.
"""

from __future__ import annotations

import asyncio
import time

from .common import BenchmarkResult, Check, Distribution, base_parser, run_cli

LONG_TEXT = (
    "Tôi nghĩ rằng chúng ta nên tạm gác lại cuộc thảo luận này và quay lại vào "
    "tuần sau khi mọi người đã có đủ thông tin cần thiết để đưa ra quyết định."
)


async def _run(args) -> BenchmarkResult:
    from app.ai.tts import PiperTts, TtsState, TtsUnavailable
    from app.core.config import load_config

    config = load_config()
    result = BenchmarkResult("B8", "Barge-in response time")

    engine = PiperTts(config)
    try:
        engine.preflight()
    except TtsUnavailable as exc:
        result.skipped = str(exc)
        return result

    runs = args.runs or 20
    response_times: list[float] = []
    time_to_last_chunk: list[float] = []
    failures = 0

    for i in range(runs):
        chunks = 0
        last_chunk_at = 0.0

        async def consume(index: int = i) -> None:
            nonlocal chunks, last_chunk_at
            async for _audio in engine.synthesize(f"barge_{index}", LONG_TEXT):
                chunks += 1
                last_chunk_at = time.perf_counter()

        task = asyncio.create_task(consume())

        # Đợi tới khi thực sự đang phát. Poll thay vì Event: ta đang đo một
        # đối tượng bên ngoài và cố tình KHÔNG thêm cơ chế đồng bộ vào đường
        # tới hạn của Barge-in chỉ để phục vụ phép đo.
        deadline = time.perf_counter() + 5.0
        while engine.state is not TtsState.PLAYING and time.perf_counter() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.002)

        if engine.state is not TtsState.PLAYING:
            failures += 1
            await engine.cancel()
            await task
            continue

        await asyncio.sleep(0.15)          # để nó đọc được một đoạn

        signalled_at = time.perf_counter()
        outcome = await engine.cancel(reason="barge_in")
        await task

        response_times.append(outcome.response_ms)
        # Thời gian tới chunk CUỐI CÙNG thực sự phát ra sau tín hiệu — đây mới
        # là thứ người nghe cảm nhận được, chặt chẽ hơn response_ms.
        if last_chunk_at > signalled_at:
            time_to_last_chunk.append((last_chunk_at - signalled_at) * 1000.0)
        else:
            time_to_last_chunk.append(0.0)

    await engine.close()

    response_dist = Distribution.of(response_times)
    audible_dist = Distribution.of(time_to_last_chunk)
    target = config.benchmark.barge_in_ms

    result.checks = [
        Check("Thời gian phản hồi P95", response_dist.p95, target),
        Check("Thời gian phản hồi Max", response_dist.max, target),
        Check("Audio còn phát ra sau tín hiệu P95", audible_dist.p95, target),
    ]
    result.details = {
        "phản hồi": response_dist.summary(),
        "audio còn lọt ra": audible_dist.summary(),
        "lần không vào được PLAYING": failures,
        "ghi chú": "Đo phía server; độ trễ buffer phát lại ở client đo riêng.",
    }
    return result


def main(args) -> BenchmarkResult:
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(run_cli(main, base_parser(__doc__)))
