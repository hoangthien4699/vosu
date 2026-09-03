"""B7 — Audio feedback loop: TTS phát qua loa trong lúc mic đang thu.

§2.4 (review v4.1): khi TTS thực sự phát qua earbud, audio output CÓ THỂ trở
thành input của chính hệ thống. Đây là điểm biến hệ thống từ pipeline một chiều
(Mic -> AI -> UI) thành hệ thống audio hai chiều đóng vòng (Mic -> AI -> Earbud).

Benchmark này đo: bao nhiêu phần trăm số lần TTS phát ra bị Whisper nhận nhầm
thành speech input mới. §6 mục 8 nói rõ đây là phép đo LẤY BASELINE — chưa cần
target cứng ở lần đo đầu, kết quả dùng làm cơ sở thiết kế Barge-in.
"""

from __future__ import annotations

import asyncio

from .common import BenchmarkResult, Check, base_parser, run_cli
from .hardware import (
    AudioIoUnavailable,
    count_speech_segments,
    pcm16_bytes_to_float32,
    play_async,
    prompt,
    record,
    stop_playback,
)

SENTENCES = [
    "Tôi nghĩ chúng ta nên tạm gác lại cuộc thảo luận này.",
    "Bạn có thể nói rõ hơn về vấn đề đó không?",
    "Được rồi, tôi đã hiểu ý của anh.",
    "Chúng ta sẽ bàn lại chuyện này vào tuần sau.",
]


async def _run(args) -> BenchmarkResult:
    from app.ai.stt import SttEngine
    from app.ai.tts import PiperTts, TtsUnavailable
    from app.core.config import load_config

    config = load_config()
    result = BenchmarkResult("B7", "Audio feedback loop (TTS phát + mic thu đồng thời)")

    tts = PiperTts(config)
    try:
        tts.preflight()
    except TtsUnavailable as exc:
        result.skipped = str(exc)
        return result

    engine = SttEngine(config)
    try:
        engine.load_sync()
    except Exception as exc:
        result.skipped = f"không nạp được Whisper: {exc}"
        return result

    runs = args.runs or len(SENTENCES)

    try:
        prompt(
            "Đặt tai nghe/loa ở vị trí SỬ DỤNG THẬT (đeo lên tai nếu là earbud).\n"
            "    Giữ im lặng hoàn toàn trong suốt phép đo — mọi speech phát hiện\n"
            "    được đều là do audio TTS lọt ngược vào mic."
        )

        false_triggers = 0
        transcribed = 0
        transcripts: list[str] = []

        for i in range(runs):
            sentence = SENTENCES[i % len(SENTENCES)]

            audio_chunks = [
                chunk async for chunk in tts.synthesize(f"fb_{i}", sentence)
            ]
            if not audio_chunks:
                continue
            tts_pcm = pcm16_bytes_to_float32(b"".join(audio_chunks))

            duration = len(tts_pcm) / config.tts.sample_rate
            play_async(tts_pcm, samplerate=config.tts.sample_rate)
            captured = record(duration + 0.5)
            stop_playback()

            segments = count_speech_segments(captured, config)
            if segments:
                false_triggers += 1

            transcript = engine.transcribe_sync(captured, is_final=True)
            if not transcript.is_empty:
                transcribed += 1
                transcripts.append(transcript.text[:70])

    except AudioIoUnavailable as exc:
        result.skipped = str(exc)
        return result
    finally:
        await tts.close()
        engine.unload()

    total = runs or 1
    vad_rate = false_triggers / total * 100
    stt_rate = transcribed / total * 100

    result.checks = [
        Check("VAD false-trigger rate", vad_rate, None, unit="%", mode="record"),
        Check("Whisper transcribe được audio TTS", stt_rate, None, unit="%", mode="record"),
    ]
    result.details = {
        "số lần thử": total,
        "VAD kích hoạt nhầm": f"{false_triggers}/{total}",
        "Whisper ra text": f"{transcribed}/{total}",
        "transcript nhầm mẫu": transcripts[:3] or "(không có)",
        "diễn giải": (
            "Tỷ lệ cao => audio TTS lọt vào mic nhiều => Barge-in sẽ liên tục "
            "tự kích hoạt bởi chính giọng AI. Cần AEC hoặc gating theo trạng "
            "thái TTS trước khi lên mobile."
        ),
        "ghi chú": "§6 mục 8 — phép đo lấy baseline, chưa có target cứng ở lần đầu.",
    }
    return result


def main(args) -> BenchmarkResult:
    return asyncio.run(_run(args))


if __name__ == "__main__":
    parser = base_parser(__doc__)
    raise SystemExit(run_cli(main, parser))
