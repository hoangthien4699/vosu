"""B9 — Speaker-source validation. 5 kịch bản (§9.1).

§9.1 (review v4.1) nâng đây thành P0 hardware validation:

    Microphone của earbud thường nằm RẤT GẦN MIỆNG NGƯỜI DÙNG, trong khi yêu
    cầu sản phẩm lại là nghe NGƯỜI ĐỐI DIỆN — đây là mismatch vật lý tiềm tàng,
    không chỉ là một giả định cần "lưu ý".

Kết quả benchmark này quyết định có phải thiết kế lại audio routing hay không.
Cần người vận hành thực hiện kịch bản — không tự động hóa được.
"""

from __future__ import annotations

from .common import BenchmarkResult, Check, base_parser, run_cli
from .hardware import (
    AudioIoUnavailable,
    ask,
    count_speech_segments,
    print_devices,
    prompt,
    record,
)

SCENARIOS = [
    ("user_only", "CHỈ NGƯỜI DÙNG nói (người đeo tai nghe)",
     "Ai được detect là speaker? Có bị nhận nhầm thành 'cần dịch' không?"),
    ("other_only", "CHỈ NGƯỜI ĐỐI DIỆN nói (cách ~1m)",
     "WER? Có nhận đúng là 'cần dịch' không? Đây là kịch bản CHÍNH của sản phẩm."),
    ("both", "CẢ HAI nói xen kẽ",
     "False activation rate — hệ thống có phân biệt được nguồn không?"),
    ("tts_playback", "TTS đang phát, KHÔNG ai nói",
     "Miss rate / false trigger — liên quan trực tiếp tới Barge-in (§2.4.1)."),
    ("background_noise", "Không ai nói, chỉ có tiếng ồn nền",
     "WER dưới điều kiện nhiễu; VAD có kích hoạt nhầm không?"),
]


def main(args) -> BenchmarkResult:
    from app.ai.stt import SttEngine
    from app.core.config import load_config

    config = load_config()
    result = BenchmarkResult("B9", "Speaker-source validation (5 kịch bản)")

    engine = SttEngine(config)
    try:
        engine.load_sync()
    except Exception as exc:
        result.skipped = f"không nạp được Whisper: {exc}"
        return result

    seconds = float(args.seconds)
    records: dict[str, dict] = {}

    try:
        print_devices()
        prompt(
            "Đeo tai nghe đúng như khi dùng thật. Mỗi kịch bản thu "
            f"{seconds:.0f} giây.\n"
            "    Kết quả quyết định có phải thiết kế lại audio routing hay không."
        )

        for key, title, question in SCENARIOS:
            prompt(f"KỊCH BẢN: {title}\n    ({question})")
            print(f"    Đang thu {seconds:.0f}s...")
            captured = record(seconds, device=args.device)

            segments = count_speech_segments(captured, config)
            transcript = engine.transcribe_sync(captured, is_final=True)
            expected = ask("Đoạn vừa nói là gì? (Enter nếu không có ai nói):")

            records[key] = {
                "kịch bản": title,
                "số đoạn speech VAD phát hiện": segments,
                "transcript": transcript.text or "(rỗng)",
                "ngôn ngữ nhận diện": transcript.language,
                "câu thật": expected or "(không có ai nói)",
                "độ lớn RMS": round(float((captured**2).mean() ** 0.5), 5),
            }
            print(f"    VAD: {segments} đoạn | Whisper: {records[key]['transcript'][:70]!r}")

    except AudioIoUnavailable as exc:
        result.skipped = str(exc)
        return result
    except KeyboardInterrupt:
        result.error = "người vận hành đã hủy"
        return result
    finally:
        engine.unload()

    silent_scenarios = ("tts_playback", "background_noise")
    false_activations = sum(
        1 for k in silent_scenarios
        if k in records and records[k]["số đoạn speech VAD phát hiện"] > 0
    )
    counted = sum(1 for k in silent_scenarios if k in records)

    result.checks = [
        Check(
            "False activation khi không ai nói",
            false_activations / counted * 100 if counted else None,
            None, unit="%", mode="record",
        )
    ]
    result.details = {**records, "diễn giải": (
        "So sánh 'user_only' và 'other_only': nếu giọng người dùng ra transcript "
        "rõ hơn hẳn giọng người đối diện thì MVP assumption ở §7.2 là SAI, và "
        "audio routing phải nâng thành P0 thiết kế lại (§9.1)."
    )}
    return result


if __name__ == "__main__":
    parser = base_parser(__doc__)
    parser.add_argument("--device", type=int, default=None, help="index thiết bị thu")
    parser.add_argument("--seconds", type=float, default=6.0, help="thời lượng mỗi kịch bản")
    raise SystemExit(run_cli(main, parser))
