"""B10 — Web mic vs Bluetooth mic, 4 kịch bản (§9.2).

§9.2 (review v4.1): Bluetooth không chỉ là vấn đề "chất lượng audio" mà còn là
vấn đề DUPLEX BEHAVIOR — cần test mic input + earbud output ĐỒNG THỜI, cụ thể
là hành vi A2DP output vs HFP/HSP/SCO khi microphone đang active. Đây có thể là
vấn đề ở tầng OS/audio profile, nhưng ảnh hưởng trực tiếp tới tính khả thi của
Barge-in.

Bốn kịch bản bắt buộc:
    Web mic (baseline)
    Bluetooth mic
    Bluetooth mic + playback (không TTS)
    Bluetooth mic + TTS playback   <- thực tế nhất
"""

from __future__ import annotations

from .common import BenchmarkResult, Check, base_parser, run_cli
from .hardware import (
    AudioIoUnavailable,
    ask,
    count_speech_segments,
    list_devices,
    print_devices,
    prompt,
    record,
)

SCENARIOS = [
    ("web_mic", "Mic máy tính (baseline)", False),
    ("bt_mic", "Mic Bluetooth earbuds", False),
    ("bt_mic_playback", "Mic Bluetooth + đang phát nhạc/audio (KHÔNG phải TTS)", True),
    ("bt_mic_tts", "Mic Bluetooth + đang phát TTS (kịch bản thực tế nhất)", True),
]

PHRASE = "I think we should table this discussion for now."


def main(args) -> BenchmarkResult:
    from app.ai.stt import SttEngine
    from app.core.config import load_config

    config = load_config()
    result = BenchmarkResult("B10", "Web mic vs Bluetooth mic (4 kịch bản)")

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
        bluetooth = [d for d in list_devices() if d.is_input and d.looks_bluetooth()]
        if bluetooth:
            print("\nThiết bị vào có vẻ là Bluetooth:")
            for device in bluetooth:
                print(f"  [{device.index}] {device.name}")

        prompt(
            f"Mỗi kịch bản: đọc CÙNG MỘT câu, cùng khoảng cách, cùng âm lượng.\n"
            f'    Câu chuẩn: "{PHRASE}"\n'
            "    Giữ điều kiện giống nhau, nếu không thì không so sánh được."
        )

        for key, title, needs_playback in SCENARIOS:
            prompt(f"KỊCH BẢN: {title}")
            device_index = ask(
                "Index thiết bị thu cho kịch bản này (Enter = mặc định):", ""
            )
            device = int(device_index) if device_index.isdigit() else None

            if needs_playback:
                prompt("Bật audio phát ra tai nghe TRƯỚC, rồi mới tiếp tục.")

            print(f"    Đang thu {seconds:.0f}s — hãy đọc câu chuẩn ngay bây giờ...")
            captured = record(seconds, device=device)

            segments = count_speech_segments(captured, config)
            transcript = engine.transcribe_sync(captured, is_final=True)
            profile = ask("OS báo audio profile nào? (a2dp/hfp/sco/không rõ):", "không rõ")

            wer = _word_error_rate(PHRASE, transcript.text)
            records[key] = {
                "kịch bản": title,
                "thiết bị": device if device is not None else "mặc định",
                "transcript": transcript.text or "(rỗng)",
                "WER ước tính": f"{wer * 100:.0f}%" if wer is not None else "n/a",
                "latency STT": f"{transcript.latency_ms:.0f}ms",
                "đoạn speech VAD": segments,
                "audio profile": profile,
                "độ lớn RMS": round(float((captured**2).mean() ** 0.5), 5),
            }
            print(f"    -> {records[key]['transcript'][:70]!r} (WER {records[key]['WER ước tính']})")

    except AudioIoUnavailable as exc:
        result.skipped = str(exc)
        return result
    except KeyboardInterrupt:
        result.error = "người vận hành đã hủy"
        return result
    finally:
        engine.unload()

    result.checks = [
        Check("Số kịch bản đã đo", float(len(records)), None, unit="", mode="record")
    ]
    result.details = {**records, "diễn giải": (
        "Nếu WER của bt_mic_tts cao hơn hẳn web_mic thì audio routing riêng cho "
        "Bluetooth là bắt buộc TRƯỚC khi build mobile app (§9.2). Ghi lại audio "
        "profile: A2DP thường có chất lượng thu kém/không có mic, HFP/SCO có mic "
        "nhưng bandwidth thấp."
    )}
    return result


def _word_error_rate(reference: str, hypothesis: str) -> float | None:
    """WER bằng khoảng cách Levenshtein trên từ."""
    ref = reference.lower().replace(".", "").split()
    hyp = hypothesis.lower().replace(".", "").split()
    if not ref:
        return None
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1,
                    previous[j - 1] + (ref_word != hyp_word))
            )
        previous = current
    return previous[-1] / len(ref)


if __name__ == "__main__":
    parser = base_parser(__doc__)
    parser.add_argument("--seconds", type=float, default=6.0)
    raise SystemExit(run_cli(main, parser))
