"""Sinh audio cho benchmarks/stt_wer.py bằng `say` của macOS.

CẢNH BÁO: giọng máy sạch hơn giọng người thật nhiều, nên WER tuyệt đối
ở đây LẠC QUAN. Chỉ dùng để so cấu hình A với cấu hình B.
"""
import pathlib
import subprocess
import sys
import wave

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from stt_cases import CASES

out = pathlib.Path(__file__).parent / "audio" / "stt_wer"
out.mkdir(exist_ok=True)
for i, (lang, voice, text) in enumerate(CASES):
    wav = out / f"{i:02d}_{lang}.wav"
    if wav.exists():
        continue
    # say ghi thẳng WAV 16kHz mono, khỏi cần ffmpeg
    subprocess.run(["say", "-v", voice, "-o", str(wav),
                    "--data-format=LEI16@16000", "--channels=1", text], check=True)
tot = 0.0
for w in sorted(out.glob("*.wav")):
    with wave.open(str(w)) as h:
        tot += h.getnframes()/h.getframerate()
print(f"{len(list(out.glob('*.wav')))} file, tổng {tot:.1f}s")
