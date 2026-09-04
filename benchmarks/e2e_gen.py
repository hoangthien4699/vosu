"""Dựng file nhiều câu VÀ ghi lại đúng vị trí mỗi câu kết thúc.

    python benchmarks/stt_gen.py        # cần chạy trước, để có câu nguồn
    python benchmarks/e2e_gen.py        # dựng file + mốc
    python benchmarks/e2e_latency.py    # đo (server phải đang chạy)


Không lấy mốc từ `stt_final` được: `duration_s` của segment gồm cả pre-roll
lẫn khoảng im lặng chờ VAD, nên nó nằm SAU lúc người ta ngừng nói.
"""
import json
import pathlib
import wave

import numpy as np

SR = 16000
WAV = pathlib.Path(__file__).parent / "audio" / "stt_wer"
src = sorted(WAV.glob("*_en.wav"))[:6]
rng = np.random.default_rng(3)


def sil(seconds):
    return rng.normal(0, 0.0005, int(SR * seconds)).astype(np.float32)


parts, ends = [sil(1.5)], []
n = len(parts[0])
for p in src:
    with wave.open(str(p)) as h:
        x = np.frombuffer(h.readframes(h.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    # cắt đuôi im lặng của chính file nguồn -> mốc "ngừng nói" mới đúng
    loud = np.where(np.abs(x) > 0.01)[0]
    x = x[: loud[-1] + 1] if loud.size else x
    parts.append(x)
    n += len(x)
    ends.append(n / SR)
    parts.append(sil(0.7))
    n += len(parts[-1])
parts.append(sil(1.5))

out = pathlib.Path("benchmarks/audio/multi_moc.wav")
data = np.concatenate(parts)
with wave.open(str(out), "w") as h:
    h.setnchannels(1)
    h.setsampwidth(2)
    h.setframerate(SR)
    h.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
pathlib.Path(str(out) + ".moc.json").write_text(json.dumps(ends))
print(f"{out.name}  {len(data)/SR:.1f}s, {len(ends)} câu")
print("  mốc ngừng nói (giây):", ", ".join(f"{e:.2f}" for e in ends))
