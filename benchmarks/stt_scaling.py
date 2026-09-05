"""Chi phí Whisper có tỉ lệ với độ dài audio không?

    python -m benchmarks.stt_scaling

Quyết định một câu hỏi kiến trúc lớn: có thể VỪA NGHE VỪA PHIÊN ÂM DẦN để bản
dịch ra sớm hơn không, hay bắt buộc phải đợi người ta nói hết câu.

KẾT QUẢ (Whisper small, CPU, Mac):

    độ dài audio   thời gian nghe   ms mỗi giây audio
             0.5s           1655ms              3309
             1.0s           1561ms              1561
             2.0s           1579ms               790
             3.0s           1604ms               535
             4.0s           1771ms               443

CHI PHÍ GẦN NHƯ CỐ ĐỊNH: 8 lần audio chỉ tốn thêm 7%. faster-whisper đệm mọi
đoạn lên cửa sổ 30 giây rồi mới đưa vào encoder, nên nó làm gần như cùng một
lượng việc bất kể đoạn dài ngắn.

BA HỆ QUẢ:

1. PHIÊN ÂM DẦN LÀ VÔ VỌNG với model này. Mỗi lần phiên âm giữa chừng tốn
   ~1.6s, gần bằng lần cuối. Làm ba lần trong một câu là đốt 4.8 giây CPU mà
   chẳng ra sớm hơn, lại còn tranh CPU với lần cuối.

   Điều này GIẢI THÍCH một quan sát cũ chưa hiểu: bật `enable_partial` trên
   Mac làm độ trễ trượt dần 644 -> 1106 -> 1552ms qua từng câu và một câu mất
   hẳn phần đọc. Lúc đó tôi chỉ ghi nhận "tranh CPU"; giờ biết vì sao — mỗi
   lần phiên âm giữa chừng gần như đắt bằng cả câu.

2. KHÔNG THỂ đẩy 1.7s đó ra sớm hơn bằng cách chia nhỏ. Nó là chi phí cố định
   mỗi lần suy luận, không phải chi phí theo giây audio.

3. Đổi lại, GHÉP CÂU GẦN NHƯ MIỄN PHÍ về mặt tính toán — nghe một đoạn dài
   gấp đôi hầu như không tốn thêm. Cơ chế gộp câu bị ngắt giữa chừng vì thế
   không phải lo về chi phí.

Muốn giảm 1.7s thì chỉ còn hai đường: model nhỏ hơn (đánh đổi độ chính xác —
đã đo, `base` cho 9.1% WER so với 1.1% của `small` ở mức nhiễu phòng họp), hoặc
chạy trên GPU thay vì CPU.
"""

from __future__ import annotations

import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

SR = 16_000
MAU = Path("benchmarks/audio/stt_wer/03_en.wav")


def main() -> int:
    sys.path.insert(0, "backend")
    from app.core.config import load_config

    if not MAU.exists():
        print(f"Chưa có {MAU} — chạy `python benchmarks/stt_gen.py` trước.")
        return 1

    from faster_whisper import WhisperModel

    cfg = load_config()
    model = WhisperModel(
        cfg.paths.whisper_model, device="cpu", compute_type="int8",
        cpu_threads=cfg.stt.cpu_threads or 4,
    )

    with wave.open(str(MAU)) as h:
        x = np.frombuffer(h.readframes(h.getnframes()), dtype="<i2")
    audio = x.astype(np.float32) / 32768.0
    model.transcribe(audio[:SR], beam_size=1, vad_filter=False)      # làm nóng

    print(f"  {'độ dài audio':>14} {'thời gian nghe':>16} {'ms mỗi giây audio':>19}")
    print("  " + "-" * 54)
    for giay in (0.5, 1.0, 2.0, 3.0, 4.0):
        doan = np.tile(audio, 3)[: int(SR * giay)]
        times = []
        for _ in range(3):
            started = time.perf_counter()
            list(model.transcribe(doan, beam_size=1, vad_filter=False,
                                  condition_on_previous_text=False)[0])
            times.append((time.perf_counter() - started) * 1000)
        med = statistics.median(times)
        print(f"  {giay:>13.1f}s {med:>14.0f}ms {med / giay:>17.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
