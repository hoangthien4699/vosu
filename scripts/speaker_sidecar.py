"""Tiến trình phụ trích vector nhận dạng giọng. Chạy trong VENV RIÊNG.

VÌ SAO TÁCH KHỎI TIẾN TRÌNH TTS: cả hai đều dùng gói `vieneu`, nhưng chúng
chạy vào những lúc CHỒNG NHAU — TTS đang đọc bản dịch câu N thì câu N+1 vừa
dứt lời và cần trích vector. Dùng chung một tiến trình là chúng xếp hàng chờ
nhau, cộng thẳng vào độ trễ.

VÌ SAO KHÔNG IMPORT THẲNG VÀO SERVER: gói `vieneu` phụ thuộc cứng vào gradio
và librosa — xem scripts/vieneu_sidecar.py.

Giao thức (nhị phân, giữ đơn giản):

    vào   : `EMBED <số byte>\\n` rồi đúng ngần ấy byte PCM 16-bit mono 16kHz
    ra    : `VEC <số chiều>\\n` rồi ngần ấy số float32
            `ERR <thông điệp>\\n` nếu hỏng

`encode_reference()` CHỈ nhận đường dẫn file, không nhận mảng — nên phải ghi
file tạm cho mỗi lần. Đã đo, phần I/O này không đáng kể so với chính việc
trích vector.
"""

import os
import sys
import tempfile
import wave

import numpy as np

SAMPLE_RATE = 16_000


def write(header: str, payload: bytes = b"") -> None:
    sys.stdout.buffer.write(header.encode("utf-8"))
    if payload:
        sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def main() -> int:
    try:
        from vieneu import Vieneu
    except ImportError as exc:
        write(f"ERR không import được vieneu: {exc}\n")
        return 1

    tts = Vieneu(backend="onnx")

    # Làm nóng: lần trích ĐẦU TIÊN mất ~4 giây vì onnxruntime dựng phiên lúc
    # đó, các lần sau ~440ms. Trả giá ngay bây giờ thay vì để câu đầu tiên của
    # người dùng gánh.
    try:
        _embed(tts, np.zeros(SAMPLE_RATE, dtype=np.float32))
    except Exception:                                       # noqa: BLE001
        pass
    write("READY\n")

    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return 0
        parts = line.split()
        if len(parts) != 2 or parts[0] != b"EMBED":
            continue
        raw = stdin.read(int(parts[1]))
        try:
            pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            vec = _embed(tts, pcm).astype("<f4")
            write(f"VEC {vec.size}\n", vec.tobytes())
        except Exception as exc:                            # noqa: BLE001
            write(f"ERR {type(exc).__name__}: {exc}\n")


def _embed(tts, pcm: np.ndarray) -> np.ndarray:
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "w") as h:
            h.setnchannels(1)
            h.setsampwidth(2)
            h.setframerate(SAMPLE_RATE)
            h.writeframes((np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2").tobytes())
        out = tts.encode_reference(path, denoise=False)
        return np.asarray(out[0], dtype=np.float32).ravel()
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    raise SystemExit(main())
