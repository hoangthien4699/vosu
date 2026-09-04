"""Tiến trình phụ đọc bằng VieNeu-TTS. Chạy trong VENV RIÊNG.

VÌ SAO TÁCH RA: gói `vieneu` phụ thuộc CỨNG vào `gradio` và `librosa` (không
phải extra tuỳ chọn). Cài chung vào venv của server sẽ nâng cấp FastAPI
0.115.6 -> 0.141.1 và thêm 53 gói — làm rung chính khung đang chạy. Dự án vốn
đã chạy `llama-server` và `piper` như tiến trình ngoài, thêm cái này là cùng
một khuôn.

Khác Piper ở một điểm quan trọng: tiến trình này SỐNG LÂU, đọc nhiều câu. Piper
phải spawn lại mỗi câu nên tốn ~520ms nạp model mỗi lần; ở đây nạp một lần rồi
thôi, tiếng đầu ra sau ~91ms.

Giao thức, cố tình giữ đơn giản để không phải thêm thư viện nào:

    vào   : mỗi dòng một yêu cầu JSON {"text": "...", "voice": "..."}
    ra    : với mỗi mẩu audio   `CHUNK <số byte>\\n` rồi đúng ngần ấy byte thô
            hết một câu          `END\\n`
            lỗi                  `ERR <thông điệp>\\n`

Audio là PCM 16-bit mono 48kHz, khớp `sample_rate` của model.
"""

import json
import sys

import numpy as np

SAMPLE_RATE = 48_000


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

    # backend="onnx": đường streaming, torch-free. Bản PyTorch tối ưu cho sinh
    # hàng loạt chứ không cho streaming.
    tts = Vieneu(backend="onnx")
    write("READY\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            text = (req.get("text") or "").strip()
            voice = req.get("voice") or None
            if not text:
                write("END\n")
                continue
            for block in tts.infer_stream(text, voice=voice):
                pcm = np.asarray(block, dtype=np.float32).squeeze()
                raw = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                write(f"CHUNK {len(raw)}\n", raw)
            write("END\n")
        except Exception as exc:                      # noqa: BLE001
            write(f"ERR {type(exc).__name__}: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
