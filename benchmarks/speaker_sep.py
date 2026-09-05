"""Nhận ra ĐỔI NGƯỜI NÓI để cắt câu cho đúng — đo tính khả thi.

    .venv-vieneu/bin/python benchmarks/speaker_sep.py

VÌ SAO CẦN: độ dài khoảng lặng KHÔNG tách được "ngập ngừng giữa câu" với "hết
câu" — đo thật, khoảng ngập ngừng của người nói chậm (800ms) còn dài hơn
khoảng nghỉ giữa hai câu của người nói bình thường (700ms). Nhưng ĐỔI NGƯỜI
NÓI là mốc chắc chắn: A nói xong thì B mới đáp.

Dùng `speaker_encoder.onnx` của VieNeu-TTS — Apache-2.0, 28MB, đã nằm trong
phụ thuộc dự án ship sẵn. Bộ ONNX chuyên nhận dạng người nói của sherpa
(csukuangfj/speaker-embedding-models) bị loại vì KHÔNG khai báo giấy phép.

KẾT QUẢ (20 đoạn, 4 giọng máy):

    cả 4 giọng, một ngưỡng cố định        95.3%
    hai giọng (bài toán thật)          96.4% - 100%   (4/6 cặp đạt 100%)
    trích vector                        195ms/đoạn

ĐO SAI CÂU HỎI THÌ RA KẾT LUẬN NGƯỢC. Lần đầu tôi đo khoảng cách giữa
`min(cùng người)` và `max(khác người)` trên cả 4 giọng: ra -0.192, tức chồng
lấn, và suýt kết luận "không dùng được". Nhưng câu hỏi vận hành không phải
"tách sạch 4 người bằng một ngưỡng" mà là "câu này có cùng người với câu trước
không" — với HAI người, và đó là 96-100%.

Giả thuyết đoạn ngắn cho vector kém cũng SAI: đoạn ngắn (<2.6s) lại có trung
vị CAO hơn (0.879 so với 0.845). Nguyên nhân lẫn là giọng giống nhau — cả hai
cặp lẫn đều là nữ với nữ.

PHẠM VI: audio là giọng máy, không phải người thật trong phòng có nhiễu. 20
đoạn, 4 giọng là mẫu nhỏ. Thư viện còn văng lỗi lúc dọn dẹp (recursive_mutex)
— chạy qua tiến trình phụ thì không ảnh hưởng server, nhưng cần biết.
"""
import itertools
import pathlib
import sys
import wave

import numpy as np

sys.path.insert(0, "benchmarks")
from stt_cases import CASES

WAV = pathlib.Path("benchmarks/audio/stt_wer")
from vieneu import Vieneu

tts = Vieneu(backend="onnx")

def dai(f):
    with wave.open(str(f)) as h:
        return h.getnframes() / h.getframerate()

emb = {}
for i, (lang, voice, _t) in enumerate(CASES):
    f = WAV / f"{i:02d}_{lang}.wav"
    if not f.exists():
        continue
    v = np.asarray(tts.encode_reference(str(f), denoise=False)[0], np.float32).ravel()
    emb.setdefault(voice, []).append((v / (np.linalg.norm(v) + 1e-9), dai(f)))

def do(nguoi, nhan):
    cung, khac = [], []
    for v in nguoi:
        for (a, _), (b, _) in itertools.combinations(emb[v], 2):
            cung.append(float(a @ b))
    for va, vb in itertools.combinations(nguoi, 2):
        for a, _ in emb[va]:
            for b, _ in emb[vb]:
                khac.append(float(a @ b))
    if not cung or not khac:
        return
    best, bt = 0, 0
    for t in np.arange(0.30, 0.95, 0.005):
        acc = (sum(x >= t for x in cung) + sum(x < t for x in khac)) / (len(cung) + len(khac))
        if acc > best:
            best, bt = acc, t
    print(f"  {nhan:<34} độ chính xác tốt nhất {best*100:>5.1f}%  (ngưỡng {bt:.2f})")

do(list(emb), "cả 4 giọng")
for a, b in itertools.combinations(emb, 2):
    do([a, b], f"chỉ 2 giọng: {a} vs {b}")

print("\n  Độ dài đoạn có phải nguyên nhân không:")
ngan = [(a @ b) for v in emb for (a, da), (b, db) in itertools.combinations(emb[v], 2) if min(da, db) < 2.6]
dai_ = [(a @ b) for v in emb for (a, da), (b, db) in itertools.combinations(emb[v], 2) if min(da, db) >= 2.6]
if ngan and dai_:
    print(f"    cùng người, đoạn NGẮN (<2.6s): trung vị {np.median(ngan):.3f}  thấp nhất {min(ngan):.3f}")
    print(f"    cùng người, đoạn DÀI (>=2.6s): trung vị {np.median(dai_):.3f}  thấp nhất {min(dai_):.3f}")
