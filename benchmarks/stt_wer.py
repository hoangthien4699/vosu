"""Đo sai số STT (WER) theo từng cấu hình Whisper.

    python benchmarks/stt_gen.py     # sinh audio (macOS `say`)
    python benchmarks/stt_wer.py     # đo

Dùng khi đổi model/tham số STT. Đây là thứ đã cho thấy `base` nghe nhầm
"thanh toán" thành "thang tòa án" còn `small` thì không.

CẢNH BÁO: audio sinh bằng `say` sạch hơn giọng người thật nhiều, nên WER
tuyệt đối ở đây LẠC QUAN. Chỉ dùng để so cấu hình A với cấu hình B.
"""
from __future__ import annotations

import itertools
import pathlib
import re
import sys
import time
import unicodedata
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from stt_cases import CASES

WAV = pathlib.Path(__file__).parent / "audio" / "stt_wer"

def norm(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\sàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", " ", s)
    return s.split()

def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1]+(r[i-1] != h[j-1]))
    return int(d[len(r), len(h)]), len(r)

def load(p):
    with wave.open(str(p)) as h:
        return np.frombuffer(h.readframes(h.getnframes()), dtype="<i2").astype(np.float32)/32768.0

_cache = {}
def model(name):
    from faster_whisper import WhisperModel

    if name not in _cache:
        _cache[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _cache[name]

def run(name, beam, lang_mode):
    m = model(name)
    errs = refs = lid_ok = lid_n = 0
    t0 = time.perf_counter()
    bad = []
    for i, (lang, _voice, text) in enumerate(CASES):
        pcm = load(WAV / f"{i:02d}_{lang}.wav")
        kw = {} if lang_mode == "auto" else {"language": lang}
        segs, info = m.transcribe(pcm, beam_size=beam, vad_filter=False,
                                  condition_on_previous_text=False, **kw)
        hyp = " ".join(s.text for s in segs).strip()
        e, n = wer(text, hyp)
        errs += e
        refs += n
        if lang_mode == "auto":
            lid_n += 1
            lid_ok += info.language == lang
            if info.language != lang:
                bad.append(f"    LID sai {i:02d}: đoán {info.language!r} (đúng {lang!r})")
        if e:
            bad.append(f"    {i:02d} {lang} [{e}/{n} lỗi] {hyp[:70]!r}")
    dt = (time.perf_counter() - t0) * 1000 / len(CASES)
    return errs/refs*100, (lid_ok/lid_n*100 if lid_n else None), dt, bad

if __name__ == "__main__":
    combos = list(itertools.product(["base", "small"], [1, 5], ["auto", "pin"]))
    print(f"{'model':>6} {'beam':>5} {'lang':>5} {'WER%':>7} {'LID%':>6} {'ms/câu':>8}")
    print("-"*46)
    detail = {}
    for name, beam, lm in combos:
        w, lid, ms, bad = run(name, beam, lm)
        detail[(name, beam, lm)] = bad
        lid_text = f"{lid:.0f}" if lid is not None else "—"
        print(f"{name:>6} {beam:>5} {lm:>5} {w:>7.1f} {lid_text:>6} {ms:>8.0f}")
    print()
    for key in (("base", 1, "auto"), ("small", 1, "auto")):
        print(f"== {key} ==")
        for line in detail[key][:12]:
            print(line)
