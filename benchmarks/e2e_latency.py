"""Từ lúc người ta NGỪNG NÓI tới lúc có TIẾNG DỊCH để nghe.

    python benchmarks/e2e_gen.py                 # dựng audio + mốc
    python benchmarks/e2e_latency.py benchmarks/audio/multi_moc.wav

Đây là con số người dùng thật sự cảm nhận. Hai cái bẫy đã mắc phải khi đo:

  1. `tts_started` KHÔNG phải lúc nghe được — server mới bắt đầu sinh tiếng,
     còn ~0.8s nữa mới có byte âm thanh đầu tiên. Phải bắt frame nhị phân.

  2. `start_s + duration_s` của `stt_final` KHÔNG phải lúc ngừng nói — segment
     gom cả pre-roll lẫn khoảng im lặng chờ VAD, đo được là nó nằm SAU lúc
     ngừng nói +0.72s. Phải dùng mốc dựng sẵn trong file .moc.json.


Mốc "ngừng nói" lấy từ file .moc.json dựng cùng audio, KHÔNG lấy từ
`stt_final`: `duration_s` của segment gồm cả pre-roll lẫn khoảng im lặng chờ
VAD nên nó nằm sau lúc ngừng nói ~0.85s.
"""
import asyncio
import json
import pathlib
import sys
import time
import wave

import websockets


async def main(path):
    moc = json.loads(pathlib.Path(path + ".moc.json").read_text())
    with wave.open(path) as h:
        pcm = h.readframes(h.getnframes())
        sr = h.getframerate()
    utt, order, cur = {}, [], None
    async with websockets.connect("ws://127.0.0.1:8000/ws/copilot", max_size=None) as ws:
        t0 = time.perf_counter()

        def now():
            return (time.perf_counter() - t0) * 1000

        async def send():
            step = sr // 10 * 2
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i+step])
                await asyncio.sleep(0.1)
        task = asyncio.create_task(send())
        try:
            while True:
                m = await asyncio.wait_for(ws.recv(), timeout=15)
                if isinstance(m, bytes):
                    if cur and "co_tieng" not in utt[cur]:
                        utt[cur]["co_tieng"] = now()
                    continue
                e = json.loads(m)
                u = e.get("utterance_id")
                if not u:
                    continue
                r = utt.setdefault(u, {})
                if e["type"] == "stt_final":
                    r["nghe_xong"] = now()
                    order.append(u)
                elif e["type"] == "translation_delta" and "chu_dau" not in r:
                    r["chu_dau"] = now()
                elif e["type"] == "tts_started":
                    cur = u
                    r["sinh_tieng"] = now()
        except asyncio.TimeoutError:
            pass
        task.cancel()

    print(f"\n{'câu':>7} {'ngừng nói':>10} | {'nghe xong':>10} {'chữ đầu':>9} "
          f"{'sinh tiếng':>11} {'CÓ TIẾNG':>10}")
    print("-" * 64)
    vals = []
    for i, u in enumerate(order):
        if i >= len(moc) or "co_tieng" not in utt[u]:
            continue
        r, h = utt[u], moc[i] * 1000

        def col(key, row=r, base=h):
            return f"+{row[key] - base:.0f}ms" if key in row else "—"

        vals.append(r["co_tieng"] - h)
        print(f"{u:>7} {h/1000:>9.2f}s | {col('nghe_xong'):>10} {col('chu_dau'):>9} "
              f"{col('sinh_tieng'):>11} {col('co_tieng'):>10}")
    if vals:
        v = sorted(vals)
        print("\n  TỪ LÚC NGỪNG NÓI TỚI KHI CÓ TIẾNG DỊCH")
        print(f"  trung bình {sum(vals)/len(vals)/1000:.2f}s · "
              f"nhanh nhất {v[0]/1000:.2f}s · chậm nhất {v[-1]/1000:.2f}s")

asyncio.run(main(sys.argv[1]))
