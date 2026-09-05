"""So prompt "dịch chính xác" với prompt "dịch hội thoại nói".

    python -m benchmarks.prompt_conversational

Prompt hiện tại đặt bài toán là DỊCH CHÍNH XÁC và cấm đổi cấu trúc câu. Bản
đối chứng đặt lại thành DỊCH HỘI THOẠI NÓI, cho phép dựng lại câu miễn không
đổi nghĩa — vì STT trả về lời NÓI, không phải câu viết chuẩn.

Kết quả trên 26 ca:

    prompt                       yếu tố giữ   thiếu ý   tiểu từ    P50
    hiện tại (dịch chính xác)    60/62 (97%)    2/26      1/5    2193ms
    hội thoại nói               59/62 (95%)    2/26      2/5    1683ms

Hoà về độ trung thực (chênh 1 yếu tố, trong mức nhiễu), tiểu từ gấp đôi, nhanh
hơn 1.3 lần vì prompt ngắn hơn.

CẢNH BÁO ĐÃ TRẢ GIÁ: lần đo đầu tôi ghi cứng "into Vietnamese" trong prompt,
nên chiều Việt->Anh KHÔNG DỊCH GÌ — bản dịch trả về vẫn nguyên tiếng Việt. Điểm
tụt xuống 53% và tôi suýt kết luận là cách này làm hỏng độ chính xác. Prompt
PHẢI nhận ngôn ngữ đích theo chiều dịch, không được ghi cứng.
"""
import asyncio
import statistics
import sys
import time

sys.path.insert(0, "backend")
sys.path.insert(0, ".")
from benchmarks.fidelity_cases import ALL, has_particle, inverted, missing

TEN = {"vi": "Vietnamese", "en": "English"}


def hoi_thoai(direction, user_language="vi", counterpart_language="en", **_):
    from app.ai.direction import Direction

    if direction is Direction.TO_COUNTERPART:
        src, dst = user_language, counterpart_language or "en"
        ai_noi = "The user is speaking. Translate what they said"
    else:
        src, dst = counterpart_language or "en", user_language
        ai_noi = "Someone is speaking to the user. Translate what they said"
    dst_ten = TEN.get(dst, dst)
    tu_tinh_thai = (
        '   Where a spoken sentence would naturally end with "nhé", "ạ", "nhỉ",\n'
        '   "đấy" or "thôi", use it.\n'
        if dst == "vi" else ""
    )
    return f"""You are a real-time conversational translator.

Source language: {TEN.get(src, src)}
Target language: {dst_ten}

{ai_noi} into natural SPOKEN {dst_ten} — the way people actually talk to each
other, not the way documents are written.

Priorities, in order:
1. Preserve the exact meaning and intent.
2. Preserve the speaker's tone, emotion and level of politeness.
3. Make it sound natural to a native speaker.
{tu_tinh_thai}4. Do not translate word by word. Rewrite unnatural structures when needed.
5. Keep every number, date, name, negation and hedge exactly as said.
6. Do not add information that is not there. Do not drop meaningful information.
7. Keep who does what to whom exactly as said.
8. Keep it concise — it will be read aloud by a speech synthesiser.

Output ONE compact JSON object and nothing else:
{{"translation":"..."}}"""


async def do(nhan, he_thong, show=False):
    from app.ai import llm as M
    from app.ai.copilot import SemanticEventParser
    from app.ai.direction import Direction
    from app.core.config import load_config
    from app.core.vram_manager import LlamaServerManager

    config = load_config()
    goc = M.system_prompt
    if he_thong is not None:
        M.system_prompt = he_thong
    manager = LlamaServerManager(config)
    await manager.start()
    client = M.LlmClient(config)
    await client.start()
    try:
        rows, lat = [], []
        for c in ALL:
            out = c.lang == "vi"
            d = Direction.TO_COUNTERPART if out else Direction.TO_USER
            t0 = time.perf_counter()
            raw, _ = await client.complete(
                client.build_prompt(c.text, c.lang, direction=d,
                                    counterpart_language="en"))
            par = SemanticEventParser()
            par.feed(raw)
            par.finish()
            lat.append((time.perf_counter() - t0) * 1000)
            rows.append((c, par.result.translation))
        tot = sum(len(c.must_keep) for c, _ in rows)
        mat = sum(len(missing(c, t)) for c, t in rows)
        casual = [(c, t) for c, t in rows if c.casual and c.lang == "en"]
        print(f"  {nhan:<30} {tot-mat}/{tot} ({(tot-mat)/tot*100:>3.0f}%)  "
              f"thiếu ý {sum(1 for c, t in rows if missing(c, t)):>2}/26  "
              f"đảo {sum(1 for c, t in rows if inverted(c, t)):>2}/26  "
              f"tiểu từ {sum(1 for _, t in casual if has_particle(t))}/{len(casual)}  "
              f"P50 {statistics.median(lat):>4.0f}ms")
        if show:
            print("\n  Câu đời thường:")
            for _, t in casual:
                print(f"    {t[:74]}")
            print("\n  Ca còn hỏng:")
            for c, t in rows:
                if missing(c, t):
                    print(f"    [{c.lang}] {c.text[:44]}")
                    print(f"       -> {t[:60]}")
    finally:
        M.system_prompt = goc
        await client.close()
        await manager.stop()


async def main():
    print("  " + "-" * 92)
    await do("hiện tại (dịch chính xác)", None)
    await do("hội thoại nói (có target)", hoi_thoai, show=True)


asyncio.run(main())
