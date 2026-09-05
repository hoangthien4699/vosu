"""Prompt ràng buộc nhiều hay ít thì dịch tốt hơn?

Prompt hiện tại 297 từ, 6 quy tắc. Quy tắc 6 ĐÃ nêu thẳng tên các tiểu từ, mà
model vẫn chỉ ra 1/5 — nên lệnh có đó nhưng bị năm quy tắc chính xác phía trên
đè. Thử nới ràng buộc xem có thoát ra không, và mất gì.
"""
import asyncio
import statistics
import sys
import time

sys.path.insert(0, "backend")
sys.path.insert(0, ".")
from benchmarks.fidelity_cases import ALL, has_particle, inverted, missing

NGAN = """Bạn là phiên dịch trực tiếp. Dịch câu người ta vừa nói sang tiếng Việt.

Dịch như người Việt NÓI với nhau, không phải như văn bản.

Chỉ trả về một JSON: {"translation":"..."}"""

VUA = """Bạn là phiên dịch trực tiếp. Dịch câu người ta vừa nói sang tiếng Việt.

Dịch như người Việt NÓI với nhau, không phải như văn bản.

Giữ đúng: con số, ngày giờ, tên riêng, phủ định, và ai làm gì với ai.

Chỉ trả về một JSON: {"translation":"..."}"""


async def do(nhan, he_thong):
    from app.ai import llm as M
    from app.ai.copilot import SemanticEventParser
    from app.ai.direction import Direction
    from app.core.config import load_config
    from app.core.vram_manager import LlamaServerManager

    config = load_config()
    goc = M.system_prompt
    if he_thong is not None:
        M.system_prompt = lambda *a, **k: he_thong
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
        print(f"  {nhan:<22} {tot-mat}/{tot} ({(tot-mat)/tot*100:>3.0f}%)  "
              f"thiếu ý {sum(1 for c,t in rows if missing(c,t)):>2}/26  "
              f"đảo {sum(1 for c,t in rows if inverted(c,t)):>2}/26  "
              f"tiểu từ {sum(1 for _,t in casual if has_particle(t))}/{len(casual)}  "
              f"P50 {statistics.median(lat):>4.0f}ms")
        return [t for _, t in rows if _.casual and _.lang == "en"]
    finally:
        M.system_prompt = goc
        await client.close()
        await manager.stop()


async def main():
    print(f"  {'prompt':<22} {'yếu tố':<13} {'':<11} {'':<9} {'':<11}")
    print("  " + "-" * 84)
    await do("hiện tại (297 từ)", None)
    await do("vừa (48 từ)", VUA)
    mau = await do("ngắn (32 từ)", NGAN)
    print("\n  Câu đời thường với prompt NGẮN:")
    for t in mau:
        print(f"    {t[:74]}")


asyncio.run(main())
