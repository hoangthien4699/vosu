"""Lượt hai chỉnh văn: được tiểu từ, mất bao nhiêu độ trung thực?

    python -m benchmarks.two_pass

BỐI CẢNH. Thiếu tiểu từ tình thái (nhé/ạ/thôi/nhỉ) là lỗi duy nhất còn lại sau
khi đã thử: bốn model (Qwen3.5-4B Q4 và Q8, Sailor2-1B, Sailor2-8B), bốn tham
số giải mã, và sáu vòng chỉnh prompt. Tôi đã hai lần kết luận đó là GIỚI HẠN
NĂNG LỰC MODEL.

KẾT LUẬN ĐÓ SAI. Vấn đề nằm ở cách đặt bài toán: prompt bắt dịch trung thực,
mà tiểu từ là chữ KHÔNG CÓ trong câu gốc. Bảo model vừa trung thành vừa thêm
chữ là hai lệnh mâu thuẫn. Tách làm hai lượt — dịch, rồi chỉnh văn — thì CÙNG
MODEL ĐÓ cho 5/5.

Đáng chú ý: Sailor2 là model chuyên Đông Nam Á và vẫn 4/5. Không phải "model
đa ngữ kém tiếng Việt", mà là "một model làm hai việc mâu thuẫn".

CÁI GIÁ, đo trên 26 ca:

    một lượt   60/62 (97%)   2/26 thiếu ý   0 đảo nghĩa   1/5 tiểu từ   2142ms
    hai lượt   57/62 (92%)   5/26 thiếu ý   0 đảo nghĩa   5/5 tiểu từ   3349ms

Không đảo nghĩa ở cả hai — lượt chỉnh văn làm RƠI chi tiết chứ không lật nghĩa.
Đó là loại lỗi chặn được bằng cách kiểm bản viết lại và quay về bản gốc nếu nó
đánh mất con số hay tên riêng. Chưa làm.

Lượt chỉnh văn PHẢI tắt grammar: nó trả văn xuôi, để grammar bật thì bị ép ra
JSON và toàn bộ phép đo thành vô nghĩa (đã mắc).
"""
import asyncio
import statistics
import sys
import time

sys.path.insert(0, "backend")
sys.path.insert(0, ".")
from benchmarks.fidelity_cases import ALL, has_particle, inverted, missing

HE_THONG = """Bạn là người Việt, đang nói chuyện với đồng nghiệp.

Viết lại câu tiếng Việt được đưa cho nghe như LỜI NÓI hằng ngày.

BẮT BUỘC giữ nguyên: mọi con số, mọi tên riêng, ai làm gì với ai, và phủ định.
Không thêm thông tin mới, không đổi người nói hay người nghe.

Người Việt nói chuyện hay có tiểu từ cuối câu: nhé, nhỉ, thôi, ạ, mà, đấy,
chứ, vậy. Thêm nếu hợp, đừng gượng.

Chỉ trả về câu đã viết lại."""

async def main():
    from app.ai.copilot import SemanticEventParser
    from app.ai.direction import Direction
    from app.ai.llm import LlmClient
    from app.core.config import load_config
    from app.core.vram_manager import LlamaServerManager

    config = load_config()
    manager = LlamaServerManager(config)
    await manager.start()
    client = LlmClient(config)
    await client.start()
    try:
        for nhan, bat in (("MỘT LƯỢT (nền)", False), ("HAI LƯỢT", True)):
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
                dich = par.result.translation
                # Chỉ chỉnh văn cho ĐẦU RA TIẾNG VIỆT — tiểu từ là chuyện của
                # tiếng Việt, câu tiếng Anh thì không liên quan.
                if bat and not out and dich.strip():
                    config.llm.grammar = False
                    try:
                        raw2, _ = await client.complete(
                            client.template.render(HE_THONG, dich))
                        moi = raw2.strip().split("\n")[0].strip().strip('"')
                        if moi:
                            dich = moi
                    finally:
                        config.llm.grammar = True
                lat.append((time.perf_counter() - t0) * 1000)
                rows.append((c, dich))

            tot = sum(len(c.must_keep) for c, _ in rows)
            mat = sum(len(missing(c, t)) for c, t in rows)
            dao = sum(1 for c, t in rows if inverted(c, t))
            casual = [(c, t) for c, t in rows if c.casual and c.lang == "en"]
            co_tu = sum(1 for _, t in casual if has_particle(t))
            print(f"\n  {nhan}")
            print(f"    yếu tố giữ được : {tot-mat}/{tot} ({(tot-mat)/tot*100:.0f}%)")
            print(f"    câu bị thiếu ý  : {sum(1 for c,t in rows if missing(c,t))}/{len(rows)}")
            print(f"    câu bị đảo nghĩa: {dao}/{len(rows)}")
            print(f"    CÓ tiểu từ      : {co_tu}/{len(casual)}")
            print(f"    P50 mỗi câu     : {statistics.median(lat):.0f}ms")
    finally:
        await client.close()
        await manager.stop()

asyncio.run(main())
