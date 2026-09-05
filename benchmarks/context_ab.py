"""Ngữ cảnh có làm bản dịch đúng hơn không?

    python -m benchmarks.context_ab


Thiết kế mạnh nhất có thể: CÙNG MỘT CÂU NGUỒN, hai ngữ cảnh khác nhau, đòi hai
bản dịch KHÁC NHAU. Không ngữ cảnh thì model chỉ đoán được một trong hai.

Hệ thống đã có bộ nhớ 6 lượt và production luôn truyền vào, nhưng không
benchmark nào truyền — nên mọi số đo trước giờ đều ở điều kiện không ngữ cảnh.

KẾT QUẢ: 3/5 -> 4/5. Ngữ cảnh đổi 2 trong 5 đầu ra, một đổi rõ theo hướng đúng:

    "I got it." sau "Did you receive the invoice?"
        không ngữ cảnh: "Mình hiểu rồi."     (sai nghĩa)
        có ngữ cảnh   : "Mình đã nhận rồi."  (đúng)

Nhưng KHÔNG phải thuốc chữa bách bệnh: "That's fine." sau một lời xin lỗi vẫn
ra "Được thôi" thay vì "Không sao" — model vẫn đọc thành đồng ý với đề xuất
chứ không phải trấn an.

n=5 là quá nhỏ để nói được tỷ lệ. Đây là tín hiệu có thật, không phải kết luận.

CẢNH BÁO ĐÃ TRẢ GIÁ (lần thứ năm cùng kiểu): tiêu chí chấm ban đầu đòi đúng
chuỗi "nhận được", model trả "đã nhận rồi" nên bị chấm sai — che mất đúng cái
ca mà ngữ cảnh phát huy tác dụng. Và cặp "Right." bị BỎ chứ không nới tiêu chí:
"Đúng vậy" là cách đáp hợp lệ cho cả hai ngữ cảnh nên ca đó không phân biệt
được gì. Nới tiêu chí cho tới khi mọi thứ đều đạt là tự lừa mình.
"""
import asyncio
import sys

sys.path.insert(0, "backend")
sys.path.insert(0, ".")

# (câu nguồn, lượt trước, phải có MỘT trong, KHÔNG được có)
CA = [
    ("That's fine.", 'Them: "Sorry, I can\'t come tomorrow."',
     ["không sao", "không vấn đề", "chẳng sao"], ["được thôi", "vậy cũng được", "ổn thôi"]),
    ("That's fine.", 'Them: "Should we push the launch to next month?"',
     ["được", "ổn", "vậy cũng"], ["không sao"]),

    ("I got it.", 'Them: "Did you receive the invoice?"',
     ["nhận"], ["hiểu rồi", "biết rồi"]),
    ("I got it.", 'Them: "Let me explain how the pricing works."',
     ["hiểu rồi", "rõ rồi", "biết rồi"], ["nhận được"]),

    ("Not yet.", 'Them: "Have you signed the contract?"',
     ["chưa"], ["không"]),
]


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
        for nhan, dung_ctx in (("KHÔNG ngữ cảnh", False), ("CÓ ngữ cảnh", True)):
            dat = 0
            print(f"\n  {nhan}")
            for cau, truoc, phai_co, cam in CA:
                raw, _ = await client.complete(
                    client.build_prompt(cau, "en", direction=Direction.TO_USER,
                                        counterpart_language="en",
                                        history=truoc if dung_ctx else ""))
                par = SemanticEventParser()
                par.feed(raw)
                par.finish()
                t = par.result.translation.lower()
                ok = any(x in t for x in phai_co) and not any(x in t for x in cam)
                dat += ok
                ngan = truoc.split(': "')[1][:34]
                print(f"    {'✓' if ok else '✗'} sau \"{ngan}...\"")
                print(f"        {cau!r} -> {par.result.translation[:48]!r}")
            print(f"    => đúng theo ngữ cảnh: {dat}/{len(CA)}")
    finally:
        await client.close()
        await manager.stop()


asyncio.run(main())
