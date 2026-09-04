"""Mọi giá trị code thật sự phát ra đều phải qua được schema.

Đã hỏng hai lần theo cùng một kiểu: đổi `reason` của TTS_CANCELLED và thêm
trường vào TTS_DONE mà quên schema. Event không hợp lệ bị CHẶN LẠI im lặng —
không có lỗi nào ném ra, client chỉ đơn giản không bao giờ nhận được nó, và
chế độ dừng-từng-câu treo mãi vì đang chờ đúng event ấy.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from app.protocol.schemas import TtsCancelledData

SRC = pathlib.Path(__file__).resolve().parents[1] / "backend" / "app"
_REASON = re.compile(r'reason="([a-z_]+)"')


def _reasons_in_code() -> set[str]:
    found: set[str] = set()
    for path in (SRC / "api" / "websocket.py", SRC / "ai" / "tts.py"):
        found |= set(_REASON.findall(path.read_text(encoding="utf-8")))
    return found


@pytest.mark.parametrize("reason", sorted(_reasons_in_code()))
def test_moi_ly_do_huy_tts_deu_qua_duoc_schema(reason):
    TtsCancelledData(reason=reason, response_ms=1.0, chunks_sent=0)


def test_tts_done_nhan_truong_prewarmed():
    from app.protocol.schemas import TtsDoneData

    assert TtsDoneData(chunks=1, synthesis_ms=1.0, prewarmed=True).prewarmed is True
    assert TtsDoneData(chunks=1, synthesis_ms=1.0).prewarmed is False
