"""Web client được phục vụ đúng cách.

Kiểu hỏng mà file này khóa lại: trang mở ra trắng trơn, không CSS không JS,
mà trình duyệt không báo lỗi gì rõ ràng — chỉ vài dòng 404 lẫn trong console.
"""
from __future__ import annotations

import contextlib
import re

import pytest
from fastapi.testclient import TestClient

from app.core.config import REPO_ROOT

CLIENT_DIR = REPO_ROOT / "client" / "web_test"


@pytest.fixture
def client(monkeypatch):
    # Không nạp model: chỉ kiểm tra tầng phục vụ file tĩnh.
    from app import main as main_module

    app = main_module.app
    app.router.lifespan_context = _noop_lifespan
    return TestClient(app)


@contextlib.asynccontextmanager
async def _noop_lifespan(app):
    yield


def test_goc_chuyen_huong_sang_app(client):
    """index.html tham chiếu asset theo đường dẫn TƯƠNG ĐỐI.

    Phục vụ nó tại "/" thì `style.css` phân giải thành `/style.css` — 404.
    Trang hiện ra không CSS không JS mà không có lỗi nào dễ thấy.
    """
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/app/"


def test_moi_asset_tuong_doi_trong_html_deu_tai_duoc(client):
    html = client.get("/app/").text
    references = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert references, "index.html không tham chiếu asset nào — có vẻ sai"

    for reference in references:
        if reference.startswith(("http://", "https://", "//", "#")):
            continue
        response = client.get(f"/app/{reference}")
        assert response.status_code == 200, (
            f"{reference!r} trả {response.status_code} — trang sẽ hỏng im lặng"
        )
        assert response.content, f"{reference!r} rỗng"


def test_asset_ton_tai_tren_dia():
    for name in ("index.html", "app.js", "style.css"):
        assert (CLIENT_DIR / name).exists(), f"thiếu {name}"


def test_html_va_js_khop_id():
    """JS truy vấn ID không có trong HTML thì phần đó chết lặng."""
    html = (CLIENT_DIR / "index.html").read_text(encoding="utf-8")
    js = (CLIENT_DIR / "app.js").read_text(encoding="utf-8")

    html_ids = set(re.findall(r'id="([^"]+)"', html))
    js_ids = set(re.findall(r'el\("([^"]+)"\)', js))

    assert not (js_ids - html_ids), f"JS tìm ID không tồn tại: {sorted(js_ids - html_ids)}"


def test_ui_xu_ly_moi_event_backend_phat():
    """Thêm event type ở backend mà quên UI thì nó bị bỏ qua âm thầm."""
    from app.protocol.events import EventType

    js = (CLIENT_DIR / "app.js").read_text(encoding="utf-8")
    handled = set(re.findall(r'case "([a-z_]+)":', js))
    backend = {e.value for e in EventType}

    assert not (backend - handled), f"UI không xử lý: {sorted(backend - handled)}"
    assert not (handled - backend), f"UI xử lý event không tồn tại: {sorted(handled - backend)}"


def test_tua_nhanh_qua_im_lang_khi_phat_file():
    """Sau khi đọc xong bản dịch, file phát tiếp từ GIỮA khoảng nghỉ.

    Người nói thật nghỉ 2-4 giây giữa hai câu, mà điểm dừng chỉ ăn 0.7s đầu —
    còn lại là im lặng phải ngồi nghe. Đo trên file nghỉ 2.5s: chết 1877ms sau
    mỗi lượt đọc; tua 8x còn 283ms.

    Gửi nhanh hơn thời gian thực không ảnh hưởng VAD vì nó đếm theo MẪU audio,
    không theo đồng hồ — nhưng chỉ đúng khi VẪN GỬI ĐỦ MỌI MẪU, chỉ gửi nhanh.
    Test này khóa đúng điều đó: nhánh im lặng phải có `ws.send`.
    """
    js = (CLIENT_DIR / "app.js").read_text(encoding="utf-8")

    assert "SILENCE_SPEEDUP" in js and "hasSpeech" in js

    start = js.index("if (!speech && ui.autoPause.checked)")
    branch = js[start : js.index("continue;", start)]
    assert "ws.send(" in branch, "tua nhanh mà BỎ mẫu thì VAD sẽ cắt câu sai chỗ"
    assert "scheduleFileChunk" not in branch, "không phát im lặng ra loa"
    assert "SILENCE_SPEEDUP" in branch

    # Gặp tiếng nói thì lập tức về nhịp thật, và đặt lại mốc lịch phát: mốc cũ
    # đã trôi qua từ lâu trong lúc tua.
    reset_at = js.index("if (skipping)")
    after = js[reset_at : js.index("scheduleFileChunk", reset_at)]
    assert "f.head = ctx.currentTime" in after
