"""Điểm khởi động FastAPI."""

from __future__ import annotations

import contextlib
import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.websocket import router as websocket_router
from .core.config import REPO_ROOT, get_config
from .core.runtime import ModelRuntime

logger = logging.getLogger(__name__)

WEB_CLIENT_DIR = REPO_ROOT / "client" / "web_test"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    configure_logging(config.server.log_level)

    device = config.device
    logger.info("=" * 68)
    logger.info("vosu — AI Conversational Copilot")
    logger.info("Build: %s", device.describe())
    for note in device.notes:
        logger.info("  · %s", note)
    logger.info("=" * 68)

    runtime = ModelRuntime(config)
    app.state.config = config
    app.state.runtime = runtime

    try:
        await runtime.start()
    except Exception:
        # Server vẫn phải lên để /health nói được vì sao hỏng — chết im lặng
        # lúc khởi động là kiểu lỗi khó chẩn đoán nhất khi chạy trên máy khác.
        logger.exception("Model Runtime khởi động thất bại — server chạy ở chế độ hỏng.")

    try:
        yield
    finally:
        await runtime.shutdown()


app = FastAPI(
    title="vosu — AI Conversational Copilot",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(websocket_router)


@app.get("/health")
async def health() -> JSONResponse:
    runtime: ModelRuntime | None = getattr(app.state, "runtime", None)
    if runtime is None:
        return JSONResponse({"ready": False, "reason": "runtime chưa khởi tạo"}, 503)
    payload = await runtime.health()
    return JSONResponse(payload, status_code=200 if payload["ready"] else 503)


@app.get("/config")
async def show_config() -> JSONResponse:
    """Config đang có hiệu lực — để đối chiếu khi chạy trên máy benchmark."""
    config = app.state.config
    payload = config.to_dict()
    payload["_device"] = {
        "platform": config.device.platform.value,
        "stt_device": config.device.stt_device,
        "stt_compute_type": config.device.stt_compute_type,
        "llm_gpu_layers_effective": config.llm_gpu_layers,
        "notes": list(config.device.notes),
    }
    return JSONResponse(payload)


if WEB_CLIENT_DIR.exists():
    app.mount("/app", StaticFiles(directory=WEB_CLIENT_DIR, html=True), name="web_test")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_CLIENT_DIR / "index.html")
