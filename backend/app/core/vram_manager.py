"""Quản lý tiến trình llama-server + đo VRAM (Task C1, C2).

Hai lỗi của baseline v1 mà module này tồn tại để sửa (§3.1):

1. `await asyncio.sleep(2)` rồi giả định server đã sẵn sàng. Đây là assumption
   nguy hiểm: model 3B nạp trên GPU yếu có thể mất >10s, mà máy nhanh thì 2s là
   lãng phí. Thay bằng health-check polling thật.

2. Không xử lý lỗi khởi động. Cần bắt: tiến trình thoát bất thường, timeout,
   port chưa sẵn sàng, lỗi nạp model, lỗi cấp phát VRAM (OOM).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Dấu hiệu OOM trong stderr của llama.cpp / CUDA.
_OOM_PATTERNS = (
    re.compile(r"out of memory", re.I),
    re.compile(r"CUDA error", re.I),
    re.compile(r"failed to allocate", re.I),
    re.compile(r"cudaMalloc failed", re.I),
)


def _absolute_binary(program: str) -> str:
    """Đường dẫn tuyệt đối — điều kiện để CPython dùng posix_spawn thay fork.

    Xem chú thích đầy đủ ở `ai/tts.py::_absolute_binary`.
    """
    resolved = shutil.which(program)
    if resolved:
        return str(Path(resolved).resolve())
    path = Path(program)
    if path.exists():
        return str(path.resolve())
    raise LlamaServerError(f"Không tìm thấy binary: {program!r}")


class LlamaServerError(RuntimeError):
    """Khởi động llama-server thất bại. Message nêu rõ nguyên nhân đã phân loại."""


class VramExceeded(RuntimeError):
    """Vượt hard ceiling (§3.1)."""


@dataclass(frozen=True)
class VramSnapshot:
    """Ảnh chụp VRAM. `available=False` khi không đo được (macOS/CPU)."""

    available: bool
    used_gb: float = 0.0
    total_gb: float = 0.0
    reason: str = ""

    @property
    def free_gb(self) -> float:
        return max(0.0, self.total_gb - self.used_gb)

    def describe(self) -> str:
        if not self.available:
            return f"VRAM: không đo được ({self.reason})"
        return f"VRAM: {self.used_gb:.2f}GB / {self.total_gb:.2f}GB"


def query_vram(device_index: int = 0) -> VramSnapshot:
    """Đo VRAM qua nvidia-smi. Trên macOS/CPU trả về available=False.

    Cố ý KHÔNG dùng torch: faster-whisper chạy trên CTranslate2, không cần
    torch, và bắt cả dự án cài torch chỉ để đọc một con số là quá đắt.
    """
    if shutil.which("nvidia-smi") is None:
        return VramSnapshot(False, reason="không có nvidia-smi (macOS/CPU build)")
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return VramSnapshot(False, reason=f"nvidia-smi lỗi: {exc}")

    if proc.returncode != 0 or not proc.stdout.strip():
        return VramSnapshot(False, reason=f"nvidia-smi trả về mã {proc.returncode}")

    try:
        used_mib, total_mib = (float(v) for v in proc.stdout.strip().split(",")[:2])
    except ValueError:
        return VramSnapshot(False, reason=f"không parse được: {proc.stdout.strip()!r}")

    return VramSnapshot(True, used_gb=used_mib / 1024.0, total_gb=total_mib / 1024.0)


class LlamaServerManager:
    """Vòng đời tiến trình con `llama-server`."""

    def __init__(self, config) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_tail: list[str] = []
        self._stderr_task: asyncio.Task | None = None
        self._started_at: float | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    def build_command(self) -> list[str]:
        cfg = self._config.llm
        binary = self._config.paths.resolve("llama_server_bin")
        program = str(binary) if binary.exists() else self._config.paths.llama_server_bin

        cmd = [
            program,
            "-m", str(self._config.paths.resolve("llm_gguf")),
            "-c", str(cfg.n_ctx),
            # ngl đến từ DeviceProfile: 36 trên CUDA, 99 (Metal) trên macOS.
            "-ngl", str(self._config.llm_gpu_layers),
            "--host", cfg.host,
            "--port", str(cfg.port),
        ]
        if cfg.continuous_batching:
            cmd.append("-cb")
        if cfg.swa_full:
            # Bắt buộc để tái dùng prefix cache với Gemma 3 (SWA). Xem chú
            # thích ở core/config.py::LlmConfig.swa_full.
            cmd.append("--swa-full")
        cmd.extend(cfg.extra_args)
        return cmd

    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        if self.is_running:
            return

        gguf = self._config.paths.resolve("llm_gguf")
        if not gguf.exists():
            raise LlamaServerError(
                f"Không tìm thấy model GGUF: {gguf}\n"
                "Chạy `scripts/download_models.sh` trước."
            )

        cmd = self.build_command()
        program = Path(cmd[0])
        if shutil.which(cmd[0]) is None and not program.exists():
            raise LlamaServerError(
                f"Không tìm thấy binary llama-server: {cmd[0]}\n"
                "Build llama.cpp (CUDA: -DGGML_CUDA=ON, macOS: -DGGML_METAL=ON) "
                "hoặc `brew install llama.cpp`."
            )

        logger.info("Khởi động llama-server: %s", " ".join(cmd))
        self._stderr_tail.clear()
        self._started_at = time.monotonic()

        try:
            self._process = await asyncio.create_subprocess_exec(
                _absolute_binary(cmd[0]), *cmd[1:],
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                # KHÔNG dùng start_new_session: nó buộc CPython đi đường
                # fork()+exec(), mà fork() sẽ deadlock trong atfork handler của
                # OpenMP/OpenBLAS nếu Whisper đang transcribe (xem chú thích ở
                # ai/tts.py::_absolute_binary). llama-server không sinh tiến
                # trình con nên không cần hạ theo nhóm — gửi tín hiệu thẳng pid.
                close_fds=False,
            )
        except OSError as exc:
            raise LlamaServerError(f"Không spawn được llama-server: {exc}") from exc

        self._stderr_task = asyncio.create_task(self._drain_stderr())

        try:
            await self._wait_until_healthy()
        except LlamaServerError:
            await self.stop()
            raise

        elapsed = time.monotonic() - self._started_at
        logger.info("llama-server sẵn sàng sau %.1fs (pid=%s)", elapsed, self.pid)

    async def _wait_until_healthy(self) -> None:
        """Poll /health cho tới khi trả 200 — KHÔNG dùng sleep cố định (§3.1)."""
        import httpx

        cfg = self._config.llm
        deadline = time.monotonic() + cfg.startup_timeout_s
        url = f"{cfg.base_url}/health"

        async with httpx.AsyncClient(timeout=2.0) as client:
            while True:
                # (a) tiến trình chết trước khi kịp lắng nghe?
                if self._process is not None and self._process.returncode is not None:
                    raise LlamaServerError(self._diagnose_exit())

                # (b) đã sẵn sàng chưa?
                with contextlib.suppress(Exception):
                    response = await client.get(url)
                    if response.status_code == 200:
                        return
                    # 503 = đang nạp model, hoàn toàn bình thường
                    if response.status_code not in (503, 500):
                        logger.debug("/health trả %d", response.status_code)

                # (c) hết giờ?
                if time.monotonic() > deadline:
                    raise LlamaServerError(
                        f"llama-server không sẵn sàng sau {cfg.startup_timeout_s:.0f}s "
                        f"(port {cfg.port}). "
                        f"stderr gần nhất:\n{self._stderr_snippet()}"
                    )

                await asyncio.sleep(cfg.health_poll_interval_s)

    def _diagnose_exit(self) -> str:
        code = self._process.returncode if self._process else None
        tail = self._stderr_snippet()

        for pattern in _OOM_PATTERNS:
            if pattern.search(tail):
                return (
                    f"llama-server thoát (mã {code}) do hết bộ nhớ GPU.\n"
                    f"Giảm -ngl (hiện {self._config.llm_gpu_layers}) hoặc "
                    f"n_ctx (hiện {self._config.llm.n_ctx}), "
                    f"hoặc dùng model lượng tử hóa nhỏ hơn.\n{tail}"
                )
        if "bind" in tail.lower() or "address already in use" in tail.lower():
            return (
                f"Port {self._config.llm.port} đã bị chiếm. "
                f"Dừng tiến trình llama-server cũ hoặc đổi llm.port.\n{tail}"
            )
        return f"llama-server thoát sớm (mã {code}).\n{tail}"

    def _stderr_snippet(self, lines: int = 15) -> str:
        return "\n".join(self._stderr_tail[-lines:]) or "(stderr trống)"

    async def _drain_stderr(self) -> None:
        """Đọc stderr liên tục — nếu không đọc, pipe đầy sẽ treo tiến trình con."""
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                self._stderr_tail.append(text)
                if len(self._stderr_tail) > 200:
                    del self._stderr_tail[:100]
                for pattern in _OOM_PATTERNS:
                    if pattern.search(text):
                        logger.error("llama-server: %s", text)
                        break
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            logger.debug("Ngừng đọc stderr của llama-server", exc_info=True)

    # ------------------------------------------------------------------ #

    async def stop(self, timeout: float = 5.0) -> None:
        process, self._process = self._process, None

        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None

        if process is None or process.returncode is not None:
            return

        logger.info("Dừng llama-server (pid=%s)", process.pid)
        with contextlib.suppress(ProcessLookupError, OSError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("llama-server không phản hồi SIGTERM — SIGKILL.")
            with contextlib.suppress(ProcessLookupError, OSError):
                process.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=timeout)


class VramGuard:
    """Theo dõi VRAM so với hard ceiling (§3.1).

    Trên build macOS/CPU, `query_vram` trả available=False và guard tự động
    trở thành no-op — unified memory không có cùng ý nghĩa vật lý với VRAM rời,
    nên áp ceiling 5.5GB ở đó là vô nghĩa.
    """

    def __init__(self, config) -> None:
        self._config = config
        self._peak_gb = 0.0
        self._exceeded_count = 0

    @property
    def peak_gb(self) -> float:
        return self._peak_gb

    @property
    def exceeded_count(self) -> int:
        return self._exceeded_count

    def sample(self) -> VramSnapshot:
        snapshot = query_vram()
        if not snapshot.available:
            return snapshot

        self._peak_gb = max(self._peak_gb, snapshot.used_gb)
        ceiling = self._config.vram.hard_ceiling_gb
        if snapshot.used_gb <= ceiling:
            return snapshot

        self._exceeded_count += 1
        policy = self._config.vram.on_exceed
        message = (
            f"VRAM {snapshot.used_gb:.2f}GB vượt hard ceiling {ceiling:.2f}GB "
            f"(lần thứ {self._exceeded_count})"
        )
        if policy == "reject":
            raise VramExceeded(message)
        logger.warning("%s — chính sách: %s", message, policy)
        return snapshot
