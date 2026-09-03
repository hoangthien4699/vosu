"""Platform abstraction: CUDA (NVIDIA) build vs macOS (Metal) build vs plain CPU.

Toàn bộ phần còn lại của backend không được phép hardcode "cuda" hay "-ngl 36".
Mọi khác biệt giữa hai build phải đi qua `DeviceProfile` ở đây.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Platform(str, Enum):
    CUDA = "cuda"
    MACOS = "macos"
    CPU = "cpu"


@dataclass(frozen=True)
class DeviceProfile:
    """Cấu hình phần cứng đã phân giải cho tiến trình hiện tại."""

    platform: Platform

    # --- STT (faster-whisper / CTranslate2) ---
    stt_device: str
    stt_compute_type: str
    stt_device_index: int = 0

    # --- LLM (llama-server) ---
    # CUDA: -ngl 36 đủ để offload Qwen2.5-3B. Metal: -ngl 99 offload toàn bộ.
    # CPU: 0.
    llm_gpu_layers: int = 0
    llm_extra_args: tuple[str, ...] = ()

    # --- Đo đạc ---
    # Chỉ CUDA mới có VRAM rời để enforce hard ceiling 5.5GB.
    # macOS dùng unified memory nên "VRAM ceiling" không cùng ý nghĩa vật lý.
    has_dedicated_vram: bool = False
    vram_query_available: bool = False

    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_cuda(self) -> bool:
        return self.platform is Platform.CUDA

    @property
    def is_macos(self) -> bool:
        return self.platform is Platform.MACOS

    def describe(self) -> str:
        return (
            f"platform={self.platform.value} "
            f"stt={self.stt_device}/{self.stt_compute_type} "
            f"llm_ngl={self.llm_gpu_layers} "
            f"vram_query={'yes' if self.vram_query_available else 'no'}"
        )


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _cuda_runtime_present() -> bool:
    """nvidia-smi tồn tại VÀ thực sự trả về ít nhất một GPU."""
    if not nvidia_smi_available():
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def detect_platform() -> Platform:
    if _cuda_runtime_present():
        return Platform.CUDA
    if platform.system() == "Darwin":
        return Platform.MACOS
    return Platform.CPU


CUDA_PROFILE = DeviceProfile(
    platform=Platform.CUDA,
    stt_device="cuda",
    stt_compute_type="int8_float16",
    llm_gpu_layers=36,
    has_dedicated_vram=True,
    vram_query_available=True,
    notes=(
        "Build mục tiêu của đặc tả: GPU 6GB, hard ceiling 5.5GB.",
        "Whisper + llama-server tranh chấp SM — bắt buộc đo Concurrent Inference E2E (B5).",
    ),
)

MACOS_PROFILE = DeviceProfile(
    platform=Platform.MACOS,
    # CTranslate2 (backend của faster-whisper) KHÔNG hỗ trợ Metal/MPS.
    # Trên Apple Silicon, STT chạy CPU với int8 là đường duy nhất còn dùng chung
    # được codepath với build CUDA.
    stt_device="cpu",
    stt_compute_type="int8",
    # llama.cpp thì CÓ Metal — offload toàn bộ layer.
    llm_gpu_layers=99,
    has_dedicated_vram=False,
    vram_query_available=False,
    notes=(
        "STT chạy CPU (CTranslate2 không có backend Metal) — latency sẽ cao hơn build CUDA.",
        "LLM chạy Metal qua llama.cpp.",
        "Unified memory: hard ceiling 5.5GB của đặc tả KHÔNG áp dụng trực tiếp.",
        "Build này dùng để phát triển/kiểm thử logic, không dùng để chốt Benchmark Gate.",
    ),
)

CPU_PROFILE = DeviceProfile(
    platform=Platform.CPU,
    stt_device="cpu",
    stt_compute_type="int8",
    llm_gpu_layers=0,
    has_dedicated_vram=False,
    vram_query_available=False,
    notes=("Fallback thuần CPU — chỉ dùng cho CI/test, không đạt target latency.",),
)

_PROFILES = {
    Platform.CUDA: CUDA_PROFILE,
    Platform.MACOS: MACOS_PROFILE,
    Platform.CPU: CPU_PROFILE,
}


def resolve_profile(override: str | None = None) -> DeviceProfile:
    """Chọn profile. `override` (từ config/env) thắng auto-detect.

    Cho phép ép profile không khớp máy hiện tại — hữu ích khi sinh config
    hoặc chạy unit test cho build kia.
    """
    if override:
        try:
            requested = Platform(override.lower())
        except ValueError as exc:
            valid = ", ".join(p.value for p in Platform)
            raise ValueError(
                f"platform không hợp lệ: {override!r}. Hợp lệ: {valid}"
            ) from exc
        detected = detect_platform()
        if requested is not detected:
            logger.warning(
                "Ép platform=%s nhưng máy hiện tại detect ra %s — "
                "một số thành phần có thể không khởi động được.",
                requested.value,
                detected.value,
            )
        return _PROFILES[requested]
    return _PROFILES[detect_platform()]
