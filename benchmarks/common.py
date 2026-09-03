"""Hạ tầng chung cho Benchmark Gate (§6/§7).

Nguyên tắc báo cáo (§7, review v4.1): realtime system KHÔNG được đo bằng một
sample đơn lẻ. Mọi phép đo lặp lại đều báo cáo theo percentile.

    "P95 là con số quan trọng nhất để theo dõi, không phải P50 — một hệ thống
     chạy 800ms ở phần lớn trường hợp nhưng cứ 5 câu lại có 1 câu vọt lên
     2.5s vẫn không đạt chuẩn realtime tốt."
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

sys.path.insert(0, str(REPO_ROOT / "backend"))


# --------------------------------------------------------------------------- #
# Thống kê
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile theo nội suy tuyến tính (khớp numpy.percentile mặc định)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


@dataclass
class Distribution:
    n: int
    p50: float
    p90: float
    p95: float
    max: float
    mean: float
    min: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Distribution:
        if not values:
            nan = float("nan")
            return cls(0, nan, nan, nan, nan, nan, nan)
        return cls(
            n=len(values),
            p50=percentile(values, 50),
            p90=percentile(values, 90),
            p95=percentile(values, 95),
            max=max(values),
            mean=sum(values) / len(values),
            min=min(values),
        )

    def summary(self, unit: str = "ms") -> str:
        if self.n == 0:
            return "không có mẫu"
        return (
            f"n={self.n} P50={self.p50:.0f}{unit} P90={self.p90:.0f}{unit} "
            f"P95={self.p95:.0f}{unit} Max={self.max:.0f}{unit}"
        )


# --------------------------------------------------------------------------- #
# Kết quả
# --------------------------------------------------------------------------- #


@dataclass
class Check:
    """Một tiêu chí có target cứng."""

    name: str
    value: float | None
    target: float | None
    unit: str = "ms"
    #: "lte" = đạt khi value <= target; "record" = chỉ ghi nhận, không PASS/FAIL
    mode: str = "lte"
    note: str = ""

    @property
    def passed(self) -> bool | None:
        if self.mode == "record" or self.target is None or self.value is None:
            return None
        return self.value <= self.target

    def render(self) -> str:
        if self.value is None:
            status, value = "SKIP", "—"
        else:
            value = f"{self.value:.2f}{self.unit}"
            status = {True: "PASS", False: "FAIL", None: "GHI"}[self.passed]
        target = "—" if self.target is None else f"<= {self.target:.2f}{self.unit}"
        return f"  [{status}] {self.name:<38} {value:>12}   target {target}"


@dataclass
class BenchmarkResult:
    id: str
    title: str
    checks: list[Check] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    skipped: str = ""
    error: str = ""

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        if self.skipped:
            return "SKIP"
        verdicts = [c.passed for c in self.checks if c.passed is not None]
        if not verdicts:
            return "RECORD"
        return "PASS" if all(verdicts) else "FAIL"

    def render(self) -> str:
        lines = [f"[{self.status:^6}] {self.id} — {self.title}"]
        if self.skipped:
            lines.append(f"  bỏ qua: {self.skipped}")
        if self.error:
            lines.append(f"  lỗi: {self.error}")
        lines.extend(c.render() for c in self.checks)
        for key, value in self.details.items():
            lines.append(f"  · {key}: {value}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "checks": [asdict(c) | {"passed": c.passed} for c in self.checks],
            "details": self.details,
            "skipped": self.skipped,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Môi trường
# --------------------------------------------------------------------------- #


def environment() -> dict[str, Any]:
    from app.core.config import load_config

    config = load_config()
    device = config.device
    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": device.platform.value,
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "stt_device": f"{device.stt_device}/{device.stt_compute_type}",
        "llm_gpu_layers": config.llm_gpu_layers,
        "whisper_model": config.paths.whisper_model,
    }
    info["gpu"] = _gpu_name()
    return info


def _gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                return f"{out.stdout.strip()} (Apple Silicon, không có VRAM rời)"
        except (OSError, subprocess.SubprocessError):
            pass
    return "không xác định"


def save_result(result: BenchmarkResult, extra: dict | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"environment": environment(), **result.to_dict(), **(extra or {})}
    path = RESULTS_DIR / f"{result.id.lower()}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_result(benchmark_id: str) -> dict | None:
    path = RESULTS_DIR / f"{benchmark_id.lower()}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Chạy
# --------------------------------------------------------------------------- #


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-n", "--runs", type=int, default=None,
                        help="số lần lặp (mặc định tùy từng benchmark)")
    parser.add_argument("--json", action="store_true", help="chỉ in JSON")
    parser.add_argument("--no-save", action="store_true", help="không ghi vào results/")
    return parser


def run_cli(fn: Callable[[argparse.Namespace], BenchmarkResult], parser) -> int:
    args = parser.parse_args()
    try:
        result = fn(args)
    except KeyboardInterrupt:
        print("\nĐã hủy.", file=sys.stderr)
        return 130

    if not args.no_save:
        save_result(result)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.render())
    return 0 if result.status in ("PASS", "RECORD", "SKIP") else 1


class Timer:
    """`with Timer() as t: ...` -> t.ms"""

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
