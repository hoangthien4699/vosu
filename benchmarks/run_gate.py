"""Chạy toàn bộ Benchmark Gate và in quyết định PASS / FAIL.

    Nếu bất kỳ mục nào FAIL:
        STOP -> tối ưu model/config -> đo lại
        (chưa code tiếp Pipeline/Frontend/Mobile)

Quy tắc phân loại ở đây:
  - B1-B6, B8  : có target cứng -> quyết định PASS/FAIL
  - B7, B9, B10: chỉ ghi nhận số liệu (§6 mục 8, 10, 11) -> KHÔNG tự PASS/FAIL,
                 nhưng BẮT BUỘC phải chạy và có người đọc kết quả. Gate chỉ được
                 coi là hoàn tất khi cả ba đã có dữ liệu.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .common import RESULTS_DIR, BenchmarkResult, environment, save_result


@dataclass
class Entry:
    id: str
    title: str
    module: str
    hard_gate: bool
    interactive: bool = False


REGISTRY = [
    Entry("B1", "STT riêng lẻ", "b1_stt", True),
    Entry("B2", "LLM riêng lẻ (TTFT + total)", "b2_llm", True),
    Entry("B3", "TTS riêng lẻ (CPU)", "b3_tts", True),
    Entry("B4", "VRAM khi Whisper + LLM active", "b4_vram", True),
    Entry("B5", "Concurrent Inference E2E", "b5_concurrent_e2e", True),
    Entry("B6", "CPU stress test", "b6_cpu_stress", True),
    Entry("B7", "Audio feedback loop", "b7_feedback_loop", False, interactive=True),
    Entry("B8", "Barge-in response time", "b8_barge_in", True),
    Entry("B9", "Speaker-source validation", "b9_speaker_source", False, interactive=True),
    Entry("B10", "Web mic vs Bluetooth mic", "b10_mic_compare", False, interactive=True),
]


class _Args:
    """Namespace tối thiểu mà từng benchmark mong đợi."""

    def __init__(self, runs=None, seconds=6.0, device=None):
        self.runs = runs
        self.seconds = seconds
        self.device = device
        self.json = False
        self.no_save = False


def run_one(entry: Entry, args: _Args) -> BenchmarkResult:
    import importlib

    module = importlib.import_module(f"benchmarks.{entry.module}")
    try:
        result = module.main(args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        result = BenchmarkResult(entry.id, entry.title)
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def verdict(results: dict[str, BenchmarkResult]) -> tuple[str, list[str]]:
    reasons: list[str] = []

    for entry in REGISTRY:
        result = results.get(entry.id)
        if result is None:
            reasons.append(f"{entry.id} chưa chạy")
            continue
        if result.status == "ERROR":
            reasons.append(f"{entry.id} lỗi: {result.error}")
        elif result.status == "FAIL":
            failed = [c.name for c in result.checks if c.passed is False]
            reasons.append(f"{entry.id} FAIL: {', '.join(failed)}")
        elif result.status == "SKIP":
            severity = "bỏ qua" if entry.hard_gate else "chưa đo"
            reasons.append(f"{entry.id} {severity}: {result.skipped}")

    return ("PASS" if not reasons else "FAIL"), reasons


def render_report(results: dict[str, BenchmarkResult], gate: str, reasons: list[str]) -> str:
    lines: list[str] = []
    env = environment()

    lines.append("=" * 76)
    lines.append("BENCHMARK GATE — AI Conversational Copilot")
    lines.append("=" * 76)
    for key in ("timestamp", "platform", "os", "machine", "gpu", "stt_device",
                "llm_gpu_layers", "whisper_model"):
        lines.append(f"  {key:<16} {env[key]}")

    if env["platform"] != "cuda":
        lines.append("")
        lines.append("  !! CẢNH BÁO: đây KHÔNG phải build CUDA.")
        lines.append("     Đặc tả §6 yêu cầu chốt Gate trên GPU NVIDIA 6GB.")
        lines.append("     Kết quả ở đây chỉ để phát triển, KHÔNG dùng để quyết định PASS.")

    lines.append("")
    for entry in REGISTRY:
        result = results.get(entry.id)
        if result is None:
            lines.append(f"[ CHƯA  ] {entry.id} — {entry.title}")
            continue
        lines.append(result.render())
        lines.append("")

    lines.append("=" * 76)
    lines.append(f"KẾT LUẬN GATE: {gate}")
    lines.append("=" * 76)
    if reasons:
        lines.append("Chưa đạt vì:")
        for reason in reasons:
            lines.append(f"  - {reason}")
        lines.append("")
        lines.append("  STOP -> tối ưu model/config -> đo lại.")
        lines.append("  Chưa code tiếp Pipeline/Frontend/Mobile (§6).")
    else:
        lines.append("Tất cả chỉ số P0 đạt target và cả 3 phép đo phần cứng đã có dữ liệu.")
        lines.append("Chuyển từ 'Conditionally Frozen' sang thực thi MVP đầy đủ.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="ID",
                        help="chỉ chạy các benchmark này, vd: --only B1 B2")
    parser.add_argument("--skip-interactive", action="store_true",
                        help="bỏ qua B7/B9/B10 (cần người vận hành + phần cứng)")
    parser.add_argument("-n", "--runs", type=int, default=None)
    parser.add_argument("--report", type=Path, default=RESULTS_DIR / "gate_report.txt")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    selected = REGISTRY
    if args.only:
        wanted = {i.upper() for i in args.only}
        selected = [e for e in REGISTRY if e.id in wanted]
        if not selected:
            print(f"Không có benchmark nào khớp {sorted(wanted)}", file=sys.stderr)
            return 2
    if args.skip_interactive:
        selected = [e for e in selected if not e.interactive]

    bench_args = _Args(runs=args.runs)
    results: dict[str, BenchmarkResult] = {}

    for entry in selected:
        print(f"\n>>> {entry.id} — {entry.title}", flush=True)
        try:
            result = run_one(entry, bench_args)
        except KeyboardInterrupt:
            print("\nĐã hủy giữa chừng.", file=sys.stderr)
            return 130
        results[entry.id] = result
        save_result(result)
        print(result.render(), flush=True)

    # Nếu chạy một phần, nạp thêm kết quả cũ để kết luận trên toàn bộ Gate
    if args.only or args.skip_interactive:
        from .common import load_result

        for entry in REGISTRY:
            if entry.id in results:
                continue
            stored = load_result(entry.id)
            if stored is None:
                continue
            results[entry.id] = _RestoredResult(stored, entry)

    gate, reasons = verdict(results)
    report = render_report(results, gate, reasons)
    print("\n" + report)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    summary = {
        "gate": gate,
        "reasons": reasons,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment(),
        "results": {k: v.to_dict() for k, v in results.items()},
    }
    (RESULTS_DIR / "gate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nBáo cáo: {args.report}")
    return 0 if gate == "PASS" else 1


class _RestoredResult(BenchmarkResult):
    """Kết quả nạp lại từ lần chạy trước (khi dùng --only/--skip-interactive)."""

    def __init__(self, stored: dict, entry: Entry) -> None:
        super().__init__(entry.id, f"{entry.title} (kết quả lần chạy trước)")
        self.skipped = stored.get("skipped", "")
        self.error = stored.get("error", "")
        self.details = stored.get("details", {})
        self._status = stored.get("status", "RECORD")

    @property
    def status(self) -> str:
        return self._status


if __name__ == "__main__":
    raise SystemExit(main())
