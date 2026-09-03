"""So sánh hai model LLM trên cùng bộ câu — công cụ, KHÔNG phải mục của Gate.

Lý do tồn tại: Qwen2.5-3B (model gốc của đặc tả) rò tiếng Trung vào bản dịch
tiếng Việt khi chạy thật. "Model nào tốt hơn" là câu hỏi phải trả lời bằng số
liệu trên chính prompt của dự án, không phải bằng bảng xếp hạng chung.

    python -m benchmarks.compare_models \\
        --model models/gemma-3-4b-it-q4_k_m.gguf \\
        --model models/qwen2.5-3b-instruct-q4_k_m.gguf

Mỗi model được khởi động riêng, lần lượt, để không tranh chấp GPU/VRAM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .common import (  # noqa: F401
    REPO_ROOT,
    BenchmarkResult,
    Check,
    Distribution,
    environment,
    save_result,
)

# Tiếng Việt dùng chữ Latin + dấu. Bất kỳ ký tự CJK/Kana/Hangul nào trong bản
# dịch tiếng Việt đều là lỗi rò ngôn ngữ.
_CJK = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]"
)

UTTERANCES = [
    ("en", "I think we should table this discussion for now."),
    ("en", "Could you walk me through the pricing structure again?"),
    ("en", "Let's circle back on this next week when everyone has the numbers."),
    ("en", "Is there a major blocker we need to resolve first?"),
    ("en", "I'm not entirely convinced this is the right approach."),
    ("en", "We'll need sign-off from legal before we can proceed."),
    ("ja", "その件については社内で確認させていただきます。"),
    ("zh", "这个方案我们需要再讨论一下。"),
]


@dataclass
class ModelReport:
    path: str
    template: str = ""
    ttft_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    json_ok: int = 0
    leaked: int = 0
    empty_translation: int = 0
    missing_replies: int = 0
    errors: int = 0

    @property
    def n(self) -> int:
        return len(self.samples)


def cjk_chars(text: str) -> list[str]:
    return _CJK.findall(text)


async def evaluate(model_path: Path, runs: int, n_predict: int) -> ModelReport:
    from app.ai.copilot import SemanticEventParser
    from app.ai.llm import LlmClient
    from app.core.config import load_config
    from app.core.vram_manager import LlamaServerManager

    config = load_config()
    config.paths.llm_gguf = str(model_path)
    config.llm.prompt_template = "auto"
    config.llm.n_predict = n_predict

    report = ModelReport(path=model_path.name)
    manager = LlamaServerManager(config)
    client = LlmClient(config)
    report.template = client.template.name

    await manager.start()
    await client.start()
    try:
        # warm-up không tính vào kết quả
        language, text = UTTERANCES[0]
        await client.complete(client.build_prompt(text, language))

        for index in range(runs):
            language, text = UTTERANCES[index % len(UTTERANCES)]
            parser = SemanticEventParser()
            try:
                # complete() tự tạo GenerationStats và trả về — không truyền vào
                raw, stats = await client.complete(client.build_prompt(text, language))
            except Exception as exc:
                report.errors += 1
                report.samples.append({"input": text, "error": str(exc)[:120]})
                continue

            for token in (raw,):
                parser.feed(token)
            parser.finish()
            result = parser.result

            if stats.ttft_ms is not None:
                report.ttft_ms.append(stats.ttft_ms)
            report.total_ms.append(stats.total_ms)

            if not result.malformed:
                report.json_ok += 1
            # Bản dịch phải LUÔN là tiếng Việt thuần, nên ký tự CJK trong đó
            # là lỗi bất kể nguồn nói tiếng gì. Chỉ soi `translation` —
            # `replies` mang ký tự CJK là ĐÚNG khi người nói dùng tiếng Trung
            # hoặc Nhật, vì người dùng cần nói lại bằng chính ngôn ngữ đó.
            leaked = cjk_chars(result.translation)
            if leaked:
                report.leaked += 1
            if not result.translation.strip():
                report.empty_translation += 1
            if len(result.replies) < 2:
                report.missing_replies += 1

            report.samples.append({
                "input": text,
                "lang": language,
                "translation": result.translation,
                "intent": result.intent,
                "replies": result.replies,
                "cjk_leak": "".join(leaked),
                "malformed": result.malformed,
                "truncated": stats.truncated,
                "ttft_ms": round(stats.ttft_ms or 0.0, 1),
                "total_ms": round(stats.total_ms, 1),
            })
    finally:
        await client.close()
        await manager.stop()

    return report


def render(reports: list[ModelReport], verbose: bool) -> str:
    lines = ["=" * 76, "SO SÁNH MODEL — trên chính prompt của dự án", "=" * 76]
    env = environment()
    lines.append(f"  {env['platform']} · {env['gpu']}")
    lines.append("")

    header = f"  {'model':<34} {'tmpl':<7} {'JSON ok':<9} {'rò CJK':<9} {'TTFT P50':<10} {'total P50'}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in reports:
        ttft = Distribution.of(r.ttft_ms)
        total = Distribution.of(r.total_ms)
        lines.append(
            f"  {r.path:<34} {r.template:<7} "
            f"{r.json_ok}/{r.n:<7} {r.leaked}/{r.n:<7} "
            f"{ttft.p50:>7.0f}ms  {total.p50:>8.0f}ms"
        )
        if r.errors or r.empty_translation or r.missing_replies:
            lines.append(
                f"    (lỗi {r.errors} · dịch rỗng {r.empty_translation} "
                f"· thiếu reply {r.missing_replies})"
            )
    lines.append("")

    for r in reports:
        lines.append(f"--- {r.path} ---")
        for sample in r.samples[: (len(r.samples) if verbose else 4)]:
            if "error" in sample:
                lines.append(f"  LỖI: {sample['error']}")
                continue
            flag = "  <-- RÒ CJK" if sample["cjk_leak"] else ""
            lines.append(f"  [{sample['lang']}] {sample['input'][:58]}")
            lines.append(f"      dịch  : {sample['translation'][:70]}{flag}")
            lines.append(f"      intent: {sample['intent'][:60]}")
            for i, reply in enumerate(sample["replies"][:2]):
                lines.append(f"      reply{i}: {reply[:60]}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", action="append", required=True, metavar="PATH",
                        help="đường dẫn GGUF; lặp lại để so nhiều model")
    parser.add_argument("-n", "--runs", type=int, default=8)
    parser.add_argument("--n-predict", type=int, default=200)
    parser.add_argument("-v", "--verbose", action="store_true", help="in mọi mẫu")
    args = parser.parse_args()

    reports: list[ModelReport] = []
    for raw in args.model:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            print(f"Không tìm thấy: {path}", file=sys.stderr)
            return 2
        print(f">>> {path.name}", flush=True)
        reports.append(asyncio.run(evaluate(path, args.runs, args.n_predict)))

    report_text = render(reports, args.verbose)
    print("\n" + report_text)

    out = REPO_ROOT / "benchmarks" / "results" / "model_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"environment": environment(),
             "reports": [{**r.__dict__} for r in reports]},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Chi tiết: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
