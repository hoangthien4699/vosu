"""So sánh chất lượng dịch giữa các model — công cụ, KHÔNG phải mục của Gate.

    python -m benchmarks.compare_models \\
        --model models/gemma-3-4b-it-q4_k_m.gguf \\
        --model models/qwen3-4b-instruct-2507-q4_k_m.gguf

Đo cả HAI chiều trên chính prompt của dự án. Mỗi model khởi động riêng, lần
lượt, để không tranh chấp GPU/VRAM.

Cái đo được bằng máy thì đo; cái không thì in ra để người đọc tự đánh giá.
"Nghe có tự nhiên không" chỉ người nói tiếng đó mới phán được.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .common import REPO_ROOT, Distribution, environment

# Tiếng Việt dùng chữ Latin + dấu. Ký tự CJK/Kana/Hangul trong bản dịch tiếng
# Việt luôn là lỗi rò ngôn ngữ.
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
#: Dấu phụ tiếng Việt — dùng để đoán một chuỗi có phải tiếng Việt không.
_VI_MARKS = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    r"ùúủũụưừứửữựỳýỷỹỵđ]", re.I
)
#: Rác cấu trúc lọt vào giá trị chuỗi.
_STRUCT = set('{}[]“”')

#: Hội thoại thật, xen kẽ hai chiều. Câu tiếng Việt cố ý có cách nói đời thường
#: ("chốt", "e là", "nhé") để lộ ra model nào dịch cứng.
CONVERSATION = [
    ("en", "We're thinking about pushing the launch to next quarter."),
    ("vi", "Tôi lo là khách hàng sẽ không hài lòng nếu chậm thêm."),
    ("en", "That's a fair concern. How much slack do we have?"),
    ("vi", "Khoảng hai tuần thôi, sau đó là hết hạn hợp đồng."),
    ("en", "Then let's lock the scope today and start Monday."),
    ("vi", "Được, nhưng anh chốt giúp em phạm vi trước chiều nay nhé."),
    ("en", "I'll send the final scope by three. Does that work?"),
    ("vi", "Vâng ạ, thế thì em kịp chuẩn bị cho họp sáng mai."),
]


@dataclass
class Sample:
    source_lang: str
    source: str
    translation: str = ""
    outbound: bool = False
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens: int = 0
    error: str = ""

    @property
    def cjk_leak(self) -> str:
        return "".join(_CJK.findall(self.translation))

    @property
    def struct_noise(self) -> str:
        return "".join(c for c in self.translation if c in _STRUCT)

    @property
    def wrong_language(self) -> bool:
        """Bản dịch có đúng ngôn ngữ đích không (đoán thô bằng dấu tiếng Việt)."""
        if not self.translation.strip():
            return False
        looks_vietnamese = bool(_VI_MARKS.search(self.translation))
        # chiều ngược -> đích là tiếng Anh -> KHÔNG được có dấu tiếng Việt
        return looks_vietnamese if self.outbound else not looks_vietnamese

    @property
    def length_ratio(self) -> float:
        """Tỷ lệ độ dài so với nguồn. Quá thấp = dịch thiếu vế."""
        if not self.source:
            return 0.0
        return len(self.translation) / len(self.source)


@dataclass
class Report:
    path: str
    template: str = ""
    samples: list[Sample] = field(default_factory=list)

    def count(self, predicate) -> int:
        return sum(1 for s in self.samples if predicate(s))


async def evaluate(model_path: Path, runs: int) -> Report:
    from app.ai.copilot import SemanticEventParser
    from app.ai.direction import Direction
    from app.ai.history import ConversationHistory
    from app.ai.llm import LlmClient
    from app.core.config import load_config
    from app.core.vram_manager import LlamaServerManager

    config = load_config()
    config.paths.llm_gguf = str(model_path)
    config.llm.prompt_template = "auto"

    report = Report(path=model_path.name)
    manager = LlamaServerManager(config)
    client = LlmClient(config)
    report.template = client.template.name

    await manager.start()
    await client.start()
    history = ConversationHistory(
        max_turns=config.llm.history_turns, max_chars=config.llm.history_chars
    )
    try:
        # warm-up: lần đầu luôn trả giá xử lý toàn bộ system prompt
        await client.complete(client.build_prompt("Hello.", "en"))

        turns = (CONVERSATION * ((runs // len(CONVERSATION)) + 1))[:runs]
        for index, (lang, text) in enumerate(turns):
            outbound = lang == config.session.user_language
            direction = (
                Direction.TO_COUNTERPART if outbound else Direction.TO_USER
            )
            uid = f"u{index}"
            history.add(uid, text, lang, is_user=outbound)

            sample = Sample(source_lang=lang, source=text, outbound=outbound)
            started = time.perf_counter()
            try:
                raw, stats = await client.complete(
                    client.build_prompt(
                        text, lang,
                        direction=direction,
                        counterpart_language="en",
                        history=history.render(exclude=uid),
                    )
                )
            except Exception as exc:
                sample.error = f"{type(exc).__name__}: {exc}"[:120]
                report.samples.append(sample)
                continue

            parser = SemanticEventParser()
            parser.feed(raw)
            parser.finish()
            sample.translation = parser.result.translation
            sample.ttft_ms = stats.ttft_ms or 0.0
            sample.total_ms = stats.total_ms or (time.perf_counter() - started) * 1000
            sample.tokens = stats.tokens
            history.set_translation(uid, sample.translation)
            report.samples.append(sample)
    finally:
        await client.close()
        await manager.stop()

    return report


def render(reports: list[Report], verbose: bool) -> str:
    env = environment()
    lines = ["=" * 78,
             "SO SÁNH MODEL — dịch hai chiều, trên chính prompt của dự án",
             "=" * 78,
             f"  {env['platform']} · {env['gpu']}", ""]

    header = (f"  {'model':<38} {'tmpl':<7} {'sai ngôn ngữ':<13} "
              f"{'rác':<6} {'TTFT P50':<10} {'tổng P50'}")
    lines += [header, "  " + "-" * (len(header) - 2)]
    for r in reports:
        n = len(r.samples)
        ttft = Distribution.of([s.ttft_ms for s in r.samples if s.ttft_ms])
        total = Distribution.of([s.total_ms for s in r.samples if s.total_ms])
        lines.append(
            f"  {r.path:<38} {r.template:<7} "
            f"{r.count(lambda s: s.wrong_language)}/{n:<11} "
            f"{r.count(lambda s: s.cjk_leak or s.struct_noise)}/{n:<4} "
            f"{ttft.p50:>7.0f}ms  {total.p50:>7.0f}ms"
        )
        errors = r.count(lambda s: bool(s.error))
        short = r.count(lambda s: 0 < s.length_ratio < 0.55)
        if errors or short:
            lines.append(f"    (lỗi {errors} · nghi dịch thiếu {short})")
    lines.append("")
    lines.append("  `sai ngôn ngữ` và `rác` đo được bằng máy. Còn CÓ TỰ NHIÊN KHÔNG")
    lines.append("  thì phải tự đọc phần dưới — chỉ người nói tiếng đó mới phán được.")
    lines.append("")

    for r in reports:
        lines += ["=" * 78, r.path, "=" * 78]
        for s in r.samples if verbose else r.samples[:8]:
            if s.error:
                lines.append(f"  LỖI: {s.error}")
                continue
            who = "BẠN" if s.outbound else "HỌ "
            flags = []
            if s.wrong_language:
                flags.append("SAI NGÔN NGỮ")
            if s.cjk_leak:
                flags.append(f"rò CJK {s.cjk_leak!r}")
            if s.struct_noise:
                flags.append(f"rác {s.struct_noise!r}")
            if 0 < s.length_ratio < 0.55:
                flags.append("nghi dịch thiếu")
            mark = ("   <<< " + ", ".join(flags)) if flags else ""
            lines.append(f"  {who} [{s.source_lang}] {s.source}")
            lines.append(f"       -> {s.translation}{mark}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", action="append", required=True, metavar="PATH")
    parser.add_argument("-n", "--runs", type=int, default=len(CONVERSATION))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    reports: list[Report] = []
    for raw in args.model:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            print(f"Không tìm thấy: {path}", file=sys.stderr)
            return 2
        print(f">>> {path.name}", flush=True)
        reports.append(asyncio.run(evaluate(path, args.runs)))

    print("\n" + render(reports, args.verbose))

    out = REPO_ROOT / "benchmarks" / "results" / "model_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"environment": environment(),
             "reports": [{"path": r.path, "template": r.template,
                          "samples": [s.__dict__ for s in r.samples]} for r in reports]},
            indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Chi tiết: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
