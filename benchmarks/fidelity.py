"""Đo độ trung thực của bản dịch — bao nhiêu yếu tố bắt buộc còn sống sót.

    python -m benchmarks.fidelity
    python -m benchmarks.fidelity --model models/gemma-3-4b-it-q4_k_m.gguf

Không thay `compare_models` (so model với model) mà bổ sung cho nó: cái này đo
MỘT cấu hình, dùng để biết một thay đổi prompt/tham số có làm bản dịch kĩ hơn
hay không.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .common import REPO_ROOT, Distribution, environment
from .fidelity_cases import ALL, missing


async def run(model: Path | None, temperature: float | None, n_predict: int | None):
    from app.ai.copilot import SemanticEventParser
    from app.ai.direction import Direction
    from app.ai.llm import LlmClient, language_name
    from app.ai.verify import failure_reason, retry_hint
    from app.core.config import load_config
    from app.core.vram_manager import LlamaServerManager

    config = load_config()
    if model is not None:
        config.paths.llm_gguf = str(model)
    if temperature is not None:
        config.llm.temperature = temperature
    if n_predict is not None:
        config.llm.n_predict = n_predict

    manager = LlamaServerManager(config)
    client = LlmClient(config)
    await manager.start()
    await client.start()

    rows, latencies, retried = [], [], 0
    try:
        await client.complete(client.build_prompt("Hello.", "en"))  # warm-up
        for case in ALL:
            outbound = case.lang == config.session.user_language
            direction = Direction.TO_COUNTERPART if outbound else Direction.TO_USER
            raw, stats = await client.complete(
                client.build_prompt(
                    case.text, case.lang,
                    direction=direction, counterpart_language="en",
                )
            )
            parser = SemanticEventParser()
            parser.feed(raw)
            parser.finish()
            translation = parser.result.translation
            total_ms = stats.total_ms

            # Lưới an toàn giống hệt pipeline thật — không có nó thì công cụ
            # đo một hệ thống khác với hệ thống đang chạy.
            target = "en" if outbound else config.session.user_language
            reason = (
                failure_reason(case.text, translation, target)
                if config.llm.retry_on_bad_translation
                else None
            )
            if reason is not None:
                retried += 1
                raw, stats = await client.complete(
                    client.build_prompt(
                        case.text, case.lang,
                        direction=direction, counterpart_language="en",
                        retry_hint=retry_hint(reason, language_name(target)),
                    )
                )
                parser = SemanticEventParser()
                parser.feed(raw)
                parser.finish()
                translation = parser.result.translation
                total_ms += stats.total_ms

            rows.append((case, translation, missing(case, translation)))
            latencies.append(total_ms)
    finally:
        await client.close()
        await manager.stop()

    return config, rows, latencies, retried


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--n-predict", type=int, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="chỉ in tổng kết, không in từng câu")
    args = parser.parse_args()

    model = args.model
    if model is not None and not model.is_absolute():
        model = REPO_ROOT / model
    if model is not None and not model.exists():
        print(f"Không tìm thấy: {model}", file=sys.stderr)
        return 2

    config, rows, latencies, retried = asyncio.run(
        run(model, args.temperature, args.n_predict)
    )

    total_elements = sum(len(c.must_keep) for c, _, _ in rows)
    lost = sum(len(m) for _, _, m in rows)
    bad_cases = sum(1 for _, _, m in rows if m)

    env = environment()
    label = args.label or Path(config.paths.llm_gguf).name
    print("=" * 78)
    print(f"ĐỘ TRUNG THỰC BẢN DỊCH — {label}")
    print("=" * 78)
    print(f"  {env['platform']} · temp={config.llm.temperature} "
          f"n_predict={config.llm.n_predict}")
    print()

    if not args.quiet:
        for case, translation, gone in rows:
            flag = "  <<< THIẾU: " + " | ".join("/".join(g) for g in gone) if gone else ""
            who = "BẠN" if case.lang == config.session.user_language else "HỌ "
            print(f"  {who} {case.text}")
            print(f"       -> {translation}{flag}")
        print()

    kept = total_elements - lost
    print(f"  yếu tố giữ được : {kept}/{total_elements} ({kept/total_elements*100:.0f}%)")
    print(f"  câu bị thiếu ý  : {bad_cases}/{len(rows)}")
    print(f"  phải dịch lại   : {retried}/{len(rows)}")
    print(f"  thời gian sinh  : {Distribution.of(latencies).summary()}")

    out = REPO_ROOT / "benchmarks" / "results" / "fidelity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"environment": env, "label": label,
               "temperature": config.llm.temperature, "n_predict": config.llm.n_predict,
               "kept": kept, "total": total_elements, "bad_cases": bad_cases,
               "retried": retried,
               "rows": [{"source": c.text, "lang": c.lang, "translation": t,
                         "missing": m, "note": c.note} for c, t, m in rows]}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Chi tiết: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
