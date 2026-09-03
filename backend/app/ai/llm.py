"""Wrapper `llama-server` (Task E2).

Đặc tả §4.4: PHẢI streaming, không batch request. Cách gọi ở baseline v1
(`await http_client.post(...)` rồi chờ `res.json()`) khiến client không thấy gì
cho tới khi LLM sinh xong toàn bộ JSON.

Module này chỉ lo token stream thô + đo TTFT. Việc biến token thành semantic
event là của `ai/copilot.py` — hai abstraction khác nhau (§4.4).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_BASE_RULES = """You are a real-time copilot for a user wearing earbuds.
Someone is speaking TO the user in a foreign language. Help the user understand
what was said, and give the user something to say back.

Output ONE compact JSON object and nothing else. No markdown, no code fence.
{schema}

LANGUAGE RULES — these matter more than anything else:
- "translation": the speech rendered in VIETNAMESE. Every word Vietnamese with
  proper diacritics. Never leave words in the source language. Never use
  Chinese characters.
- "intent": what the speaker actually wants, in VIETNAMESE, under 10 words.
- reply text: what the USER SAYS BACK to the speaker. The speaker does not
  understand Vietnamese, so it MUST be in the SAME LANGUAGE THE SPEAKER USED —
  English speech gets English replies, Japanese gets Japanese. Vietnamese ONLY
  if the speaker spoke Vietnamese.
  Exactly 2 replies, each under 15 words, meaningfully different.
{purpose_rule}
Example — the speaker said, in English: "We need more time."
{example}

Output the JSON immediately. Do not explain."""

_PLAIN_SCHEMA = '{"translation":"...","intent":"...","replies":["...","..."]}'
_PLAIN_EXAMPLE = (
    '{"translation":"Chúng tôi cần thêm thời gian.",'
    '"intent":"Muốn xin gia hạn thêm thời gian.",'
    '"replies":["How much more time do you need?",'
    '"That\'s fine, take the time you need."]}'
)

_PURPOSE_SCHEMA = (
    '{"translation":"...","intent":"...",'
    '"replies":[{"text":"...","purpose":"..."},{"text":"...","purpose":"..."}]}'
)
_PURPOSE_RULE = """- "purpose": why the user would pick THIS reply — what it achieves in the
  conversation. ALWAYS VIETNAMESE, under 12 words. Write "purpose" in
  Vietnamese even though the reply next to it is English or Japanese — the
  reply is for the speaker, the purpose is for the user. Two replies must have
  clearly different purposes, not two wordings of the same move.
"""
_PURPOSE_EXAMPLE = (
    '{"translation":"Chúng tôi cần thêm thời gian.",'
    '"intent":"Muốn xin gia hạn thêm thời gian.",'
    '"replies":[{"text":"How much more time do you need?",'
    '"purpose":"Ép đối phương chốt một mốc cụ thể"},'
    '{"text":"That\'s fine, take the time you need.",'
    '"purpose":"Nhượng bộ để giữ quan hệ tốt"}]}'
)


def system_prompt(with_purpose: bool = True) -> str:
    """Prompt hệ thống. `with_purpose` thêm lý do nên chọn cho từng reply.

    §4.4 cố ý bỏ trường mô tả cho từng reply vì token thêm làm tăng latency.
    `purpose` khác `meaning` của v1 (bản dịch reply) — nó nói MỤC ĐÍCH khi chọn
    câu đó.

    Không miễn phí: prompt dài thêm ~360 ký tự nên first-useful-result tăng
    198ms -> 282ms và tổng thời gian sinh tăng 69% (đo trên Gemma 3 4B / M4).
    Tắt được qua `llm.reply_purpose`.
    """
    if with_purpose:
        return _BASE_RULES.format(
            schema=_PURPOSE_SCHEMA, purpose_rule=_PURPOSE_RULE, example=_PURPOSE_EXAMPLE
        )
    return _BASE_RULES.format(
        schema=_PLAIN_SCHEMA, purpose_rule="", example=_PLAIN_EXAMPLE
    )


#: Giữ tên cũ cho test và mã gọi sẵn có.
SYSTEM_PROMPT = system_prompt(with_purpose=False)


# --------------------------------------------------------------------------- #
# Prompt template theo họ model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PromptTemplate:
    """Cách gói system + user thành prompt thô cho `/completion`.

    Mỗi họ model có định dạng lượt hội thoại riêng, và dùng SAI định dạng thì
    model vẫn sinh ra chữ — chỉ là chất lượng tệ đi một cách khó truy vết, chứ
    không báo lỗi. Vì vậy template phải là dữ liệu tường minh, không hardcode.
    """

    name: str
    stop: tuple[str, ...]
    #: True nếu họ model không có vai trò `system` riêng — chỉ dẫn hệ thống
    #: phải gộp vào lượt user đầu tiên.
    system_in_user_turn: bool
    _turn: str

    def render(self, system: str, user: str) -> str:
        if self.system_in_user_turn:
            return self._turn.format(content=f"{system}\n\n{user}")
        return self._turn.format(system=system, user=user)


CHATML = PromptTemplate(
    name="chatml",
    stop=("<|im_end|>", "<|im_start|>"),
    system_in_user_turn=False,
    _turn=(
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
)

# Gemma 2/3: không có vai trò `system`, và KHÔNG tự thêm <bos> ở đây —
# llama-server đã tokenize prompt với add_special=true nên tự chèn BOS. Thêm
# tay nữa sẽ thành BOS kép, làm chất lượng tụt mà không có dấu hiệu gì.
GEMMA = PromptTemplate(
    name="gemma",
    stop=("<end_of_turn>", "<start_of_turn>"),
    system_in_user_turn=True,
    _turn="<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
)

TEMPLATES: dict[str, PromptTemplate] = {t.name: t for t in (CHATML, GEMMA)}

#: Nhận diện họ model từ tên file GGUF khi `llm.prompt_template = "auto"`.
_FILENAME_HINTS = (
    ("gemma", GEMMA),
    ("qwen", CHATML),
    ("chatml", CHATML),
)


def resolve_template(config) -> PromptTemplate:
    """Chọn template theo config; `auto` thì suy ra từ tên file GGUF."""
    requested = (config.llm.prompt_template or "auto").lower()
    if requested != "auto":
        template = TEMPLATES.get(requested)
        if template is None:
            valid = ", ".join(sorted(TEMPLATES))
            raise ValueError(
                f"llm.prompt_template không hợp lệ: {requested!r}. Hợp lệ: auto, {valid}"
            )
        return template

    name = str(config.paths.llm_gguf).lower()
    for hint, template in _FILENAME_HINTS:
        if hint in name:
            return template

    logger.warning(
        "Không suy được prompt template từ %r — mặc định ChatML. "
        "Đặt llm.prompt_template tường minh nếu model dùng định dạng khác.",
        config.paths.llm_gguf,
    )
    return CHATML


def build_prompt(
    text: str,
    language: str | None,
    template: PromptTemplate = CHATML,
    *,
    with_purpose: bool = False,
) -> str:
    lang = language or "unknown"
    user = f'Language: {lang}\nSpeech: "{text}"'
    return template.render(system_prompt(with_purpose), user)


@dataclass
class GenerationStats:
    ttft_ms: float | None = None
    total_ms: float = 0.0
    tokens: int = 0
    #: model dừng vì chạm n_predict thay vì sinh xong -> JSON có thể bị cụt
    truncated: bool = False
    stop_reason: str = ""
    raw: str = field(default="", repr=False)


class LlmClient:
    """Client streaming tới endpoint `/completion` của llama-server."""

    def __init__(self, config) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self.template = resolve_template(config)
        logger.info(
            "LLM prompt template: %s (stop: %s)",
            self.template.name,
            ", ".join(self.template.stop),
        )

    def build_prompt(self, text: str, language: str | None) -> str:
        """Prompt đúng định dạng của model đang nạp."""
        return build_prompt(
            text, language, self.template,
            with_purpose=self._config.llm.reply_purpose,
        )

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.llm.base_url,
                timeout=httpx.Timeout(self._config.llm.request_timeout_s, connect=5.0),
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LlmClient chưa start()")
        return self._client

    async def health(self) -> bool:
        try:
            response = await self._require_client().get("/health", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    # ------------------------------------------------------------------ #

    async def stream(
        self,
        prompt: str,
        *,
        stats: GenerationStats | None = None,
        n_predict: int | None = None,
    ) -> AsyncIterator[str]:
        """Sinh token. `stats` (nếu truyền vào) được cập nhật tại chỗ.

        TTFT được đo từ lúc gửi request tới token ĐẦU TIÊN — đây là con số
        quyết định perceived latency, không phải tổng thời gian sinh (§7).
        """
        cfg = self._config.llm
        stats = stats if stats is not None else GenerationStats()
        payload = {
            "prompt": prompt,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "n_predict": n_predict if n_predict is not None else cfg.n_predict,
            "stream": True,
            "cache_prompt": True,
            # Stop token theo họ model — dùng nhầm bộ của họ khác thì model
            # chạy tới hết n_predict và JSON bị cắt cụt.
            "stop": list(self.template.stop),
        }

        started = time.perf_counter()
        chunks: list[str] = []
        try:
            async with self._require_client().stream(
                "POST", "/completion", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    token = _parse_sse_line(line)
                    if token is None:
                        continue
                    content, stop, stop_reason = token
                    if content:
                        if stats.ttft_ms is None:
                            stats.ttft_ms = (time.perf_counter() - started) * 1000.0
                        stats.tokens += 1
                        chunks.append(content)
                        yield content
                    if stop:
                        stats.stop_reason = stop_reason
                        stats.truncated = stop_reason == "limit"
                        break
        finally:
            stats.total_ms = (time.perf_counter() - started) * 1000.0
            stats.raw = "".join(chunks)

    async def complete(self, prompt: str, **kwargs) -> tuple[str, GenerationStats]:
        """Gom toàn bộ output. Dùng cho benchmark và test, KHÔNG dùng trong pipeline."""
        stats = GenerationStats()
        parts = [chunk async for chunk in self.stream(prompt, stats=stats, **kwargs)]
        return "".join(parts), stats


def _parse_sse_line(line: str) -> tuple[str, bool, str] | None:
    """Trả về (content, stop, stop_reason) hoặc None nếu dòng không mang dữ liệu.

    llama-server phát Server-Sent Events: `data: {...}`. Dòng trống là ngăn cách
    giữa các event, không phải lỗi.
    """
    if not line or not line.startswith("data:"):
        return None
    body = line[5:].strip()
    if not body or body == "[DONE]":
        return ("", True, "done") if body == "[DONE]" else None
    try:
        message = json.loads(body)
    except json.JSONDecodeError:
        logger.debug("Bỏ qua dòng SSE không parse được: %r", body[:120])
        return None

    content = message.get("content", "")
    stop = bool(message.get("stop", False))
    reason = ""
    if stop:
        if message.get("stopped_limit"):
            reason = "limit"
        elif message.get("stopped_word") or message.get("stopped_eos"):
            reason = "stop_word"
        else:
            reason = "done"
    return content, stop, reason
