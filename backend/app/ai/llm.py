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

from .direction import Direction

logger = logging.getLogger(__name__)

#: Ngôn ngữ đích, viết bằng tiếng Anh cho model hiểu. Chỉ để dựng prompt.
_LANGUAGE_NAMES = {
    "vi": "Vietnamese", "en": "English", "ja": "Japanese", "zh": "Chinese",
    "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish",
    "th": "Thai", "id": "Indonesian", "ru": "Russian",
}


def language_name(code: str | None) -> str:
    if not code:
        return "English"
    return _LANGUAGE_NAMES.get(code.strip().lower().split("-")[0], code)


_TO_USER = """You are a live interpreter for a user wearing earbuds.
Someone is speaking TO the user in a foreign language. Translate what they said
into {target}.

Output ONE compact JSON object and nothing else:
{{"translation":"..."}}

Accuracy comes first. A translation that reads beautifully but loses or
changes what they said is a failure.

Rules, in order of priority:
1. Carry over EVERY element: each clause, every noun, number, date, name,
   negation ("not", "won't"), and hedge ("probably", "might", "about"). If
   they said two sentences, produce two sentences.
2. Do not substitute, generalise, or invent. Translate the noun they actually
   used — not a related one you think fits better. If you are unsure of a term,
   render it plainly rather than guessing a fancier word.
3. Leave nothing in the source language. Every word must be {target}, with
   correct spelling and diacritics, and no characters from a script {target}
   does not use.
4. Keep WHO DOES WHAT exactly as they said it. Never swap who is asking and
   who is offering.
5. Read times, dates and numbers the way a person means them out loud: "by
   three" is three o'clock, not the third day; "this afternoon" is the
   afternoon, not the evening.
6. Within those constraints, make it sound like natural spoken {target} rather
   than written prose.
- Translate only. Do not answer, explain, or add commentary.{history}"""

_TO_COUNTERPART = """You are a live interpreter for a user wearing earbuds.
The USER just spoke in {source}. The person they are talking to does not
understand {source}. Translate what the user said into {target}, so the user
can say it out loud.

Output ONE compact JSON object and nothing else:
{{"translation":"..."}}

Accuracy comes first. The user will say your translation out loud and be held
to it, so it must carry exactly what they meant — no more, no less.

Rules, in order of priority:
1. Carry over EVERY element: each clause, every noun, number, date, name,
   negation, and hedge. Do not compress two clauses into one.
2. Do not add ideas the user did not say, and do not soften or strengthen what
   they said.
3. Keep WHO DOES WHAT exactly as the user said it. If the user is ASKING the
   other person to do something, the translation must ask them too — never turn
   a request into an offer, and never swap who helps whom.
4. {source} pronouns like "anh", "em", "chị" mark politeness and who is
   speaking to whom. Carry that relationship over; do not invert it.
5. Write it entirely in {target}, and read times the way they were meant:
   "chiều nay" is this afternoon, not tonight.
6. Within those constraints, make it sound like something a person would
   actually say out loud — not stiff, not written prose.

NEVER copy the {source} sentence into the output. Even if the sentence is long
or hard, produce {target}. Repeating the input unchanged is always wrong.
- Translate only. Do not answer, explain, or add commentary.{history}"""

_HISTORY_RULE = """

=== CONTEXT ONLY — DO NOT TRANSLATE ANY LINE BELOW ===
Earlier turns, oldest first. "Them" is the other person, "You" is the user.
Use these only to resolve pronouns like "it" or "that one" and to keep the
thread.

{history}
=== END OF CONTEXT ==="""


def system_prompt(
    direction: Direction = Direction.TO_USER,
    *,
    user_language: str = "vi",
    counterpart_language: str | None = "en",
    history: str = "",
    retry_hint: str = "",
) -> str:
    """Prompt hệ thống theo chiều dịch.

    Output rút gọn còn đúng một trường `translation`. Gợi ý trả lời đã được BỎ:
    sản phẩm đổi mô hình — thay vì máy nghĩ hộ câu trả lời, người dùng tự nói
    bằng tiếng Việt và máy dịch sang tiếng đối phương để họ nói theo.

    Hệ quả tốt cho latency: output từ ~110 token xuống ~25 token.
    """
    context = _HISTORY_RULE.format(history=history.strip()) if history.strip() else ""
    if direction is Direction.TO_COUNTERPART:
        return _TO_COUNTERPART.format(
            source=language_name(user_language),
            target=language_name(counterpart_language),
            history=context,
        )
    return _TO_USER.format(target=language_name(user_language), history=context)


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

# Qwen3 / Qwen3.5: ChatML, nhưng có chế độ "thinking".
#
# Chat template chính thức tắt thinking bằng cách ĐIỀN SẴN một khối rỗng
# `<think>\n\n</think>\n\n` ngay đầu lượt assistant. Nếu không điền, model tự
# sinh `<think>` và suy nghĩ dài dòng — với dịch câu ngắn thì đó là latency
# thuần túy, không được gì.
#
# Grammar GBNF cũng chặn được thinking (nó ép ký tự đầu là `{`), nhưng lúc đó
# model đang bị bẻ ngược ý định của nó. Điền sẵn khối rỗng thì model biết mình
# đang ở chế độ không suy nghĩ và đi thẳng vào việc.
QWEN3 = PromptTemplate(
    name="qwen3",
    stop=("<|im_end|>", "<|im_start|>"),
    system_in_user_turn=False,
    _turn=(
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
)

TEMPLATES: dict[str, PromptTemplate] = {t.name: t for t in (CHATML, GEMMA, QWEN3)}

#: Nhận diện họ model từ tên file GGUF khi `llm.prompt_template = "auto"`.
#: Thứ tự QUAN TRỌNG: "qwen3" phải đứng trước "qwen", nếu không mọi model Qwen
#: đều rơi vào ChatML và Qwen3 sẽ suy nghĩ lan man.
#:
#: Lưu ý: các bản `-instruct-2507` của Qwen3 KHÔNG có thinking — với chúng nên
#: đặt `llm.prompt_template: chatml` tường minh.
_FILENAME_HINTS = (
    ("gemma", GEMMA),
    ("qwen3", QWEN3),
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


#: Grammar GBNF ràng buộc output. Dùng thay `json_schema` vì JSON Schema KHÔNG
#: kiểm soát được nội dung BÊN TRONG chuỗi.
#:
#: `{`, `}`, `[`, `]` và dấu ngoặc kép cong là ký tự hợp lệ trong chuỗi JSON,
#: nên json_schema cho phép model viết `”}` giữa chuỗi rồi lảm nhảm tiếp. Nó
#: "tưởng" đã đóng JSON trong khi grammar thì chưa.
#:
#: Đo thật (Gemma 3 4B, prompt CÓ lịch sử hội thoại, 6 câu):
#:      json_schema   0/6 sạch
#:      GBNF          6/6 sạch
#:
#: Không có lịch sử thì json_schema cũng sạch — lịch sử chứa nhiều chuỗi trong
#: ngoặc kép nên nó mồi cho model sinh đúng cái mẫu gây lỗi. Test không có lịch
#: sử sẽ cho kết quả sạch GIẢ.
RESPONSE_GRAMMAR = r"""
root   ::= "{" ws "\"translation\"" ws ":" ws string ws "}"
string ::= "\"" char* "\""
char   ::= [^"\\{}\[\]\u201C\u201D] | "\\" ["\\/bfnrt]
ws     ::= [ \t\n]*
"""


def response_grammar() -> str:
    """Grammar cho output. Xem chú thích ở RESPONSE_GRAMMAR."""
    return RESPONSE_GRAMMAR


def build_prompt(
    text: str,
    language: str | None,
    template: PromptTemplate = CHATML,
    *,
    direction: Direction = Direction.TO_USER,
    user_language: str = "vi",
    counterpart_language: str | None = "en",
    history: str = "",
        retry_hint: str = "",
) -> str:
    """Dựng prompt. Thứ tự các phần quyết định hiệu quả prefix cache:

        [system prompt + lịch sử]        [câu hiện tại]
         chỉ mọc thêm ở cuối              đổi mỗi lượt

    Lịch sử nằm cuối system prompt và chỉ nối thêm, nên tiền tố của lượt trước
    vẫn dùng lại được; chỉ phần mới cộng câu hiện tại phải xử lý lại.
    """
    system = system_prompt(
        direction,
        user_language=user_language,
        counterpart_language=counterpart_language,
        history=history,
            retry_hint=retry_hint,
    )
    speaker = "You" if direction is Direction.TO_COUNTERPART else "Them"
    # Phân định thật rõ câu cần dịch với khối lịch sử. Model 2B đã từng dịch
    # nhầm một dòng trong lịch sử thay vì câu hiện tại — với ranh giới mờ thì
    # model càng nhỏ càng dễ lạc.
    user = (
        f"TRANSLATE EXACTLY THIS ONE LINE, nothing else.\n"
        f"{speaker} said: {text}"
    )
    if retry_hint:
        # Lần dịch lại: lời nhắc nằm SAU câu cần dịch để nó là thứ model
        # đọc cuối cùng trước khi sinh.
        user = f"{user}\n\n{retry_hint}"
    return template.render(system, user)


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

    def build_prompt(
        self,
        text: str,
        language: str | None,
        *,
        direction: Direction = Direction.TO_USER,
        counterpart_language: str | None = None,
        history: str = "",
    ) -> str:
        """Prompt đúng định dạng của model đang nạp và đúng chiều dịch."""
        session = self._config.session
        return build_prompt(
            text,
            language,
            self.template,
            direction=direction,
            user_language=session.user_language,
            counterpart_language=counterpart_language or session.counterpart_language,
            history=history,
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
            "repeat_penalty": cfg.repeat_penalty,
            "n_predict": n_predict if n_predict is not None else cfg.n_predict,
            "stream": True,
            "cache_prompt": True,
            # Stop token theo họ model — dùng nhầm bộ của họ khác thì model
            # chạy tới hết n_predict và JSON bị cắt cụt.
            "stop": list(self.template.stop),
        }
        if cfg.grammar:
            # GBNF chứ không phải json_schema — xem RESPONSE_GRAMMAR.
            payload["grammar"] = response_grammar()

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
