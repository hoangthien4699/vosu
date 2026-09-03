"""Utterance State Machine (Task D3).

Đặc tả §4.3: một session chứa NHIỀU utterance. Khi utterance B xuất hiện trong
lúc TTS của utterance A còn đang phát, hệ thống phải biết chính xác cancel TTS
của utterance NÀO — nếu không, việc quản lý output bất đồng bộ (đặc biệt với
Barge-in §2.4.1) sẽ nhanh chóng rối loạn. Đó là lý do `utterance_id` tồn tại.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class UtteranceState(str, Enum):
    LISTENING = "listening"          # VAD đã báo speech_started
    TRANSCRIBING = "transcribing"    # đang chạy final STT (sau VAD endpoint)
    COPILOT = "copilot"              # LLM đang sinh
    SPEAKING = "speaking"            # TTS đang synthesize/phát
    DONE = "done"
    CANCELLED = "cancelled"          # bị Barge-in hoặc utterance mới chiếm chỗ
    FAILED = "failed"


TERMINAL_STATES = frozenset(
    {UtteranceState.DONE, UtteranceState.CANCELLED, UtteranceState.FAILED}
)

#: Transition hợp lệ. Mọi chuyển trạng thái ngoài bảng này bị từ chối.
_ALLOWED: dict[UtteranceState, frozenset[UtteranceState]] = {
    UtteranceState.LISTENING: frozenset(
        {UtteranceState.TRANSCRIBING, UtteranceState.CANCELLED, UtteranceState.FAILED}
    ),
    UtteranceState.TRANSCRIBING: frozenset(
        {
            UtteranceState.COPILOT,
            UtteranceState.DONE,      # STT ra text rỗng -> kết thúc sớm, hợp lệ
            UtteranceState.CANCELLED,
            UtteranceState.FAILED,
        }
    ),
    UtteranceState.COPILOT: frozenset(
        {
            UtteranceState.SPEAKING,
            UtteranceState.DONE,      # TTS tắt hoặc không có gì để đọc
            UtteranceState.CANCELLED,
            UtteranceState.FAILED,
        }
    ),
    UtteranceState.SPEAKING: frozenset(
        {UtteranceState.DONE, UtteranceState.CANCELLED, UtteranceState.FAILED}
    ),
}


class InvalidTransition(RuntimeError):
    pass


@dataclass
class Utterance:
    id: str
    state: UtteranceState = UtteranceState.LISTENING
    created_at: float = field(default_factory=time.monotonic)

    partial_text: str = ""
    final_text: str = ""
    language: str | None = None

    #: mốc VAD endpoint — gốc thời gian để đo E2E "first useful result" (§7)
    endpoint_at: float | None = None
    first_useful_at: float | None = None
    history: list[tuple[UtteranceState, float]] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def e2e_ms(self) -> float | None:
        """speech endpoint -> first useful copilot result, tính bằng ms.

        Đây là metric E2E ĐÚNG theo §7 (review v4.1) — không phải tổng cộng dồn
        thời gian xử lý riêng lẻ của từng thành phần.
        """
        if self.endpoint_at is None or self.first_useful_at is None:
            return None
        return (self.first_useful_at - self.endpoint_at) * 1000.0

    def mark_endpoint(self) -> None:
        if self.endpoint_at is None:
            self.endpoint_at = time.monotonic()

    def mark_first_useful(self) -> None:
        if self.first_useful_at is None:
            self.first_useful_at = time.monotonic()


class SessionState:
    """Quản lý toàn bộ utterance trong MỘT session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._counter = itertools.count(1)
        self._utterances: dict[str, Utterance] = {}
        self._current_id: str | None = None
        self.started_at = time.monotonic()

    # -- truy vấn --------------------------------------------------------- #

    @property
    def current(self) -> Utterance | None:
        if self._current_id is None:
            return None
        return self._utterances.get(self._current_id)

    @property
    def total_utterances(self) -> int:
        return len(self._utterances)

    def get(self, utterance_id: str) -> Utterance | None:
        return self._utterances.get(utterance_id)

    def active(self) -> list[Utterance]:
        return [u for u in self._utterances.values() if not u.is_terminal]

    # -- thay đổi trạng thái ---------------------------------------------- #

    def begin_utterance(self) -> Utterance:
        """Mở utterance mới. Utterance đang chạy (nếu có) bị đánh dấu CANCELLED.

        Một người chỉ nói một câu tại một thời điểm — utterance mới xuất hiện
        nghĩa là câu trước đã bị chiếm chỗ. Việc hủy TTS tương ứng do
        `ai/tts.py` xử lý qua tín hiệu Barge-in.
        """
        previous = self.current
        if previous is not None and not previous.is_terminal:
            logger.debug(
                "Utterance %s bị chiếm chỗ ở trạng thái %s",
                previous.id,
                previous.state.value,
            )
            self.transition(previous.id, UtteranceState.CANCELLED)

        utterance = Utterance(id=f"utt_{next(self._counter):03d}")
        utterance.history.append((UtteranceState.LISTENING, time.monotonic()))
        self._utterances[utterance.id] = utterance
        self._current_id = utterance.id
        return utterance

    def transition(self, utterance_id: str, target: UtteranceState) -> Utterance:
        utterance = self._utterances.get(utterance_id)
        if utterance is None:
            raise KeyError(f"không có utterance {utterance_id!r}")

        if utterance.state is target:
            return utterance

        allowed = _ALLOWED.get(utterance.state, frozenset())
        if target not in allowed:
            raise InvalidTransition(
                f"{utterance_id}: {utterance.state.value} -> {target.value} không hợp lệ "
                f"(cho phép: {sorted(s.value for s in allowed) or 'không có — trạng thái kết thúc'})"
            )

        utterance.state = target
        utterance.history.append((target, time.monotonic()))
        return utterance

    def cancel_all_active(self, exclude: str | None = None) -> list[Utterance]:
        cancelled = []
        for utterance in self.active():
            if utterance.id == exclude:
                continue
            self.transition(utterance.id, UtteranceState.CANCELLED)
            cancelled.append(utterance)
        return cancelled

    def completed_e2e_ms(self) -> list[float]:
        """Danh sách E2E đã đo được — dùng để báo cáo percentile (§7)."""
        return [u.e2e_ms for u in self._utterances.values() if u.e2e_ms is not None]
