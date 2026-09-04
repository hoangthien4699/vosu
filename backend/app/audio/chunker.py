"""Audio accumulation theo VAD (Task D2).

CONTRACT BẮT BUỘC (§2.2, review v4.1) — đây là chỗ dễ code sai nhất của dự án:

    `min_partial_window_s` (1.5s) CHỈ là ngưỡng tối thiểu để kích hoạt
    partial STT. Nó KHÔNG phải speech boundary.

    VAD endpoint MỚI là điều kiện bắt buộc để chạy final STT — áp dụng cho
    MỌI độ dài câu, kể cả câu ngắn hơn 1.5s ("Yes.", "Okay.").

Nếu chỉ dựa cứng vào "speech > 1.5s mới chạy Whisper", các utterance ngắn sẽ
không bao giờ được transcribe. Test `tests/test_chunker.py` khóa contract này.

Ràng buộc thứ hai (§2.2, review v3.0): không trượt cửa sổ quá dày. Trên GPU
6GB, transcribe lại 3-5 lần/giây sẽ đẩy GPU lên 100% và làm nghẽn hàng đợi
lệnh CUDA — ảnh hưởng trực tiếp latency của cả STT lẫn LLM. Vì vậy partial chỉ
được kích hoạt khi (a) đủ cửa sổ tối thiểu, VÀ (b) đã qua cooldown, VÀ
(c) chưa có partial nào cho lượng audio hiện tại.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .vad import VadEvent, VadEventType, VadProcessor

logger = logging.getLogger(__name__)


class SegmentKind(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"


# eq=False: field `pcm` là ndarray — so sánh mặc định sẽ chạy elementwise và
# ném ValueError ở bất kỳ chỗ nào dùng `==`/`in`/`.index()`. Dùng identity.
@dataclass(eq=False)
class AudioSegment:
    kind: SegmentKind
    pcm: np.ndarray
    sample_rate: int
    #: mốc bắt đầu đoạn speech trong stream (giây)
    start_s: float
    duration_s: float
    #: lý do kích hoạt — hữu ích khi debug và khi viết báo cáo benchmark
    trigger: str

    @property
    def is_final(self) -> bool:
        return self.kind is SegmentKind.FINAL


@dataclass
class _Accumulator:
    frames: list[np.ndarray] = field(default_factory=list)
    samples: int = 0
    start_s: float = 0.0
    last_partial_at_s: float = -1e9
    last_partial_samples: int = 0

    def add(self, pcm: np.ndarray) -> None:
        self.frames.append(pcm)
        self.samples += pcm.size

    def collect(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(0, dtype=np.float32)
        if len(self.frames) > 1:
            self.frames = [np.concatenate(self.frames)]
        return self.frames[0]

    def clear(self) -> None:
        self.frames.clear()
        self.samples = 0
        self.last_partial_samples = 0
        self.last_partial_at_s = -1e9


class AudioChunker:
    """Tích lũy PCM theo VAD và quyết định khi nào gọi STT.

    Dùng dạng generator: `for segment in chunker.feed(pcm_bytes): ...`
    """

    def __init__(
        self,
        vad: VadProcessor,
        *,
        min_partial_window_s: float = 1.5,
        partial_cooldown_s: float = 0.8,
        max_utterance_s: float = 30.0,
        enable_partial: bool = True,
        preroll_ms: int = 300,
        on_vad_event: Callable[[VadEvent], None] | None = None,
    ) -> None:
        self._vad = vad
        self.sample_rate = vad.sample_rate
        self.min_partial_window_s = min_partial_window_s
        self.partial_cooldown_s = partial_cooldown_s
        self.max_utterance_s = max_utterance_s
        self.enable_partial = enable_partial
        self._acc = _Accumulator()
        self._active = False
        self._stream_s = 0.0
        # Hook để tầng trên nhận speech_started ngay lập tức — Barge-in
        # (§2.4.1) phải phản ứng trong <200ms, không thể đợi tới khi có
        # segment audio hoàn chỉnh.
        self._on_vad_event = on_vad_event
        # VAD chỉ khẳng định speech sau `min_speech_ms`, và backdate mốc bắt đầu.
        # Không có pre-roll thì phần đầu từ đầu tiên đã bị vứt mất -> Whisper
        # nghe hụt phụ âm đầu.
        self._preroll_frames: deque[np.ndarray] = deque(
            maxlen=max(1, round(preroll_ms / (1000.0 * vad.frame_samples / vad.sample_rate)))
        )

    @property
    def is_active(self) -> bool:
        """Đang trong một utterance (đã speech_started, chưa speech_ended)."""
        return self._active

    def reset(self) -> None:
        self._vad.reset()
        self._acc.clear()
        self._active = False
        self._stream_s = 0.0
        self._preroll_frames.clear()

    # ------------------------------------------------------------------ #

    def feed(self, pcm: np.ndarray) -> Iterator[AudioSegment]:
        """Nạp PCM float32; sinh AudioSegment khi tới lúc gọi STT.

        Duyệt theo từng frame VAD để event luôn khớp đúng biên audio — một
        chunk lớn có thể chứa cả endpoint của câu này lẫn mở đầu của câu sau.
        """
        frame_s = self._vad.frame_samples / self.sample_rate

        for frame, events in self._vad.iter_frames(pcm):
            self._stream_s += frame_s

            for event in events:
                if self._on_vad_event is not None:
                    self._on_vad_event(event)
                if event.type is VadEventType.SPEECH_STARTED:
                    self._acc.clear()
                    self._acc.start_s = event.timestamp_s
                    # nạp lại pre-roll để không mất đầu câu
                    for buffered in self._preroll_frames:
                        self._acc.add(buffered)
                    self._active = True
                elif event.type is VadEventType.SPEECH_ENDED and self._active:
                    self._acc.add(frame)
                    yield from self._on_endpoint()
                    continue

            if self._active:
                self._acc.add(frame)
                segment = self._maybe_partial()
                if segment is not None:
                    yield segment
                if self._acc.samples / self.sample_rate >= self.max_utterance_s:
                    logger.warning(
                        "Utterance vượt %.1fs — cưỡng bức final STT.",
                        self.max_utterance_s,
                    )
                    yield self._emit_final(trigger="max_duration")
            else:
                self._preroll_frames.append(frame)

    @property
    def stream_s(self) -> float:
        """Đã nạp vào bao nhiêu giây AUDIO tính từ đầu phiên.

        Khác đồng hồ thật: lúc client dừng phát file thì đồng hồ này đứng yên.
        Mọi cửa sổ chờ liên quan tới lời nói phải đo bằng cái này, không đo
        bằng đồng hồ thật — nếu không thì client dừng file là cửa sổ tự hết
        giờ dù chưa nghe thêm được gì.
        """
        return self._stream_s

    def feed_bytes(self, pcm16: bytes) -> Iterator[AudioSegment]:
        from .vad import pcm16_to_float32

        return self.feed(pcm16_to_float32(pcm16))

    # ------------------------------------------------------------------ #

    def _on_endpoint(self) -> Iterator[AudioSegment]:
        """VAD endpoint → final STT BẮT BUỘC, bất kể độ dài câu."""
        if not self._active:
            return
        if self._acc.samples == 0:
            logger.debug("VAD endpoint nhưng chưa tích lũy được audio — bỏ qua.")
            self._active = False
            return
        yield self._emit_final(trigger="vad_endpoint")

    def _emit_final(self, trigger: str) -> AudioSegment:
        pcm = self._acc.collect()
        segment = AudioSegment(
            kind=SegmentKind.FINAL,
            pcm=pcm,
            sample_rate=self.sample_rate,
            start_s=self._acc.start_s,
            duration_s=pcm.size / self.sample_rate,
            trigger=trigger,
        )
        self._acc.clear()
        self._active = False
        return segment

    def _maybe_partial(self) -> AudioSegment | None:
        """Partial là TÙY CHỌN — chỉ để có phản hồi sớm cho câu dài."""
        if not self.enable_partial:
            return None

        window_s = self._acc.samples / self.sample_rate
        if window_s < self.min_partial_window_s:
            return None

        # (b) cooldown — chặn sliding-window quá dày
        since_last = self._stream_s - self._acc.last_partial_at_s
        if since_last < self.partial_cooldown_s:
            return None

        # (c) phải có audio MỚI so với partial trước, nếu không là transcribe lại y hệt
        if self._acc.samples <= self._acc.last_partial_samples:
            return None

        # Ưu tiên chạy partial khi probability đang suy giảm (sắp dứt câu) —
        # partial lúc đó gần với final nhất, giá trị hiển thị cao nhất.
        self._acc.last_partial_at_s = self._stream_s
        self._acc.last_partial_samples = self._acc.samples
        return AudioSegment(
            kind=SegmentKind.PARTIAL,
            pcm=self._acc.collect().copy(),
            sample_rate=self.sample_rate,
            start_s=self._acc.start_s,
            duration_s=window_s,
            trigger="decay" if self._vad.is_decaying else "min_window",
        )
