"""Chọn engine TTS cho từng lượt đọc.

Có hai engine vì không cái nào làm được cả hai việc:

    Piper   giọng máy móc, nhưng ĐỔI ĐƯỢC TỐC ĐỘ ĐỌC
    VieNeu  giọng truyền cảm, 20 giọng, nhưng KHÔNG có tham số tốc độ

Chiều dịch ngược đọc chậm (`coach_length_scale`, mặc định 1.35) để người dùng
nói theo — đó là yêu cầu có thật của sản phẩm, không phải tùy chọn trang trí.
Nên lượt nào cần đọc chậm thì đi Piper, còn lại đi engine đã cấu hình.

Router giữ nguyên bề mặt của một engine để chỗ gọi không phải biết gì.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable

from .tts import CancelResult, PiperTts, TtsJob, TtsState

logger = logging.getLogger(__name__)


def create_tts(config):
    """Engine theo config. Trả về `PiperTts` trần nếu không cần định tuyến.

    Chọn `vieneu` mà chưa dựng venv riêng thì LÙI VỀ PIPER kèm cảnh báo, chứ
    không tắt hẳn TTS: người dùng vẫn nghe được, chỉ là giọng cũ. Tắt hẳn thì
    họ mất tiếng mà không hiểu vì sao.
    """
    if config.tts.engine != "vieneu":
        return PiperTts(config)

    from .tts_vieneu import VieNeuTts

    primary = VieNeuTts(config)
    try:
        primary.preflight()
    except Exception as exc:
        logger.warning("Chưa dùng được VieNeu-TTS (%s) — quay lại Piper", exc)
        return PiperTts(config)
    return TtsRouter(config, primary=primary, fallback=PiperTts(config))


class TtsRouter:
    supports_length_scale = True

    @property
    def needs_preload(self) -> bool:
        return self._primary.needs_preload

    def __init__(self, config, *, primary, fallback) -> None:
        self._config = config
        self._primary = primary
        self._fallback = fallback
        self._active = primary

    # -- chọn engine ------------------------------------------------------ #

    def _pick(self, length_scale: float | None):
        wants_speed = length_scale is not None and abs(length_scale - 1.0) > 0.01
        if wants_speed and not self._primary.supports_length_scale:
            return self._fallback
        return self._primary

    # -- bề mặt engine ---------------------------------------------------- #

    @property
    def sample_rate(self) -> int:
        return self._active.sample_rate

    @property
    def state(self) -> TtsState:
        return self._active.state

    @property
    def current_job(self) -> TtsJob | None:
        return self._active.current_job

    @property
    def is_active(self) -> bool:
        return self._primary.is_active or self._fallback.is_active

    @property
    def used_standby(self) -> bool:
        return self._active.used_standby

    def preflight(self, voice: str | None = None) -> None:
        self._primary.preflight(voice)

    def resolve_voice(self, voice: str | None = None):
        return self._active.resolve_voice(voice)

    def prewarm(self, voice: str | None = None, length_scale: float | None = None) -> None:
        self._pick(length_scale).prewarm(voice, length_scale)

    def synthesize(
        self,
        utterance_id: str,
        text: str,
        *,
        field: str = "translation",
        voice: str | None = None,
        length_scale: float | None = None,
        on_chunk: Callable[[bytes, int], None] | None = None,
    ) -> AsyncIterator[bytes]:
        engine = self._pick(length_scale)
        if engine is not self._active:
            logger.info(
                "TTS đổi engine -> %s (%s)",
                type(engine).__name__,
                "cần đọc chậm" if engine is self._fallback else "mặc định",
            )
        self._active = engine
        return engine.synthesize(
            utterance_id, text, field=field, voice=voice,
            length_scale=length_scale, on_chunk=on_chunk,
        )

    async def cancel(self, reason: str = "barge_in") -> CancelResult:
        # Hủy CẢ HAI: engine vừa đổi thì cái cũ có thể còn đang đọc dở.
        first = await self._primary.cancel(reason)
        second = await self._fallback.cancel(reason)
        return first if first.cancelled else second

    async def close(self) -> None:
        await self._primary.close()
        await self._fallback.close()
