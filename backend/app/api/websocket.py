"""Endpoint `/ws/copilot` — điều phối toàn bộ pipeline (Task C4, F4, F5).

Sửa lỗi baseline v1 (§3.1, §6):
  - `acquire_hardware()` nằm TRONG `try`, không phải trước. Lỗi giữa chừng lúc
    acquire vẫn phải chạy `finally: release` — nếu không sẽ rò rỉ VRAM.
  - Không transcribe blocking trong async route: mọi inference đi qua executor.
  - Có backpressure khi client gửi audio nhanh hơn tốc độ xử lý.
  - Có cơ chế cancellation (Barge-in).

Luồng (HAI CHIỀU — chiều suy từ ngôn ngữ Whisper nhận diện):
    đối phương nói  -> dịch sang tiếng người dùng, đọc giọng thường
    người dùng nói  -> dịch sang tiếng đối phương, đọc CHẬM để nói theo

    PCM binary -> AudioChunker -> STT -> LLM -> SemanticEventParser -> EventBus
                       |                                                  |
                       └-- speech_started -> Barge-in -> hủy TTS          v
                                                                     WebSocket
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from dataclasses import dataclass

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..ai.copilot import SemanticEventParser, TranslationDelta, clean_value
from ..ai.direction import Direction
from ..ai.direction import resolve as resolve_direction
from ..ai.history import ConversationHistory
from ..ai.llm import GenerationStats
from ..ai.tts import PiperTts, SentenceSplitter, TtsUnavailable
from ..ai.verify import failure_reason, retry_hint
from ..audio.chunker import AudioChunker, AudioSegment
from ..audio.session import SessionState, UtteranceState
from ..audio.vad import VadEvent, VadEventType, build_vad
from ..core.runtime import ModelRuntime, SessionRejected
from ..protocol.events import BINARY_EVENTS, Event, EventBus, EventType
from ..protocol.schemas import ClientControl, SchemaValidationError, validate_event

logger = logging.getLogger(__name__)
router = APIRouter()


@dataclass
class _TtsItem:
    """Một mẩu cần đọc. `is_end` đánh dấu utterance đã hết phần cần đọc."""

    utterance_id: str
    text: str = ""
    field: str = "translation"
    is_end: bool = False
    #: Do client yêu cầu tường minh. Được đọc kể cả khi utterance đã kết thúc —
    #: chế độ nghe lại phát từng phần sau khi pipeline đã xong từ lâu.
    manual: bool = False
    voice: str | None = None
    #: Ghi đè tốc độ đọc. Dùng cho chiều dịch ngược: đọc chậm để nói theo.
    length_scale: float | None = None


@router.websocket("/ws/copilot")
async def copilot_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    runtime: ModelRuntime = websocket.app.state.runtime
    config = websocket.app.state.config
    session_id = f"sess_{uuid.uuid4().hex[:10]}"

    # acquire NẰM TRONG try — lỗi giữa chừng vẫn phải đi qua finally (§3.1).
    try:
        async with runtime.session(session_id):
            handler = CopilotSession(websocket, runtime, config, session_id)
            await handler.run()
    except SessionRejected as exc:
        logger.warning("Từ chối %s: %s", session_id, exc)
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {
                    "session_id": session_id,
                    "utterance_id": None,
                    "sequence": 0,
                    "type": EventType.ERROR.value,
                    "timestamp": "",
                    "data": {"message": str(exc), "code": "session_rejected",
                             "recoverable": False},
                }
            )
    except WebSocketDisconnect:
        logger.info("%s: client ngắt kết nối", session_id)
    except Exception:
        logger.exception("%s: lỗi không lường trước", session_id)
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


class CopilotSession:
    """Toàn bộ trạng thái của một kết nối WebSocket."""

    def __init__(
        self, websocket: WebSocket, runtime: ModelRuntime, config, session_id: str
    ) -> None:
        self.ws = websocket
        self.runtime = runtime
        self.config = config
        self.session_id = session_id

        self.bus = EventBus(session_id, maxsize=config.session.event_queue_maxsize)
        self.state = SessionState(session_id)
        # Bộ nhớ hội thoại: mỗi câu không còn được dịch biệt lập nữa (§10).
        self.history = ConversationHistory(
            max_turns=config.llm.history_turns,
            max_chars=config.llm.history_chars,
        )
        #: Ngôn ngữ đối phương, cập nhật theo lượt họ nói gần nhất. Dùng làm
        #: đích khi dịch ngược câu của người dùng.
        self._counterpart_language = config.session.counterpart_language
        #: Chiều của lượt trước — dùng khi Whisper không nhận diện được ngôn
        #: ngữ (câu ngắn), thay vì đoán bừa.
        self._last_direction = Direction.TO_USER
        self.tts = PiperTts(config) if config.tts.enabled else None

        vad = build_vad(config)
        self.chunker = AudioChunker(
            vad,
            min_partial_window_s=config.chunker.min_partial_window_s,
            partial_cooldown_s=config.chunker.partial_cooldown_s,
            max_utterance_s=config.chunker.max_utterance_s,
            enable_partial=config.chunker.enable_partial,
            on_vad_event=self._on_vad_event,
        )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._pipeline_task: asyncio.Task | None = None
        self._partial_task: asyncio.Task | None = None
        # TTS chạy qua hàng đợi + worker riêng. Nếu await ngay trong vòng lặp
        # token của LLM thì token stream bị treo suốt lúc phát tiếng — đúng
        # thứ §2.4 cấm ("TTS là nhánh song song, không chặn E2E của text").
        self._tts_queue: asyncio.Queue[_TtsItem | None] = asyncio.Queue()
        self._tts_worker: asyncio.Task | None = None
        self._tts_current: asyncio.Task | None = None
        self._audio_started = False
        self._dropped_audio_chunks = 0
        self._tts_available = config.tts.enabled
        #: "auto" = server tự đọc theo §2.4.1. "manual" = client tự điều phối
        #: thứ tự đọc (chế độ nghe lại từng câu).
        self._tts_mode = "auto"

    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        sender = asyncio.create_task(self._sender_loop(), name=f"sender-{self.session_id}")
        if self.tts is not None:
            self._tts_worker = asyncio.create_task(
                self._tts_worker_loop(), name=f"tts-{self.session_id}"
            )

        await self.bus.emit(
            EventType.SESSION_STARTED,
            data={
                "platform": self.config.device.platform.value,
                "max_concurrent_sessions": self.config.session.max_concurrent_sessions,
                "sample_rate": self.config.audio.sample_rate,
                "tts_enabled": bool(self.tts),
            },
        )

        try:
            await self._receive_loop()
        finally:
            await self._cancel_all_work()
            with contextlib.suppress(Exception):
                await self.bus.emit(
                    EventType.SESSION_ENDED,
                    data={"reason": "client_disconnect",
                          "utterances": self.state.total_utterances},
                )
            await self.bus.close()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(sender, timeout=3.0)
            if self.tts is not None:
                await self.tts.close()

    async def _receive_loop(self) -> None:
        while True:
            message = await self.ws.receive()

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))

            if (payload := message.get("bytes")) is not None:
                await self._on_audio(payload)
            elif (text := message.get("text")) is not None:
                await self._on_control(text)

    # -- audio vào -------------------------------------------------------- #

    async def _on_audio(self, payload: bytes) -> None:
        if not self._audio_started:
            self._audio_started = True
            await self.bus.emit(
                EventType.AUDIO_STARTED,
                data={"sample_rate": self.config.audio.sample_rate,
                      "channels": self.config.audio.channels},
            )

        # Backpressure (F5): nếu pipeline trước còn đang chạy và hàng đợi event
        # đã đầy, tiếp tục nuốt audio chỉ làm phình bộ nhớ. VAD vẫn được chạy để
        # không mất mốc đầu/cuối câu, nhưng ta ghi nhận và cảnh báo.
        if self.bus.dropped > self._dropped_audio_chunks + 50:
            self._dropped_audio_chunks = self.bus.dropped
            logger.warning(
                "%s: event bus đang quá tải (%d event bị drop)",
                self.session_id, self.bus.dropped,
            )

        for segment in self.chunker.feed_bytes(payload):
            if segment.is_final:
                await self._start_pipeline(segment)
            else:
                self._start_partial(segment)

    def _on_vad_event(self, event: VadEvent) -> None:
        """Chạy đồng bộ trong lúc parse audio — phải cực nhẹ.

        Đây là đường tới hạn của Barge-in (< 200ms), nên chỉ lên lịch task chứ
        không await gì ở đây.
        """
        if event.type is not VadEventType.SPEECH_STARTED:
            return
        if self._loop is None:
            return
        if self.tts is not None and self.tts.is_active:
            self._loop.create_task(self._barge_in())

    async def _barge_in(self) -> None:
        """§2.4.1: speech mới khi TTS đang PLAYING -> hủy, quay lại lắng nghe."""
        if self.tts is None:
            return
        job = self.tts.current_job
        dropped = self._drain_tts_queue()
        result = await self.tts.cancel(reason="barge_in")
        if not result.cancelled:
            if dropped:
                logger.debug("Barge-in: bỏ %d mẩu chưa đọc", dropped)
            return
        await self.bus.emit(
            EventType.TTS_CANCELLED,
            utterance_id=job.utterance_id if job else None,
            data={
                "reason": "barge_in",
                "response_ms": round(result.response_ms, 2),
                "chunks_sent": result.chunks_sent,
            },
        )

    # -- STT partial ------------------------------------------------------ #

    def _start_partial(self, segment: AudioSegment) -> None:
        # Partial là TÙY CHỌN. Nếu partial trước còn đang chạy thì bỏ qua cái
        # mới — chạy chồng chỉ tổ tranh chấp GPU với final STT và LLM.
        if self._partial_task is not None and not self._partial_task.done():
            return
        utterance = self.state.current
        if utterance is None or utterance.is_terminal:
            return
        self._partial_task = asyncio.create_task(self._run_partial(segment, utterance.id))

    async def _run_partial(self, segment: AudioSegment, utterance_id: str) -> None:
        try:
            async with self.runtime.job("stt_partial"):
                transcript = await self.runtime.stt.transcribe(segment.pcm, is_final=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("%s: partial STT lỗi", self.session_id)
            return

        utterance = self.state.get(utterance_id)
        if transcript.is_empty or utterance is None or utterance.is_terminal:
            return
        utterance.partial_text = transcript.text
        await self.bus.emit(
            EventType.STT_PARTIAL,
            utterance_id=utterance_id,
            data={"text": transcript.text, "language": transcript.language,
                  "window_s": round(segment.duration_s, 3)},
        )

    # -- pipeline chính --------------------------------------------------- #

    async def _start_pipeline(self, segment: AudioSegment) -> None:
        # Utterance mới chiếm chỗ utterance cũ: hủy pipeline cũ, hủy TTS cũ.
        if self._pipeline_task is not None and not self._pipeline_task.done():
            self._pipeline_task.cancel()
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()
        # Utterance mới chiếm chỗ: mọi thứ còn chờ đọc của câu cũ không còn
        # ý nghĩa — hội thoại đã đi tiếp.
        self._drain_tts_queue()

        utterance = self.state.begin_utterance()
        utterance.mark_endpoint()
        self._pipeline_task = asyncio.create_task(
            self._run_pipeline(segment, utterance.id), name=f"pipeline-{utterance.id}"
        )

    async def _run_pipeline(self, segment: AudioSegment, utterance_id: str) -> None:
        try:
            await self._transcribe_and_respond(segment, utterance_id)
        except asyncio.CancelledError:
            logger.debug("%s: pipeline %s bị hủy", self.session_id, utterance_id)
            raise
        except Exception as exc:
            logger.exception("%s: pipeline %s lỗi", self.session_id, utterance_id)
            with contextlib.suppress(Exception):
                self.state.transition(utterance_id, UtteranceState.FAILED)
            await self.bus.emit(
                EventType.ERROR,
                data={"message": str(exc), "code": "pipeline_error", "recoverable": True},
            )

    async def _transcribe_and_respond(self, segment: AudioSegment, utterance_id: str) -> None:
        utterance = self.state.get(utterance_id)
        if utterance is None:
            return

        # --- final STT (bắt buộc, do VAD endpoint kích hoạt) ---
        self.state.transition(utterance_id, UtteranceState.TRANSCRIBING)
        async with self.runtime.job("stt_final"):
            transcript = await self.runtime.stt.transcribe(segment.pcm, is_final=True)

        if transcript.is_empty:
            logger.debug("%s: %s transcript rỗng", self.session_id, utterance_id)
            self.state.transition(utterance_id, UtteranceState.DONE)
            return

        utterance.final_text = transcript.text
        utterance.language = transcript.language

        direction = resolve_direction(
            transcript.language,
            user_language=self.config.session.user_language,
            fallback=self._last_direction,
        )
        self._last_direction = direction
        if direction is Direction.TO_USER and transcript.language:
            # Nghe được họ nói tiếng gì thì lấy đó làm đích cho chiều ngược,
            # thay vì bám mãi vào mặc định trong config.
            self._counterpart_language = transcript.language

        if self.config.llm.history_turns > 0:
            self.history.add(
                utterance_id,
                transcript.text,
                transcript.language,
                is_user=direction.is_outbound,
            )
        await self.bus.emit(
            EventType.STT_FINAL,
            utterance_id=utterance_id,
            data={
                "text": transcript.text,
                "language": transcript.language,
                "duration_s": round(transcript.audio_s, 3),
                "latency_ms": round(transcript.latency_ms, 2),
                # Vị trí câu trong dòng byte audio — client đếm cùng dòng đó
                # nên cắt lại được đúng đoạn audio GỐC để nghe đối chiếu.
                "start_s": round(max(0.0, segment.start_s), 3),
                "direction": direction.value,
            },
        )

        # --- LLM streaming -> semantic events ---
        self.state.transition(utterance_id, UtteranceState.COPILOT)
        await self.bus.emit(
            EventType.COPILOT_STARTED,
            utterance_id=utterance_id,
            data={
                "source_text": transcript.text,
                "language": transcript.language,
                "direction": direction.value,
            },
        )

        target_language = (
            self._counterpart_language
            if direction.is_outbound
            else self.config.session.user_language
        )
        parser = SemanticEventParser()
        stats = GenerationStats()
        splitter = SentenceSplitter(self.config.tts.min_sentence_chars)
        # Loại chính câu này khỏi lịch sử: nó đã nằm ở phần "Now they said"
        # của prompt, đưa vào hai lần sẽ khiến model tưởng bị nói lặp.
        context = (
            self.history.render(exclude=utterance_id)
            if self.config.llm.history_turns > 0
            else ""
        )
        prompt = self.runtime.llm.build_prompt(
            transcript.text,
            transcript.language,
            direction=direction,
            counterpart_language=self._counterpart_language,
            history=context,
        )

        async with self.runtime.job("llm"):
            token_stream = self.runtime.llm.stream(prompt, stats=stats)
            async for token in token_stream:
                for event in parser.feed(token):
                    await self._emit_semantic(utterance_id, event, utterance, splitter)

        for event in parser.finish():
            await self._emit_semantic(utterance_id, event, utterance, splitter)

        # Lưới an toàn: model 2B thỉnh thoảng chép nguyên văn thay vì dịch.
        # Dịch lại MỘT lần — lần hai còn hỏng thì lần ba cũng thế.
        if self.config.llm.retry_on_bad_translation:
            reason = failure_reason(
                transcript.text, parser.result.translation, target_language
            )
            if reason is not None:
                logger.info(
                    "%s: bản dịch hỏng (%s) — dịch lại một lần",
                    self.session_id, reason,
                )
                parser = await self._retranslate(
                    utterance_id, utterance, transcript, direction,
                    context, target_language, reason, stats,
                )

        self.history.set_translation(utterance_id, parser.result.translation)

        await self.bus.emit(
            EventType.COPILOT_DONE,
            utterance_id=utterance_id,
            data={
                "ttft_ms": round(stats.ttft_ms, 2) if stats.ttft_ms is not None else None,
                "total_ms": round(stats.total_ms, 2),
                "tokens": stats.tokens,
                "truncated": stats.truncated,
            },
        )

        # --- TTS phần translation còn lại ---
        remainder = splitter.flush()
        if remainder and self._should_speak("translation"):
            self._speak_translation(utterance_id, remainder)

        # Utterance chỉ DONE sau khi worker đọc hết phần đã xếp hàng. Nếu
        # chuyển DONE ngay ở đây thì các mẩu còn trong hàng đợi sẽ bị worker
        # bỏ qua vì utterance đã ở trạng thái kết thúc.
        if not utterance.is_terminal:
            if self._tts_available and self.tts is not None:
                self._mark_speech_end(utterance_id)
            else:
                with contextlib.suppress(Exception):
                    self.state.transition(utterance_id, UtteranceState.DONE)

    async def _retranslate(
        self, utterance_id, utterance, transcript, direction,
        context, target_language, reason, stats,
    ) -> SemanticEventParser:
        """Dịch lại với lời nhắc cứng hơn. Trả về parser của lần dịch mới.

        TTS của lần đầu (nếu có) đã bị hủy trước khi đọc bản hỏng — không để
        người dùng nghe câu tiếng mình đọc bằng giọng nước ngoài.
        """
        from ..ai.llm import language_name

        if self.tts is not None and self.tts.is_active:
            self._drain_tts_queue()
            await self.tts.cancel(reason="new_utterance")

        prompt = self.runtime.llm.build_prompt(
            transcript.text,
            transcript.language,
            direction=direction,
            counterpart_language=self._counterpart_language,
            history=context,
            retry_hint=retry_hint(reason, language_name(target_language)),
        )
        parser = SemanticEventParser()
        splitter = SentenceSplitter(self.config.tts.min_sentence_chars)
        async with self.runtime.job("llm_retry"):
            async for token in self.runtime.llm.stream(prompt, stats=stats):
                for event in parser.feed(token):
                    await self._emit_semantic(utterance_id, event, utterance, splitter)
        for event in parser.finish():
            await self._emit_semantic(utterance_id, event, utterance, splitter)

        remainder = splitter.flush()
        if remainder and self._should_speak("translation"):
            self._speak_translation(utterance_id, remainder)
        return parser

    async def _emit_semantic(self, utterance_id, event, utterance, splitter) -> None:
        if not isinstance(event, TranslationDelta):
            return

        direction = self._last_direction
        target = (
            self._counterpart_language
            if direction.is_outbound
            else self.config.session.user_language
        )
        utterance.mark_first_useful()
        await self.bus.emit(
            EventType.TRANSLATION_DELTA,
            utterance_id=utterance_id,
            data={
                "text": event.text,
                "full": event.full,
                "direction": direction.value,
                "language": target,
            },
        )

        # Streaming TTS theo câu (§2.4 / Task E6): bắt đầu đọc ngay khi có một
        # câu hoàn chỉnh, không đợi cả JSON.
        if self._should_speak("translation") and self.config.tts.stream_by_sentence:
            for sentence in splitter.feed(event.text):
                self._speak_translation(utterance_id, sentence)

    def _speak_translation(self, utterance_id: str, text: str) -> None:
        """Đọc bản dịch, đúng giọng và đúng tốc độ cho chiều hiện tại.

        Dọn rác NGAY TẠI ĐÂY chứ không dựa vào việc `result.translation` đã
        được dọn: các mẩu câu đi tới TTS được cắt ra TRONG LÚC token còn đang
        về, trước khi chuỗi JSON đóng và `clean_value` kịp chạy. Không dọn ở
        đây thì người dùng NHÌN thấy bản dịch sạch nhưng NGHE thấy rác — đã
        quan sát thật với `"...nên hoãn.},{"`.
        """
        text = clean_value(text)
        if not any(ch.isalnum() for ch in text):
            # Mẩu chỉ toàn dấu và ký tự cấu trúc — không có gì để đọc.
            return

        direction = self._last_direction
        if direction.is_outbound:
            # Người dùng phải NÓI THEO, không chỉ nghe hiểu — đọc chậm hẳn và
            # bằng giọng của tiếng đối phương.
            self._speak(
                utterance_id,
                text,
                field="coach",
                voice=self._voice_for(self._counterpart_language),
                length_scale=self.config.tts.coach_length_scale,
            )
        else:
            self._speak(
                utterance_id,
                text,
                field="translation",
                voice=self._voice_for(self.config.session.user_language),
            )

    def _voice_for(self, language: str | None) -> str:
        """Chọn giọng theo ngôn ngữ. Chỉ có hai giọng nên phần còn lại về "en"."""
        code = (language or "en").strip().lower().split("-")[0]
        return "vi" if code == "vi" else "en"

    # -- TTS -------------------------------------------------------------- #

    def _should_speak(self, field: str) -> bool:
        if self.tts is None or not self._tts_available:
            return False
        if self._tts_mode == "manual":
            return False
        cfg = self.config.tts
        return cfg.auto_read_translation if field == "translation" else False

    def _speak(
        self,
        utterance_id: str,
        text: str,
        *,
        field: str,
        manual: bool = False,
        voice: str | None = None,
        length_scale: float | None = None,
    ) -> None:
        """Xếp một đoạn vào hàng đợi đọc. KHÔNG chờ đọc xong.

        Worker phát tuần tự nên giọng không chồng lên nhau, nhưng vòng lặp
        token của LLM vẫn chạy tiếp trong lúc đó.
        """
        if self.tts is None or not self._tts_available:
            return
        if not text.strip():
            return
        self._tts_queue.put_nowait(
            _TtsItem(
                utterance_id, text, field,
                manual=manual, voice=voice, length_scale=length_scale,
            )
        )

    def _mark_speech_end(self, utterance_id: str) -> None:
        if self.tts is None or not self._tts_available:
            return
        self._tts_queue.put_nowait(_TtsItem(utterance_id, is_end=True))

    async def _tts_worker_loop(self) -> None:
        """Đọc tuần tự từng mẩu trong hàng đợi."""
        try:
            while True:
                item = await self._tts_queue.get()
                if item is None:
                    return

                utterance = self.state.get(item.utterance_id)
                if utterance is None:
                    continue
                # Yêu cầu thủ công vẫn đọc được sau khi utterance đã kết thúc —
                # đó chính là lúc người dùng bấm chọn một gợi ý trả lời.
                if utterance.is_terminal and not item.manual:
                    continue

                if item.is_end:
                    if utterance.state in (UtteranceState.SPEAKING, UtteranceState.COPILOT):
                        with contextlib.suppress(Exception):
                            self.state.transition(item.utterance_id, UtteranceState.DONE)
                    continue

                if utterance.state is UtteranceState.COPILOT:
                    self.state.transition(item.utterance_id, UtteranceState.SPEAKING)

                self._tts_current = asyncio.create_task(
                    self._run_tts(
                        item.utterance_id, item.text, item.field,
                        item.voice, item.length_scale,
                    )
                )
                with contextlib.suppress(asyncio.CancelledError):
                    await self._tts_current
                self._tts_current = None
        except asyncio.CancelledError:
            raise

    def _drain_tts_queue(self) -> int:
        """Bỏ mọi mẩu chưa đọc. Dùng khi Barge-in hoặc utterance mới chiếm chỗ."""
        dropped = 0
        while True:
            try:
                item = self._tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                return dropped
            if item is not None and not item.is_end:
                dropped += 1

    async def _run_tts(
        self,
        utterance_id: str,
        text: str,
        field: str,
        voice: str | None = None,
        length_scale: float | None = None,
    ) -> None:
        assert self.tts is not None
        voice = voice or self.config.tts.voice
        chunks = 0
        try:
            self.tts.preflight(voice)
        except TtsUnavailable as exc:
            # Thiếu Piper không được làm sập pipeline: text vẫn phải hiển thị.
            # Tắt TTS cho phần còn lại của session và báo một lần.
            self._tts_available = False
            logger.warning("%s: tắt TTS — %s", self.session_id, exc)
            await self.bus.emit(
                EventType.TTS_ERROR,
                utterance_id=utterance_id,
                data={"message": str(exc), "stage": "synthesis"},
            )
            return

        await self.bus.emit(
            EventType.TTS_STARTED,
            utterance_id=utterance_id,
            data={"utterance_field": field, "text": text, "voice": voice,
                  "sample_rate": self.config.tts.sample_rate},
        )

        import time as _time

        started = _time.monotonic()
        try:
            async for audio in self.tts.synthesize(
                utterance_id, text, field=field, voice=voice,
                length_scale=length_scale,
            ):
                chunks += 1
                await self.bus.emit(
                    EventType.TTS_AUDIO_CHUNK,
                    utterance_id=utterance_id,
                    data={"chunk_index": chunks, "bytes": len(audio),
                          "sample_rate": self.config.tts.sample_rate},
                    binary=audio,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("%s: TTS lỗi", self.session_id)
            await self.bus.emit(
                EventType.TTS_ERROR,
                utterance_id=utterance_id,
                data={"message": str(exc), "stage": "synthesis"},
            )
            return

        await self.bus.emit(
            EventType.TTS_DONE,
            utterance_id=utterance_id,
            data={"chunks": chunks,
                  "synthesis_ms": round((_time.monotonic() - started) * 1000.0, 2)},
        )

    # -- control frame (client -> server) --------------------------------- #

    async def _on_control(self, raw: str) -> None:
        try:
            control = ClientControl.model_validate(json.loads(raw))
        except Exception as exc:
            await self.bus.emit(
                EventType.ERROR,
                data={"message": f"control frame không hợp lệ: {exc}",
                      "code": "bad_control", "recoverable": True},
            )
            return

        if control.action == "ping":
            return

        if control.action == "cancel_tts" and self.tts is not None:
            job = self.tts.current_job
            self._drain_tts_queue()
            result = await self.tts.cancel(reason="client_request")
            if result.cancelled:
                await self.bus.emit(
                    EventType.TTS_CANCELLED,
                    utterance_id=job.utterance_id if job else None,
                    data={"reason": "client_request",
                          "response_ms": round(result.response_ms, 2),
                          "chunks_sent": result.chunks_sent},
                )
            return

        if control.action == "set_tts_mode":
            self._tts_mode = control.mode or "auto"
            logger.info("%s: chế độ TTS -> %s", self.session_id, self._tts_mode)
            return

        if control.action == "speak":
            # Client tự điều phối thứ tự đọc (chế độ nghe lại từng câu).
            utterance = self.state.current
            if utterance is None or self.tts is None or not control.text:
                return
            self._speak(
                utterance.id,
                control.text,
                field=control.field or "translation",
                manual=True,
                voice=control.voice,
            )
            return

        if control.action == "reset":
            await self._cancel_all_work()
            self.chunker.reset()
            self.history.clear()

    # -- dọn dẹp ---------------------------------------------------------- #

    async def _cancel_all_work(self) -> None:
        if self.tts is not None:
            await self.tts.cancel(reason="session_end")
        self._drain_tts_queue()
        for task in (self._pipeline_task, self._partial_task, self._tts_current,
                     self._tts_worker):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self.state.cancel_all_active()

    # -- gửi ra client ---------------------------------------------------- #

    async def _sender_loop(self) -> None:
        """Tuần tự hóa việc gửi. Chỉ MỘT task được chạm vào WebSocket."""
        try:
            while True:
                event = await self.bus.get()
                if event is None:
                    return
                await self._send(event)
        except (WebSocketDisconnect, RuntimeError):
            logger.debug("%s: sender dừng vì kết nối đã đóng", self.session_id)
        except asyncio.CancelledError:
            raise

    async def _send(self, event: Event) -> None:
        try:
            payload = validate_event(event)
        except SchemaValidationError as exc:
            # Event không hợp lệ KHÔNG được lọt ra client (§4.3 / Task F3).
            logger.error("%s: %s", self.session_id, exc)
            return

        await self.ws.send_json(payload)
        if event.type in BINARY_EVENTS and event.binary:
            await self.ws.send_bytes(event.binary)
