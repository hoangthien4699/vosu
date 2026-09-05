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

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..ai.completeness import looks_complete
from ..ai.copilot import SemanticEventParser, TranslationDelta, clean_value
from ..ai.direction import Direction
from ..ai.direction import resolve as resolve_direction
from ..ai.history import ConversationHistory
from ..ai.llm import GenerationStats
from ..ai.speaker import SpeakerEmbedder, SpeakerTracker
from ..ai.tts import SentenceSplitter, TtsUnavailable
from ..ai.tts_router import create_tts
from ..ai.verify import failure_reason, retry_hint
from ..audio.chunker import AudioChunker, AudioSegment
from ..audio.session import SessionState, UtteranceState
from ..audio.vad import VadEvent, VadEventType, build_vad
from ..core.runtime import ModelRuntime, SessionRejected
from ..protocol.events import BINARY_EVENTS, Event, EventBus, EventType
from ..protocol.schemas import ClientControl, SchemaValidationError, validate_event

logger = logging.getLogger(__name__)
router = APIRouter()


@dataclass(eq=False)
class _HeldSegment:
    """Câu nghe ra còn dở, đang chờ xem người ta có nói tiếp không."""

    segment: AudioSegment
    transcript: object
    utterance_id: str
    merges: int
    #: Người nói của mảnh này, hoặc None nếu không đủ chắc.
    speaker: str | None = None


def _doi_nguoi(truoc: str | None, sau: str | None) -> bool:
    """Chắc chắn đã đổi người nói chưa.

    Chỉ True khi BIẾT CẢ HAI và chúng khác nhau. Không biết một trong hai thì
    trả False — quyết bừa theo hướng "người khác" sẽ cắt đôi một câu đang nói
    dở, tệ hơn hẳn so với để cơ chế cũ xử lý.
    """
    return truoc is not None and sau is not None and truoc != sau


def _join_segments(first: AudioSegment, second: AudioSegment) -> AudioSegment:
    """Nối hai đoạn thành một câu liền.

    Chèn lại đúng khoảng lặng giữa hai đoạn thay vì dán sát: dán sát thì
    Whisper nghe ra một từ ghép không có thật ở chỗ nối.
    """
    gap_s = max(0.0, second.start_s - (first.start_s + first.duration_s))
    gap = np.zeros(int(gap_s * first.sample_rate), dtype=first.pcm.dtype)
    pcm = np.concatenate([first.pcm, gap, second.pcm])
    return AudioSegment(
        kind=second.kind,
        pcm=pcm,
        sample_rate=first.sample_rate,
        start_s=first.start_s,
        duration_s=pcm.size / first.sample_rate,
        trigger=f"{first.trigger}+merge",
    )


@dataclass(eq=False)  # AudioSegment chứa numpy — so sánh mặc định sẽ nổ
class _PipelineItem:
    """Một câu đang chờ tới lượt xử lý."""

    segment: AudioSegment
    utterance_id: str


@dataclass(eq=False)
class _LlmItem:
    """Một câu đã nghe xong, đang chờ dịch."""

    utterance_id: str
    transcript: object
    direction: Direction
    target_language: str
    counterpart_language: str
    #: Lịch sử chốt tại thời điểm NGHE XONG. Chặng STT chạy trước chặng dịch,
    #: nên nếu render lúc dịch thì prompt sẽ chứa cả những câu nói SAU câu này.
    context: str


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
        self.tts = create_tts(config) if config.tts.enabled else None
        #: Gom các đoạn nói thành người nói. Đổi người = mốc CHẮC CHẮN để
        #: cắt câu, khác hẳn độ dài khoảng lặng vốn không tách được.
        self.speakers = SpeakerTracker(
            same_threshold=config.stt.speaker_same_threshold,
            diff_threshold=config.stt.speaker_diff_threshold,
        )
        self.speakers_embedder = SpeakerEmbedder(config)

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
        # Các câu được XẾP HÀNG chứ không chiếm chỗ nhau. Trước đây câu mới
        # hủy thẳng pipeline của câu cũ; khi các câu nối nhau sát hơn thời gian
        # xử lý một câu thì câu cũ bị vứt lặng lẽ — người dùng mất hẳn câu đó
        # mà log vẫn sạch. Đo thật với Whisper `small`: file ba câu chỉ ra
        # bản dịch của câu CUỐI, hai câu đầu có transcript rồi biến mất.
        self._pipeline_queue: asyncio.Queue[_PipelineItem | None] = asyncio.Queue()
        self._pipeline_worker: asyncio.Task | None = None
        self._pipeline_task: asyncio.Task | None = None
        # Chặng 2 chạy SONG SONG với chặng 1. Chúng dùng hai tài nguyên khác
        # nhau — STT ăn CPU, LLM ăn GPU — nên xếp nối tiếp là phí: thông lượng
        # tụt xuống tổng của hai chặng thay vì chặng chậm nhất. Đo thật khi còn
        # nối tiếp, file 6 câu: độ trễ dồn 3.6s -> 6.5s, mỗi câu tụt thêm ~570ms
        # và không có điểm dừng.
        #: Câu nghe ra còn dở, đang giữ lại chờ ghép với đoạn nói tiếp.
        self._held: _HeldSegment | None = None
        self._hold_timer: asyncio.Task | None = None
        self._llm_queue: asyncio.Queue[_LlmItem | None] = asyncio.Queue()
        self._llm_worker: asyncio.Task | None = None
        self._llm_task: asyncio.Task | None = None
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
        self._pipeline_worker = asyncio.create_task(
            self._pipeline_worker_loop(), name=f"stt-{self.session_id}"
        )
        self._llm_worker = asyncio.create_task(
            self._llm_worker_loop(), name=f"llm-{self.session_id}"
        )
        if self.tts is not None:
            self._tts_worker = asyncio.create_task(
                self._tts_worker_loop(), name=f"tts-{self.session_id}"
            )

        if self.config.stt.speaker_split:
            # Tiến trình phụ mất ~9.4s khởi động (dựng phiên ONNX + làm nóng).
            # Không gọi ở đây thì CÂU ĐẦU TIÊN gánh trọn ngần ấy.
            self._loop.create_task(self.speakers_embedder.start())

        if self.tts is not None and getattr(self.tts, "needs_preload", False):
            # Engine có model thường trú thì nạp NGAY từ bây giờ, ở nền — tới
            # lúc bản dịch đầu tiên xong (~6s) là vừa kịp. Piper không cần:
            # nó hâm nóng tiến trình dự phòng SAU mỗi lượt đọc, và làm ở đây
            # chỉ tổ spawn thừa một tiến trình cho mọi session.
            self.tts.prewarm()

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
            # CÓ GIỚI HẠN THỜI GIAN. Dọn dẹp mà treo thì `run()` không bao giờ
            # trả về, session ở lại trong sổ đếm, và MỌI kết nối sau đều bị từ
            # chối "đã đạt giới hạn 1 session" cho tới khi restart server. Đã
            # gặp thật hai lần, hai nguyên nhân khác nhau — nên chặn ở đây một
            # lần cho mọi nguyên nhân, thay vì chỉ vá từng cái.
            try:
                await asyncio.wait_for(self._cancel_all_work(), timeout=5.0)
            except asyncio.TimeoutError:
                stuck = [
                    t.get_name() for t in asyncio.all_tasks()
                    if t is not asyncio.current_task() and not t.done()
                ]
                logger.error(
                    "%s: dọn dẹp quá 5s, bỏ qua để giải phóng session. "
                    "Task còn chạy: %s",
                    self.session_id, ", ".join(stuck) or "không rõ",
                )
            with contextlib.suppress(Exception):
                await self.bus.emit(
                    EventType.SESSION_ENDED,
                    data={"reason": "client_disconnect",
                          "utterances": self.state.total_utterances},
                )
            await self.bus.close()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
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

        if self._held is not None:
            await self._expire_hold_by_audio()

    def _on_vad_event(self, event: VadEvent) -> None:
        """Chạy đồng bộ trong lúc parse audio — phải cực nhẹ.

        Đây là đường tới hạn của Barge-in (< 200ms), nên chỉ lên lịch task chứ
        không await gì ở đây.
        """
        if event.type is not VadEventType.SPEECH_STARTED:
            return
        if self._loop is None:
            return
        if not self.config.tts.barge_in:
            # Xem TtsConfig.barge_in: tiếng nói mới ở sản phẩm này thường là
            # đối phương nói tiếp, không phải người dùng chen ngang. Cắt lời
            # lúc đó là làm mất chính nội dung họ cần nghe.
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
        """Xếp câu vừa dứt vào hàng đợi. KHÔNG hủy câu đang xử lý.

        Bản dịch là sản phẩm chính: mất một câu thì người dùng không bao giờ
        biết đối phương vừa nói gì. Chậm một nhịp còn chữa được, mất thì không.
        """
        # Partial của câu cũ thì bỏ được — nó chỉ là bản nháp trên màn hình.
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()

        utterance = self.state.begin_utterance()
        utterance.mark_endpoint()

        # Nếu tụt lại quá xa thì bản dịch cũ đã lỗi thời so với cuộc nói
        # chuyện đang diễn ra — bỏ câu CŨ NHẤT để bám sát thời gian thực,
        # và nói rõ ra chứ không im lặng.
        limit = self.config.session.max_pending_utterances
        while self._pipeline_queue.qsize() >= limit:
            try:
                stale = self._pipeline_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._pipeline_queue.task_done()
            if stale is None:
                continue
            logger.warning(
                "%s: xử lý không kịp, bỏ câu %s (hàng đợi đã %d)",
                self.session_id, stale.utterance_id, limit,
            )
            await self.bus.emit(
                EventType.UTTERANCE_DROPPED,
                utterance_id=stale.utterance_id,
                data={"reason": "backlog", "pending": limit},
            )

        self._pipeline_queue.put_nowait(_PipelineItem(segment, utterance.id))

        # Báo NGAY, trước khi STT chạy: client phát file dừng đúng chỗ câu vừa
        # dứt thay vì đợi `stt_final` (chậm hơn ~1.9s, đã lấn sang câu sau).
        await self.bus.emit(
            EventType.UTTERANCE_ENDPOINT,
            utterance_id=utterance.id,
            data={
                "start_s": round(max(0.0, segment.start_s), 3),
                "duration_s": round(segment.duration_s, 3),
                "trigger": segment.trigger,
            },
        )

    async def _pipeline_worker_loop(self) -> None:
        """Chạy từng câu một, đúng thứ tự nghe được.

        Một câu một lúc là cố ý: STT và LLM đã tranh nhau CPU/GPU rồi, chạy
        chồng chỉ làm cả hai cùng chậm.
        """
        while True:
            item = await self._pipeline_queue.get()
            try:
                if item is None:
                    return
                # Task riêng cho từng câu: `reset` phải hủy được câu đang chạy
                # mà KHÔNG giết worker, nếu không thì sau reset không câu nào
                # được xử lý nữa. `asyncio.wait` chờ xong mà không ném lại lỗi
                # của task con, nên hủy câu và hủy worker phân biệt được nhau.
                task = asyncio.create_task(
                    self._run_pipeline(item.segment, item.utterance_id),
                    name=f"utt-{item.utterance_id}",
                )
                self._pipeline_task = task
                try:
                    await asyncio.wait({task})
                except asyncio.CancelledError:
                    task.cancel()
                    raise
                # Câu vừa xử lý có thể đã GIỮ LẠI một mảnh chờ ghép. Phép kiểm
                # hết-giờ theo đồng hồ audio chỉ chạy khi có audio mới tới —
                # mà audio có thể đã gửi hết từ trước. Kiểm ngay tại đây, chỗ
                # duy nhất chắc chắn pipeline đã rảnh.
                #
                # Thiếu bước này thì mảnh câu phải chờ tới lưới dự phòng đồng
                # hồ thật (15s) mới được dịch. Đã đo: một test từ 0.33s lên
                # 15.01s.
                if self._held is not None:
                    await self._expire_hold_by_audio()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s: pipeline worker lỗi", self.session_id)
            finally:
                self._pipeline_queue.task_done()

    @staticmethod
    def _drain(queue: asyncio.Queue) -> int:
        dropped = 0
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return dropped
            queue.task_done()
            if item is not None:
                dropped += 1

    def _drain_pipeline_queue(self) -> int:
        """Bỏ mọi câu còn chờ ở cả hai chặng. Dùng khi reset/kết thúc session."""
        return self._drain(self._pipeline_queue) + self._drain(self._llm_queue)

    def _abort_in_flight(self) -> None:
        """Vứt hết việc đang làm nhưng GIỮ worker sống, để còn nhận câu mới."""
        self._drain_pipeline_queue()
        for task in (self._pipeline_task, self._llm_task, self._partial_task):
            if task is not None and not task.done():
                task.cancel()

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

        # Mảnh câu đang giữ (lần trước nghe ra một câu dở) -> ghép audio rồi
        # nghe lại trên đoạn đã liền. Nghe lại chứ không nối hai transcript:
        # Whisper trên audio liền mạch cho ra câu đúng ngữ pháp hơn hẳn.
        held = self._held
        merges = 0
        speaker: str | None = None

        if held is None:
            # Đường phổ biến: nghe và trích vector giọng SONG SONG. Whisper mất
            # ~2s, trích vector ~440ms — chạy cùng lúc thì vector miễn phí.
            async with self.runtime.job("stt_final"):
                transcript, vector = await asyncio.gather(
                    self.runtime.stt.transcribe(segment.pcm, is_final=True),
                    self._embed(segment),
                )
            speaker = self.speakers.assign(vector) if vector is not None else None
        else:
            # Có mảnh câu đang giữ: phải biết giọng TRƯỚC mới quyết được có
            # ghép hay không, nên trích trước rồi mới nghe. Tốn thêm ~440ms,
            # nhưng chỉ ở nhánh này.
            self._cancel_hold_timer()
            self._held = None
            vector = await self._embed(segment)
            speaker = self.speakers.assign(vector) if vector is not None else None

            if _doi_nguoi(held.speaker, speaker):
                # ĐỔI NGƯỜI NÓI = câu trước đã xong, chắc chắn. Không ghép.
                # Đây là thứ độ dài khoảng lặng không bao giờ nói được: khoảng
                # ngập ngừng giữa câu (800ms) còn dài hơn khoảng nghỉ giữa hai
                # câu (700ms).
                logger.info(
                    "%s: %s đổi người nói (%s -> %s) — không ghép",
                    self.session_id, utterance_id, held.speaker, speaker,
                )
                await self._flush_held(held)
            else:
                segment = _join_segments(held.segment, segment)
                merges = held.merges + 1
                self._retire(held.utterance_id)
                logger.info(
                    "%s: ghép %s vào %s (lần %d) — câu trước còn dở",
                    self.session_id, held.utterance_id, utterance_id, merges,
                )

            async with self.runtime.job("stt_final"):
                transcript = await self.runtime.stt.transcribe(segment.pcm, is_final=True)

        if transcript.is_empty:
            logger.debug("%s: %s transcript rỗng", self.session_id, utterance_id)
            self.state.transition(utterance_id, UtteranceState.DONE)
            return

        if self._should_hold(transcript.text, segment, merges):
            await self._hold_for_more(segment, transcript, utterance_id, merges, speaker)
            return

        await self._after_transcript(segment, transcript, utterance_id, utterance)

    # -- gộp câu bị ngắt giữa chừng --------------------------------------- #

    async def _embed(self, segment: AudioSegment):
        if not self.config.stt.speaker_split:
            return None
        return await self.speakers_embedder.embed(segment.pcm)

    async def _flush_held(self, held: _HeldSegment) -> None:
        """Đẩy mảnh câu đang giữ đi tiếp như một câu riêng.

        Dùng khi đổi người nói: câu trước đã xong, đừng bắt nó chờ ghép nữa.
        """
        utterance = self.state.get(held.utterance_id)
        if utterance is None or utterance.is_terminal:
            return
        await self._after_transcript(
            held.segment, held.transcript, held.utterance_id, utterance
        )

    def _should_hold(self, text: str, segment: AudioSegment, merges: int) -> bool:
        cfg = self.config.stt
        if not cfg.merge_incomplete or merges >= cfg.max_merges:
            return False
        if segment.duration_s >= cfg.max_merged_s:
            return False
        return not looks_complete(text)

    async def _hold_for_more(
        self, segment: AudioSegment, transcript, utterance_id: str, merges: int,
        speaker: str | None = None,
    ) -> None:
        """Giữ câu dở lại, chờ xem người ta có nói tiếp không.

        Phải BÁO RA client: ở chế độ phát file, client đã dừng file ngay khi
        nghe `utterance_endpoint`. Không báo thì nó dừng vĩnh viễn — audio cần
        để quyết định sẽ không bao giờ tới.
        """
        self._held = _HeldSegment(segment, transcript, utterance_id, merges, speaker)
        logger.info(
            "%s: %s nghe ra câu dở (%r) — chờ nói tiếp",
            self.session_id, utterance_id, transcript.text[:48],
        )
        await self.bus.emit(
            EventType.UTTERANCE_CONTINUED,
            utterance_id=utterance_id,
            data={
                "text": transcript.text,
                "reason": "incomplete",
                "wait_ms": self.config.stt.merge_window_ms,
            },
        )
        # Backstop bằng đồng hồ thật, đề phòng client ngừng gửi audio hẳn —
        # lúc đó đồng hồ audio đứng yên và câu giữ lại sẽ kẹt mãi.
        self._hold_timer = asyncio.create_task(
            self._flush_hold_after(utterance_id, self.config.stt.merge_backstop_s),
            name=f"hold-{utterance_id}",
        )

    async def _expire_hold_by_audio(self) -> None:
        """Đã nghe thêm đủ lâu mà không ai nói tiếp -> thôi chờ.

        Đo bằng ĐỒNG HỒ AUDIO. Đo bằng đồng hồ thật thì hỏng ở đúng chế độ
        đang dùng: client dừng file ngay tại endpoint, nên trong lúc chờ không
        có audio nào chạy — cửa sổ tự hết giờ trước khi đoạn nói tiếp kịp tới,
        và câu không bao giờ được ghép. Đã đo thấy đúng như vậy.
        """
        held = self._held
        if held is None:
            return
        # Neo vào chỗ đoạn audio KẾT THÚC, không vào lúc bắt đầu giữ: lúc bắt
        # đầu giữ là sau khi STT xong, mà STT xong lúc nào thì tùy client có
        # đang dừng file hay không. Neo vào vị trí audio thì hai chế độ đo ra
        # cùng một thứ.
        end_s = held.segment.start_s + held.segment.duration_s
        window_s = self.config.stt.merge_window_ms / 1000
        if self.chunker.stream_s - end_s < window_s:
            return
        if self.chunker.is_active:
            # Người ta ĐANG nói tiếp. Bỏ chờ lúc này là cắt ngay giữa đoạn nói
            # tiếp — đúng thứ cả cơ chế này sinh ra để tránh. Chờ hết câu đã.
            return
        if self._pipeline_pending():
            # Đoạn nói tiếp vừa chốt endpoint và đang xếp hàng chờ nghe. Bỏ
            # chờ lúc này là thua cuộc đua: câu dở bị dịch riêng ngay trước
            # khi chặng nghe kịp ghép nó vào. Đã đo thấy đúng như vậy.
            return
        self._cancel_hold_timer()
        self._held = None
        utterance = self.state.get(held.utterance_id)
        if utterance is None or utterance.is_terminal:
            return
        logger.info(
            "%s: %s không ai nói tiếp — dịch nguyên câu dở",
            self.session_id, held.utterance_id,
        )
        await self._after_transcript(
            held.segment, held.transcript, held.utterance_id, utterance
        )

    def _pipeline_pending(self) -> bool:
        """Còn câu nào đang chờ nghe hoặc đang nghe dở không."""
        if self._pipeline_queue.qsize() > 0:
            return True
        task = self._pipeline_task
        return task is not None and not task.done()

    def _cancel_hold_timer(self) -> None:
        if self._hold_timer is not None and not self._hold_timer.done():
            self._hold_timer.cancel()
        self._hold_timer = None

    def _retire(self, utterance_id: str) -> None:
        """Đóng utterance của mảnh câu đã được ghép vào câu sau."""
        with contextlib.suppress(Exception):
            self.state.transition(utterance_id, UtteranceState.DONE)

    async def _flush_hold_after(self, utterance_id: str, delay_s: float) -> None:
        """Backstop: client ngừng gửi audio hẳn thì đồng hồ audio đứng yên.

        Người ta có quyền bỏ lửng câu. Giữ mãi thì mất hẳn câu đó.
        """
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            return
        held = self._held
        if held is None or held.utterance_id != utterance_id:
            return
        self._held = None
        self._hold_timer = None
        utterance = self.state.get(utterance_id)
        if utterance is None or utterance.is_terminal:
            return
        logger.info("%s: %s hết giờ chờ — dịch nguyên câu dở", self.session_id, utterance_id)
        await self._after_transcript(held.segment, held.transcript, utterance_id, utterance)

    async def _after_transcript(
        self, segment: AudioSegment, transcript, utterance_id: str, utterance
    ) -> None:
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

        # Hết chặng NGHE. Chuyển sang hàng đợi DỊCH và quay lại nghe câu kế
        # tiếp ngay — không đứng chờ LLM.
        target_language = (
            self._counterpart_language
            if direction.is_outbound
            else self.config.session.user_language
        )
        # Loại chính câu này khỏi lịch sử: nó đã nằm ở phần "Now they said"
        # của prompt, đưa vào hai lần sẽ khiến model tưởng bị nói lặp.
        #
        # Chốt lịch sử NGAY BÂY GIỜ chứ không lúc dịch: chặng nghe chạy trước
        # chặng dịch, nên nếu render lúc dịch thì prompt của câu này sẽ chứa
        # cả những câu được nói SAU nó.
        context = (
            self.history.render(exclude=utterance_id)
            if self.config.llm.history_turns > 0
            else ""
        )
        self._llm_queue.put_nowait(
            _LlmItem(
                utterance_id=utterance_id,
                transcript=transcript,
                direction=direction,
                target_language=target_language,
                counterpart_language=self._counterpart_language,
                context=context,
            )
        )

    async def _llm_worker_loop(self) -> None:
        """Chặng 2: dịch. Chạy song song với chặng nghe."""
        while True:
            item = await self._llm_queue.get()
            try:
                if item is None:
                    return
                task = asyncio.create_task(
                    self._respond(item), name=f"llm-{item.utterance_id}"
                )
                self._llm_task = task
                try:
                    await asyncio.wait({task})
                except asyncio.CancelledError:
                    task.cancel()
                    raise
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s: llm worker lỗi", self.session_id)
            finally:
                self._llm_queue.task_done()

    async def _respond(self, item: _LlmItem) -> None:
        utterance_id = item.utterance_id
        utterance = self.state.get(utterance_id)
        if utterance is None or utterance.is_terminal:
            return
        transcript = item.transcript
        direction = item.direction
        target_language = item.target_language
        context = item.context

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

        parser = SemanticEventParser()
        stats = GenerationStats()
        splitter = SentenceSplitter(self.config.tts.min_sentence_chars)
        prompt = self.runtime.llm.build_prompt(
            transcript.text,
            transcript.language,
            direction=direction,
            counterpart_language=item.counterpart_language,
            history=context,
        )

        async with self.runtime.job("llm"):
            token_stream = self.runtime.llm.stream(prompt, stats=stats)
            async for token in token_stream:
                for event in parser.feed(token):
                    await self._emit_semantic(utterance_id, event, utterance, splitter, item)

        for event in parser.finish():
            await self._emit_semantic(utterance_id, event, utterance, splitter, item)

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
                parser = await self._retranslate(item, utterance, reason, stats)

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
            self._speak_translation(utterance_id, remainder, item)

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
        self, item: _LlmItem, utterance, reason, stats,
    ) -> SemanticEventParser:
        """Dịch lại với lời nhắc cứng hơn. Trả về parser của lần dịch mới.

        TTS của lần đầu (nếu có) đã bị hủy trước khi đọc bản hỏng — không để
        người dùng nghe câu tiếng mình đọc bằng giọng nước ngoài.
        """
        from ..ai.llm import language_name

        utterance_id = item.utterance_id
        if self.tts is not None and self.tts.is_active:
            self._drain_tts_queue()
            await self.tts.cancel(reason="bad_translation")

        prompt = self.runtime.llm.build_prompt(
            item.transcript.text,
            item.transcript.language,
            direction=item.direction,
            counterpart_language=item.counterpart_language,
            history=item.context,
            retry_hint=retry_hint(reason, language_name(item.target_language)),
        )
        parser = SemanticEventParser()
        splitter = SentenceSplitter(self.config.tts.min_sentence_chars)
        async with self.runtime.job("llm_retry"):
            async for token in self.runtime.llm.stream(prompt, stats=stats):
                for event in parser.feed(token):
                    await self._emit_semantic(utterance_id, event, utterance, splitter, item)
        for event in parser.finish():
            await self._emit_semantic(utterance_id, event, utterance, splitter, item)

        remainder = splitter.flush()
        if remainder and self._should_speak("translation"):
            self._speak_translation(utterance_id, remainder, item)
        return parser

    async def _emit_semantic(
        self, utterance_id, event, utterance, splitter, item: _LlmItem
    ) -> None:
        """`item` mang chiều dịch của ĐÚNG câu này.

        Không được đọc `self._last_direction`: chặng nghe chạy trước chặng dịch
        nên biến đó đã thuộc về một câu nói sau — giọng đọc và tốc độ sẽ sai.
        """
        if not isinstance(event, TranslationDelta):
            return

        direction = item.direction
        target = item.target_language
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
                self._speak_translation(utterance_id, sentence, item)

    def _speak_translation(self, utterance_id: str, text: str, item: _LlmItem) -> None:
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

        # Chiều của ĐÚNG câu này, không phải chiều của câu đang nghe dở.
        if item.direction.is_outbound:
            # Người dùng phải NÓI THEO, không chỉ nghe hiểu — đọc chậm hẳn và
            # bằng giọng của tiếng đối phương.
            self._speak(
                utterance_id,
                text,
                field="coach",
                voice=self._voice_for(item.counterpart_language),
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
        self._trim_read_backlog(manual)
        self._tts_queue.put_nowait(
            _TtsItem(
                utterance_id, text, field,
                manual=manual, voice=voice, length_scale=length_scale,
            )
        )

    def _trim_read_backlog(self, manual: bool) -> None:
        """Chặn hàng đợi đọc phình vô hạn khi người ta nói nhanh hơn máy đọc.

        Từ khi bỏ cắt lời (xem TtsConfig.barge_in), mọi bản dịch đều được đọc
        trọn — đúng thứ người dùng cần. Nhưng đọc một câu mất ~2.5-3s trong khi
        người nói nhanh có thể ra câu mới mỗi ~3s: nếu đọc chậm hơn nhịp nói
        thì phần đọc tụt lại mãi và cuối cùng người dùng nghe bản dịch của
        chuyện xảy ra một phút trước.

        Vượt hạn thì bỏ bản CŨ NHẤT chứ không bỏ bản mới: bản cũ đã lạc hậu so
        với cuộc nói chuyện. Và phải BÁO RA — mất một câu trong im lặng còn tệ
        hơn chính việc mất câu.

        Yêu cầu đọc thủ công (client tự điều phối) không bị cắt: lúc đó client
        đang chờ đúng lượt đọc đó, bỏ đi là nó treo.
        """
        if manual:
            return
        limit = self.config.tts.max_pending_reads
        while self._tts_queue.qsize() > limit:
            try:
                stale = self._tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if stale is None or stale.is_end or stale.manual:
                # Không phải mẩu chữ để đọc -> trả lại, đừng làm hỏng trạng thái.
                self._tts_queue.put_nowait(stale)
                return
            logger.warning(
                "%s: đọc không kịp nhịp nói, bỏ bản dịch cũ của %s",
                self.session_id, stale.utterance_id,
            )
            self._loop.create_task(
                self.bus.emit(
                    EventType.UTTERANCE_DROPPED,
                    utterance_id=stale.utterance_id,
                    data={"reason": "read_backlog", "pending": limit},
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
                  "sample_rate": self.tts.sample_rate},
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
                          "sample_rate": self.tts.sample_rate},
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
                  "synthesis_ms": round((_time.monotonic() - started) * 1000.0, 2),
                  "prewarmed": self.tts.used_standby},
        )

        # Hâm nóng sẵn cho lượt sau, ngay lúc này — vừa đọc xong thì thường
        # còn cả vài giây trước khi có câu tiếp theo, dùng để nạp model thay vì
        # để lượt sau phải chờ. Đo thật: bớt được ~500ms mỗi câu.
        self.tts.prewarm(voice, length_scale)

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
            #
            # Gắn vào ĐÚNG câu client yêu cầu, không phải `state.current`:
            # chặng nghe chạy trước chặng dịch nên `state.current` thường đã là
            # một câu nói sau. Client dựa vào `utterance_id` của `tts_done` để
            # biết lượt đọc nào vừa xong.
            utterance = (
                self.state.get(control.utterance_id)
                if control.utterance_id
                else self.state.current
            )
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
            if self.tts is not None:
                await self.tts.cancel(reason="reset")
            self._drain_tts_queue()
            self._abort_in_flight()
            self.chunker.reset()
            self.history.clear()
            self.state.cancel_all_active()

    # -- dọn dẹp ---------------------------------------------------------- #

    async def _cancel_all_work(self) -> None:
        with contextlib.suppress(Exception):
            await self.speakers_embedder.close()
        if self.tts is not None:
            await self.tts.cancel(reason="session_end")
        self._drain_tts_queue()
        self._drain_pipeline_queue()
        self._cancel_hold_timer()
        for task in (self._pipeline_worker, self._llm_worker, self._pipeline_task,
                     self._llm_task, self._partial_task, self._tts_current,
                     self._tts_worker, self._hold_timer):
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
            # Đóng bus NGAY: không còn ai rút event ra, để các worker cứ phát
            # tiếp thì hàng đợi đầy và chúng chặn lại trên `put`.
            await self.bus.close()
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
