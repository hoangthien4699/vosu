"""Nhận ra ĐỔI NGƯỜI NÓI, để cắt đúng câu của từng người.

VÌ SAO CẦN: độ dài khoảng lặng KHÔNG tách được "ngập ngừng giữa câu" với "hết
câu". Đo thật: khoảng ngập ngừng của người nói chậm (800ms) còn DÀI HƠN khoảng
nghỉ giữa hai câu của người nói bình thường (700ms). Không ngưỡng thời gian
nào đúng cho cả hai.

Đổi người nói thì khác — nó là mốc CHẮC CHẮN. A nói xong thì B mới đáp, nên
hễ giọng đổi là câu trước đã kết thúc, bất kể khoảng lặng dài ngắn ra sao.

ĐO ĐƯỢC (benchmarks/speaker_sep.py, 20 đoạn giọng máy):

    cả 4 giọng, một ngưỡng cố định        95.3%
    hai giọng — bài toán thật          96.4% - 100%   (4/6 cặp đạt 100%)

Phạm vi: giọng máy, không phải người thật trong phòng có nhiễu. Vector đến từ
`speaker_encoder.onnx` của VieNeu, vốn là bộ mã hoá cho NHÂN BẢN GIỌNG chứ
không phải bộ xác minh người nói — nó tách được, nhưng không phải công cụ
chuyên dụng. Cả hai cặp bị lẫn đều là nữ với nữ.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _Speaker:
    id: str
    centroid: np.ndarray
    count: int = 1


@dataclass
class SpeakerTracker:
    """Gom các đoạn nói thành người nói, trực tuyến.

    KHÔNG cần ghi danh trước: người đầu tiên nói thành `spk_1`, giọng nào đủ
    khác thì thành `spk_2`. Sản phẩm chỉ cần biết GIỌNG CÓ ĐỔI KHÔNG, không
    cần biết tên ai.
    """

    #: Trên ngưỡng này thì coi là cùng người.
    same_threshold: float = 0.78
    #: Dưới ngưỡng này thì coi là người khác. Ở GIỮA hai ngưỡng là vùng KHÔNG
    #: CHẮC — trả về None thay vì đoán bừa. Đoán sai theo hướng "người khác"
    #: sẽ cắt đôi một câu đang nói dở, tệ hơn là không quyết.
    diff_threshold: float = 0.62
    max_speakers: int = 4
    _speakers: list[_Speaker] = field(default_factory=list)

    def reset(self) -> None:
        self._speakers.clear()

    @property
    def known(self) -> int:
        return len(self._speakers)

    def assign(self, vector: np.ndarray) -> str | None:
        """Người nói của đoạn này, hoặc None nếu không đủ chắc.

        Cập nhật tâm cụm luôn: giọng một người thay đổi theo âm lượng, khoảng
        cách tới mic và cảm xúc, nên tâm cụm phải trôi theo.
        """
        vec = _normalise(vector)
        if vec is None:
            return None

        if not self._speakers:
            return self._add(vec)

        scores = [float(s.centroid @ vec) for s in self._speakers]
        best = int(np.argmax(scores))
        score = scores[best]

        if score >= self.same_threshold:
            self._update(self._speakers[best], vec)
            return self._speakers[best].id
        if score >= self.diff_threshold:
            return None                      # vùng không chắc — đừng quyết
        if len(self._speakers) >= self.max_speakers:
            # Hết chỗ: gán về người gần nhất còn hơn tạo người mới vô tận.
            self._update(self._speakers[best], vec)
            return self._speakers[best].id
        return self._add(vec)

    # -- nội bộ ----------------------------------------------------------- #

    def _add(self, vec: np.ndarray) -> str:
        spk = _Speaker(id=f"spk_{len(self._speakers) + 1}", centroid=vec)
        self._speakers.append(spk)
        logger.info("Nghe thấy giọng mới: %s (tổng %d)", spk.id, len(self._speakers))
        return spk.id

    @staticmethod
    def _update(spk: _Speaker, vec: np.ndarray) -> None:
        spk.count += 1
        moved = spk.centroid + (vec - spk.centroid) / spk.count
        spk.centroid = _normalise(moved)


def _normalise(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    vec = np.asarray(vector, dtype=np.float32).ravel()
    if vec.size == 0:
        return None
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return None
    return vec / norm


class SpeakerEmbedder:
    """Client của tiến trình phụ trích vector giọng.

    Tiến trình RIÊNG với tiến trình TTS: cả hai dùng gói `vieneu`, nhưng chúng
    chạy chồng nhau — TTS đang đọc bản dịch câu N thì câu N+1 vừa dứt lời và
    cần trích vector. Dùng chung thì xếp hàng chờ nhau, cộng thẳng vào độ trễ.

    Đo trên máy Mac: lần trích đầu ~4s (onnxruntime dựng phiên), các lần sau
    ~440ms. Tiến trình phụ tự làm nóng trước khi báo READY, nên câu đầu tiên
    của người dùng không phải gánh.
    """

    def __init__(self, config) -> None:
        self._config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._broken = False

    @property
    def available(self) -> bool:
        return not self._broken

    async def start(self) -> None:
        """Khởi động ở NỀN. Hỏng thì tắt tính năng, không làm sập pipeline."""
        try:
            await self._ensure()
        except Exception as exc:                              # noqa: BLE001
            self._broken = True
            logger.warning("Không dùng được nhận dạng giọng (%s) — bỏ qua", exc)

    async def _ensure(self) -> asyncio.subprocess.Process:
        if self._proc is not None and self._proc.returncode is None:
            return self._proc
        interpreter = shutil.which(self._config.paths.vieneu_python) or (
            self._config.paths.vieneu_python
        )
        script = Path(self._config.paths.speaker_sidecar)
        if not script.is_absolute():
            from ..core.config import REPO_ROOT

            script = REPO_ROOT / script
        if not Path(interpreter).exists() and shutil.which(interpreter) is None:
            raise RuntimeError(f"không có Python của VieNeu: {interpreter}")
        if not script.exists():
            raise RuntimeError(f"không có script: {script}")

        started = time.perf_counter()
        self._proc = await asyncio.create_subprocess_exec(
            interpreter, str(script),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, close_fds=False,
        )
        # Đo thật: sẵn sàng sau ~9.4s. 60s là dư, nhưng phải CÓ hạn — treo ở
        # đây là treo luôn câu đầu tiên.
        line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=60.0)
        if line.strip() != b"READY":
            raise RuntimeError(f"tiến trình phụ không khởi động được: {line!r}")
        logger.info("Nhận dạng giọng sẵn sàng sau %.0fms",
                    (time.perf_counter() - started) * 1000)
        return self._proc

    async def embed(self, pcm: np.ndarray) -> np.ndarray | None:
        """Vector giọng của đoạn này. None nếu không lấy được.

        Hỏng thì trả None chứ không ném: đây là tín hiệu PHỤ để cắt câu cho
        khéo hơn, mất nó thì hệ thống quay về cách cũ chứ không được chết.
        """
        if self._broken:
            return None
        try:
            async with self._lock:
                proc = await self._ensure()
                raw = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                proc.stdin.write(b"EMBED %d\n" % len(raw) + raw)
                await proc.stdin.drain()
                head = await proc.stdout.readline()
                if head.startswith(b"ERR") or not head.startswith(b"VEC "):
                    logger.warning("Trích vector giọng hỏng: %r", head[:80])
                    return None
                n = int(head.split()[1])
                body = await proc.stdout.readexactly(n * 4)
                return np.frombuffer(body, dtype="<f4")
        except Exception as exc:                              # noqa: BLE001
            logger.warning("Trích vector giọng hỏng (%s) — tắt tính năng", exc)
            self._broken = True
            return None

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(Exception):
            proc.stdin.close()
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=3.0)
