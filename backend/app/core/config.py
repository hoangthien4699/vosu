"""Config tập trung (Task A3).

Nguồn cấu hình, ưu tiên tăng dần:
    1. giá trị mặc định trong file này
    2. `config.yaml` (đường dẫn qua VOSU_CONFIG, mặc định ./config.yaml)
    3. biến môi trường `VOSU_*` / file `.env`

Cố ý dùng dataclass thay vì pydantic để module này (và `benchmarks/`) import
được mà không cần cài gì ngoài stdlib — Benchmark Gate phải chạy được trên máy
mới cài chưa đủ dependency của server.
"""

from __future__ import annotations

import json
import os
import types
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from .device import DeviceProfile, resolve_profile

REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Các nhóm cấu hình
# --------------------------------------------------------------------------- #


@dataclass
class PathsConfig:
    models_dir: str = "models"
    # faster-whisper: tên model trên HF hoặc đường dẫn thư mục đã convert.
    whisper_model: str = "small"
    llm_gguf: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    llama_server_bin: str = "models/llama-server"
    piper_bin: str = "piper"
    piper_voice_vi: str = "models/piper/vi_VN-vais1000-medium.onnx"
    piper_voice_en: str = "models/piper/en_US-lessac-medium.onnx"
    silero_vad_onnx: str = "models/silero_vad.onnx"

    def resolve(self, name: str) -> Path:
        raw = Path(getattr(self, name))
        return raw if raw.is_absolute() else (REPO_ROOT / raw)


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    # Silero VAD yêu cầu frame 512 sample @16kHz (32ms).
    frame_samples: int = 512


@dataclass
class VadConfig:
    backend: str = "silero"  # silero | energy (energy = fallback không cần model)
    threshold: float = 0.5
    # §7 — ngưỡng im lặng để chốt câu. 300–500ms; cộng thẳng vào E2E.
    min_silence_ms: int = 400
    min_speech_ms: int = 200
    speech_pad_ms: int = 100
    # Ngưỡng để coi speech probability là "đang suy giảm" (dấu hiệu sắp dứt câu).
    decay_ratio: float = 0.6


@dataclass
class ChunkerConfig:
    """Contract §2.2 (review v4.1).

    `min_partial_window_s` CHỈ là ngưỡng tối thiểu để kích hoạt partial STT.
    Nó KHÔNG phải speech boundary. Final STT luôn được kích hoạt bởi VAD
    endpoint, kể cả với utterance ngắn hơn ngưỡng này ("Yes.", "Okay.").
    """

    min_partial_window_s: float = 1.5
    # Chặn sliding-window quá dày (§2.2 review v3.0 — GPU 6GB sẽ tải 100%).
    partial_cooldown_s: float = 0.8
    max_utterance_s: float = 30.0
    enable_partial: bool = True


@dataclass
class SttConfig:
    beam_size: int = 1
    # Whisper tự nhận diện ngôn ngữ (LID). None = auto.
    language: str | None = None
    vad_filter: bool = False  # VAD đã làm ở tầng audio/, không làm lại trong Whisper
    partial_beam_size: int = 1
    condition_on_previous_text: bool = False


@dataclass
class LlmConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    n_ctx: int = 2048
    n_predict: int = 160
    temperature: float = 0.2
    top_p: float = 0.9
    # Số layer offload lên GPU. None = lấy từ DeviceProfile.
    n_gpu_layers: int | None = None
    startup_timeout_s: float = 90.0
    health_poll_interval_s: float = 0.25
    request_timeout_s: float = 20.0
    continuous_batching: bool = True
    # Định dạng lượt hội thoại: auto | chatml | gemma
    #
    # "auto" suy từ tên file GGUF. Dùng SAI template thì model vẫn sinh chữ,
    # chỉ là chất lượng tệ đi khó truy vết — không có lỗi nào được ném ra.
    prompt_template: str = "auto"
    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class TtsConfig:
    enabled: bool = True
    voice: str = "vi"  # vi | en
    sample_rate: int = 22_050
    chunk_ms: int = 120
    length_scale: float = 1.0
    # §2.4.1 MVP scope — chốt phạm vi đọc tự động.
    auto_read_translation: bool = True
    auto_read_intent: bool = False
    auto_read_replies: bool = False
    # Streaming theo câu/cụm để giảm time-to-first-audio (§2.4).
    stream_by_sentence: bool = True
    min_sentence_chars: int = 12
    # Đẩy chunk theo nhịp thời gian thực thay vì dốc hết ra ngay.
    #
    # Piper tổng hợp XONG cả câu rồi mới xuất PCM, nên không pace thì server
    # đẩy toàn bộ audio sang client trong vài ms. Khi người dùng chen lời,
    # server không còn gì để hủy — toàn bộ đã nằm trong buffer của client, và
    # contract "server hủy được trong <200ms" (§2.4.1) trở thành vô nghĩa.
    #
    # Pace cũng làm giảm lượng audio đã cam kết phát tại thời điểm Barge-in,
    # nên trực tiếp giảm mức TTS lọt ngược vào mic (B7).
    realtime_pacing: bool = True
    # Đẩy trước ngần này để chịu được jitter mạng mà vẫn giữ quyền hủy.
    pacing_lead_ms: int = 250


@dataclass
class VramConfig:
    expected_gb: float = 5.0
    hard_ceiling_gb: float = 5.5
    poll_interval_s: float = 2.0
    # Hành vi khi vượt ceiling: warn | reject | unload
    on_exceed: str = "warn"


@dataclass
class SessionConfig:
    max_concurrent_sessions: int = 1
    idle_timeout_s: float = 180.0  # §8 Giai đoạn 4 — giải phóng VRAM sau 3 phút
    # Backpressure (F5): số chunk audio tối đa xếp hàng trước khi cảnh báo/drop.
    audio_queue_maxsize: int = 256
    event_queue_maxsize: int = 512


@dataclass
class BenchmarkTargets:
    """Target của Benchmark Gate (§6/§7). Dùng chung bởi benchmarks/run_gate.py."""

    vad_endpoint_ms: float = 150.0
    stt_ms: float = 400.0
    llm_ttft_ms: float = 200.0
    llm_total_ms: float = 500.0
    tts_ms: float = 400.0
    vram_gb: float = 5.5
    e2e_p50_ms: float = 1000.0
    e2e_p90_ms: float = 1300.0
    e2e_p95_ms: float = 1500.0
    e2e_max_ms: float = 2000.0
    e2e_error_rate: float = 0.02
    barge_in_ms: float = 200.0
    e2e_utterances: int = 30


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


@dataclass
class Config:
    platform: str | None = None  # None = auto-detect
    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    vram: VramConfig = field(default_factory=VramConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    benchmark: BenchmarkTargets = field(default_factory=BenchmarkTargets)
    server: ServerConfig = field(default_factory=ServerConfig)

    _device: DeviceProfile | None = field(default=None, repr=False, compare=False)

    @property
    def device(self) -> DeviceProfile:
        if self._device is None:
            self._device = resolve_profile(self.platform)
        return self._device

    @property
    def llm_gpu_layers(self) -> int:
        """n_gpu_layers hiệu lực: config thắng, nếu không thì lấy từ device profile."""
        if self.llm.n_gpu_layers is not None:
            return self.llm.n_gpu_layers
        return self.device.llm_gpu_layers

    def to_dict(self) -> dict[str, Any]:
        return _asdict_public(self)


# --------------------------------------------------------------------------- #
# Nạp config
# --------------------------------------------------------------------------- #

#: Biến VOSU_* là ĐIỀU KHIỂN, không phải khóa config — không ánh xạ vào Config.
RESERVED_ENV = frozenset({"VOSU_CONFIG"})

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@cache
def _hints(cls: type) -> dict[str, Any]:
    """Phân giải annotation thành type object thật.

    Bắt buộc: `from __future__ import annotations` khiến `dataclasses.fields().type`
    trả về CHUỖI, nên không thể so sánh trực tiếp với `int`/`bool`/...
    """
    return get_type_hints(cls)


def _coerce(raw: Any, target_type: Any) -> Any:
    """Ép kiểu giá trị thô (str từ env, hoặc scalar từ YAML) theo annotation."""
    if raw is None:
        return None
    # `X | None` cho ra types.UnionType, `Optional[X]` cho ra typing.Union — khác nhau.
    if get_origin(target_type) in (Union, types.UnionType):
        non_none = [t for t in get_args(target_type) if t is not type(None)]
        if len(non_none) == 1:
            target_type = non_none[0]
        else:
            return raw
    if target_type is bool:
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise ValueError(f"không phải boolean: {raw!r}")
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    if target_type is str:
        return str(raw)
    if target_type is list or get_origin(target_type) is list:
        if isinstance(raw, list):
            return raw
        return [p for p in str(raw).split(",") if p]
    return raw


def _apply_mapping(obj: Any, data: dict[str, Any], path: str = "") -> None:
    type_hints = _hints(type(obj))
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if key not in type_hints:
            raise ValueError(f"khóa config không nhận diện được: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_mapping(current, value, path=f"{path}{key}.")
        else:
            setattr(obj, key, _coerce(value, type_hints[key]))


def _apply_env(obj: Any, env: dict[str, str], prefix: str = "VOSU_") -> None:
    """VOSU_LLM__PORT=9090 -> config.llm.port. VOSU_PLATFORM=cuda -> config.platform."""
    for raw_key, raw_value in env.items():
        if not raw_key.startswith(prefix) or raw_key in RESERVED_ENV:
            continue
        parts = raw_key[len(prefix) :].lower().split("__")
        target = obj
        try:
            for part in parts[:-1]:
                target = getattr(target, part)
            leaf = parts[-1]
            hints = _hints(type(target))
            if leaf not in hints:
                raise AttributeError(leaf)
            setattr(target, leaf, _coerce(raw_value, hints[leaf]))
        except (AttributeError, TypeError) as exc:
            raise ValueError(
                f"biến môi trường không ánh xạ được vào config: {raw_key}"
            ) from exc


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".json"}:
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise RuntimeError(
            f"Cần PyYAML để đọc {path}. Cài `pip install pyyaml` "
            "hoặc dùng config dạng .json."
        ) from exc
    return yaml.safe_load(text) or {}


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    env: dict[str, str] | None = None,
) -> Config:
    cfg = Config()

    env = dict(os.environ if env is None else env)
    dotenv = _read_dotenv(REPO_ROOT / ".env")
    for key, value in dotenv.items():
        env.setdefault(key, value)

    candidate = config_path or env.get("VOSU_CONFIG")
    if candidate:
        path = Path(candidate)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file config: {path}")
    else:
        path = REPO_ROOT / "config.yaml"

    if path.exists():
        _apply_mapping(cfg, _load_structured(path))

    _apply_env(cfg, env)
    return cfg


def _asdict_public(obj: Any) -> Any:
    if is_dataclass(obj):
        return {
            f.name: _asdict_public(getattr(obj, f.name))
            for f in fields(obj)
            if not f.name.startswith("_")
        }
    if isinstance(obj, list | tuple):
        return [_asdict_public(v) for v in obj]
    return obj


_cached: Config | None = None


def get_config(reload: bool = False) -> Config:
    global _cached
    if _cached is None or reload:
        _cached = load_config()
    return _cached
