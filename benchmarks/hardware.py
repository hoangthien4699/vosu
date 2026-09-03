"""I/O âm thanh cho các benchmark phụ thuộc phần cứng (B7, B9, B10).

Ba benchmark này KHÔNG tự động hóa hoàn toàn được: chúng đo hành vi vật lý
(micro thu được gì, loa phát ra sao, Bluetooth định tuyến thế nào). Module này
lo phần thu/phát, còn kịch bản do người vận hành thực hiện theo hướng dẫn.

`sounddevice` là dependency TÙY CHỌN (cần portaudio ở tầng hệ thống). Thiếu nó
thì benchmark báo SKIP với hướng dẫn cài, không làm hỏng cả bộ Gate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

SR = 16_000


class AudioIoUnavailable(RuntimeError):
    pass


def _sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise AudioIoUnavailable(
            f"Không dùng được sounddevice ({exc}).\n"
            "  macOS : brew install portaudio && pip install sounddevice\n"
            "  Linux : sudo apt install libportaudio2 && pip install sounddevice"
        ) from exc
    return sd


@dataclass
class Device:
    index: int
    name: str
    max_input: int
    max_output: int

    @property
    def is_input(self) -> bool:
        return self.max_input > 0

    def looks_bluetooth(self) -> bool:
        lowered = self.name.lower()
        return any(k in lowered for k in ("bluetooth", "airpods", "bt", "hands-free", "hfp"))


def list_devices() -> list[Device]:
    sd = _sounddevice()
    return [
        Device(i, d["name"], d["max_input_channels"], d["max_output_channels"])
        for i, d in enumerate(sd.query_devices())
    ]


def print_devices() -> None:
    try:
        devices = list_devices()
    except AudioIoUnavailable as exc:
        print(exc, file=sys.stderr)
        return
    print("Thiết bị âm thanh:")
    for device in devices:
        kinds = []
        if device.max_input:
            kinds.append(f"in:{device.max_input}")
        if device.max_output:
            kinds.append(f"out:{device.max_output}")
        flag = "  [có vẻ là Bluetooth]" if device.looks_bluetooth() else ""
        print(f"  [{device.index:>2}] {device.name}  ({', '.join(kinds)}){flag}")


def record(seconds: float, device: int | None = None) -> np.ndarray:
    """Thu mono 16kHz float32."""
    sd = _sounddevice()
    frames = int(seconds * SR)
    data = sd.rec(frames, samplerate=SR, channels=1, dtype="float32", device=device)
    sd.wait()
    return data.reshape(-1)


def play(pcm: np.ndarray, samplerate: int = SR, device: int | None = None,
         blocking: bool = True) -> None:
    sd = _sounddevice()
    sd.play(pcm, samplerate=samplerate, device=device, blocking=blocking)


def play_async(pcm: np.ndarray, samplerate: int = SR, device: int | None = None) -> None:
    play(pcm, samplerate, device, blocking=False)


def stop_playback() -> None:
    try:
        _sounddevice().stop()
    except AudioIoUnavailable:
        pass


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    if len(data) % 2:
        data = data[:-1]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def prompt(message: str) -> None:
    """Dừng chờ người vận hành. Bỏ qua nếu chạy không có terminal."""
    if not sys.stdin.isatty():
        print(f"[tự động bỏ qua chờ] {message}")
        return
    input(f"\n>>> {message}\n    Nhấn Enter khi sẵn sàng... ")


def ask(message: str, default: str = "") -> str:
    if not sys.stdin.isatty():
        return default
    answer = input(f"    {message} ").strip()
    return answer or default


def count_speech_segments(pcm: np.ndarray, config) -> int:
    """Số đoạn speech mà VAD nhận diện — nền tảng của false-trigger rate."""
    from app.audio.vad import VadEventType, build_vad

    vad = build_vad(config)
    return sum(1 for e in vad.feed(pcm) if e.type is VadEventType.SPEECH_ENDED)
