#!/usr/bin/env bash
# Tải model cho vosu. Chạy được trên cả macOS và Linux.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS/piper"

have() { command -v "$1" >/dev/null 2>&1; }

fetch() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    echo "  đã có: $(basename "$dest")"
    return
  fi
  echo "  tải: $(basename "$dest")"
  if have curl; then curl -fL --progress-bar "$url" -o "$dest.part"
  elif have wget; then wget -q --show-progress "$url" -O "$dest.part"
  else echo "Cần curl hoặc wget." >&2; exit 1
  fi
  mv "$dest.part" "$dest"
}

echo "==> Silero VAD"
fetch "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
      "$MODELS/silero_vad.onnx"

echo
echo "==> Qwen2.5-3B-Instruct Q4_K_M (~2.0 GB)"
fetch "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" \
      "$MODELS/qwen2.5-3b-instruct-q4_k_m.gguf"

echo
echo "==> Giọng Piper (VI + EN)"
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
fetch "$PIPER_BASE/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx" \
      "$MODELS/piper/vi_VN-vais1000-medium.onnx"
fetch "$PIPER_BASE/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json" \
      "$MODELS/piper/vi_VN-vais1000-medium.onnx.json"
fetch "$PIPER_BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
      "$MODELS/piper/en_US-lessac-medium.onnx"
fetch "$PIPER_BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
      "$MODELS/piper/en_US-lessac-medium.onnx.json"

echo
echo "==> Faster-Whisper"
echo "  Không tải ở đây — thư viện tự tải vào cache HuggingFace ở lần chạy đầu."

echo
echo "==> llama-server"
if [[ -x "$MODELS/llama-server" ]]; then
  echo "  đã có: models/llama-server"
elif have llama-server; then
  ln -sf "$(command -v llama-server)" "$MODELS/llama-server"
  echo "  đã liên kết tới $(command -v llama-server)"
else
  cat <<'MSG'
  CHƯA CÓ. Cần build llama.cpp với backend đúng cho máy này:

    macOS (Metal):
      brew install llama.cpp
      # hoặc build tay:
      git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
      cmake -B build -DGGML_METAL=ON && cmake --build build -j --config Release

    NVIDIA (CUDA):
      git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
      cmake -B build -DGGML_CUDA=ON && cmake --build build -j --config Release

  Rồi: ln -sf /đường/dẫn/llama-server models/llama-server
MSG
fi

echo
echo "==> Piper binary"
if have piper; then
  echo "  đã có: $(command -v piper)"
else
  echo "  CHƯA CÓ.  macOS: brew install piper   |   pip: pip install piper-tts"
fi

echo
echo "Xong. Kiểm tra: python -m benchmarks.run_gate --only B1 B3"
