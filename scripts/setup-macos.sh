#!/usr/bin/env bash
# Dựng môi trường build macOS (Apple Silicon).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Script này dành cho macOS. Trên Linux/NVIDIA dùng scripts/setup-cuda.sh" >&2
  exit 1
fi

cat <<'NOTE'
==> LƯU Ý VỀ BUILD MACOS

  LLM chạy Metal (nhanh). Nhưng STT chạy CPU: CTranslate2 — backend của
  faster-whisper — KHÔNG có backend Metal/MPS. Latency STT sẽ cao hơn
  build CUDA đáng kể.

  Build này dùng để PHÁT TRIỂN và KIỂM THỬ LOGIC pipeline.
  Đặc tả §6 yêu cầu chốt Benchmark Gate trên GPU NVIDIA 6GB.

NOTE

echo "==> Phụ thuộc hệ thống"
if ! command -v brew >/dev/null 2>&1; then
  echo "LỖI: cần Homebrew — https://brew.sh" >&2
  exit 1
fi
# Homebrew KHÔNG có formula `piper` (gõ `brew install piper` sẽ ra `piphero`).
# Piper cài qua pip ở bước Python bên dưới.
for pkg in llama.cpp portaudio; do
  if brew list "$pkg" >/dev/null 2>&1; then
    echo "  đã có: $pkg"
  else
    echo "  cài: $pkg"
    brew install "$pkg" || echo "  (bỏ qua $pkg — cài tay nếu cần)"
  fi
done

echo
echo "==> Môi trường Python"
python3 -m venv .venv 2>/dev/null || true
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements/macos.txt -r requirements/dev.txt
.venv/bin/pip install piper-tts -q || echo "  (piper-tts không cài được — TTS sẽ tắt)"
.venv/bin/pip install sounddevice -q || echo "  (sounddevice không cài được — B7/B9/B10 sẽ SKIP)"

echo
echo "==> Model"
bash scripts/download_models.sh

cat <<'MSG'

Xong. Chạy thử:

    make test                              # unit test, không cần model
    make run                               # rồi mở http://localhost:8000
    .venv/bin/python -m benchmarks.run_gate --skip-interactive

MSG
