#!/usr/bin/env bash
# Dựng venv RIÊNG cho VieNeu-TTS.
#
# Vì sao riêng: gói `vieneu` phụ thuộc CỨNG vào gradio và librosa. Cài chung
# vào venv của server sẽ nâng FastAPI 0.115.6 -> 0.141.1 và thêm 53 gói.
# Server nói chuyện với nó qua scripts/vieneu_sidecar.py, giống cách đang chạy
# llama-server và piper.
#
# Model (~240MB bản int8 ONNX) tự tải về cache HuggingFace ở lần chạy đầu.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv-vieneu"
if [[ -x "$VENV/bin/python" ]]; then
  echo "==> $VENV đã có"
else
  echo "==> Tạo $VENV"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi

echo "==> Cài vieneu"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" vieneu
else
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install vieneu
fi

echo "==> Kiểm tiến trình phụ (lần đầu sẽ tải model, có thể lâu)"
echo '{"text":"Xin chào.","voice":"Ngọc Huyền"}' \
  | "$VENV/bin/python" scripts/vieneu_sidecar.py 2>/dev/null \
  | head -c 32 | grep -q READY && echo "    OK" || { echo "    HỎNG"; exit 1; }

echo
echo "Xong. Bật bằng cách đặt trong config.yaml:"
echo "    tts:"
echo "      engine: vieneu"
