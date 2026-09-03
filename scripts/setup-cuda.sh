#!/usr/bin/env bash
# Dựng môi trường build NVIDIA/CUDA.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Kiểm tra NVIDIA (Task A1)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "LỖI: không có nvidia-smi. Build này cần GPU NVIDIA + driver CUDA." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
if (( TOTAL_MIB < 6000 )); then
  echo "CẢNH BÁO: GPU chỉ có ${TOTAL_MIB}MiB — đặc tả nhắm 6GB." >&2
fi

echo
echo "==> Môi trường Python"
python3 -m venv .venv 2>/dev/null || true
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements/cuda.txt -r requirements/dev.txt

echo
echo "==> Model"
bash scripts/download_models.sh

cat <<'MSG'

Xong. Bước tiếp theo — CHẠY BENCHMARK GATE TRƯỚC KHI LÀM GÌ KHÁC (§6):

    make gate

Nếu FAIL: dừng, tối ưu model/config, đo lại. Chưa code tiếp Pipeline/Frontend.
Nếu PASS: make run   rồi mở http://localhost:8000
MSG
