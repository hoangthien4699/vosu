#!/usr/bin/env bash
# Sinh audio mẫu giọng thật cho benchmarks/audio/ (WAV 16kHz mono).
#
# Vì sao cần: Silero VAD được huấn luyện trên giọng người thật và KHÔNG kích
# hoạt với tín hiệu tổng hợp (sóng hài + nhiễu). Dùng audio tổng hợp để đo
# latency thì được, nhưng để kiểm chứng VAD/STT thì phải có giọng thật.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/benchmarks/audio"
mkdir -p "$OUT"

PHRASES=(
  "Yes."
  "Okay, got it."
  "I think we should table this discussion for now."
  "Could you walk me through the pricing structure again please?"
  "Let's circle back on this next week when everyone has the numbers."
  "Is there a major blocker we need to resolve first?"
)

if command -v say >/dev/null 2>&1; then                       # macOS
  i=1
  for phrase in "${PHRASES[@]}"; do
    say -o "$OUT/sample_$(printf %02d $i).wav" --data-format=LEI16@16000 "$phrase"
    echo "  sample_$(printf %02d $i).wav — \"$phrase\""
    i=$((i + 1))
  done
elif command -v espeak-ng >/dev/null 2>&1; then               # Linux
  i=1
  for phrase in "${PHRASES[@]}"; do
    espeak-ng -w "$OUT/sample_$(printf %02d $i).wav" -s 150 "$phrase"
    if command -v ffmpeg >/dev/null 2>&1; then
      ffmpeg -loglevel error -y -i "$OUT/sample_$(printf %02d $i).wav" \
             -ar 16000 -ac 1 "$OUT/tmp.wav" && mv "$OUT/tmp.wav" "$OUT/sample_$(printf %02d $i).wav"
    fi
    echo "  sample_$(printf %02d $i).wav — \"$phrase\""
    i=$((i + 1))
  done
else
  echo "Cần 'say' (macOS) hoặc 'espeak-ng' (Linux)." >&2
  exit 1
fi

cat <<'MSG'

Xong. Lưu ý: giọng TTS dễ nhận diện hơn giọng người thật trong môi trường ồn.
Để chốt Benchmark Gate, thay bằng bản THU THẬT của người nói trong điều kiện
sử dụng thực tế (§6 mục 10, 11).
MSG
