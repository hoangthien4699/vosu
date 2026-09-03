# Nhánh `release-mac` — build macOS (Apple Silicon)

Nhánh triển khai cho Mac. Dùng để **phát triển và kiểm thử logic pipeline**.

> **Không dùng nhánh này để chốt Benchmark Gate.** Đặc tả §6 yêu cầu GPU
> NVIDIA 6GB. `benchmarks/run_gate.py` in cảnh báo đậm nếu bạn thử. Build
> để chốt Gate nằm ở nhánh `release`.

## Khác gì so với `main`

Chỉ ba file. Code Python **giống hệt** `main` và `release`:

| File | Khác biệt |
|---|---|
| `config.yaml` | Được commit, ghim `platform: macos`, `whisper_model: base`, `enable_partial: false` |
| `.gitignore` | Không còn bỏ qua `config.yaml`; cấu hình cục bộ chuyển sang `config.local.yaml` |
| `BUILD.md` | File này |

## Vì sao STT chạy CPU

CTranslate2 — backend của faster-whisper — **không có backend Metal/MPS**.
Đây là giới hạn thư viện, không phải thiếu sót cấu hình. LLM thì có Metal
qua llama.cpp và chạy tốt.

Hệ quả trực tiếp lên cấu hình nhánh này:

| | `release` (CUDA) | `release-mac` | Vì sao |
|---|---|---|---|
| STT | `cuda` / `int8_float16` | `cpu` / `int8` | CTranslate2 không có Metal |
| Whisper | `small` | `base` | STT trên CPU đắt; base ~640ms trên M4 |
| Partial STT | bật | **tắt** | Nhường toàn bộ CPU cho final STT |
| LLM | `-ngl 36` | `-ngl 99` | Metal offload toàn bộ layer |
| VRAM ceiling | 5.5GB, enforce | không áp dụng | Unified memory (18GB trên M4) |

LLM model thì **giống nhau ở cả hai nhánh**: Gemma 3 4B với `--swa-full`.
Chênh 451 MB so với Qwen là vấn đề của build CUDA (sát trần 5.5GB), không phải
của Mac.

Tắt partial STT không làm mất câu nào: final STT do **VAD endpoint** kích
hoạt, không phụ thuộc partial (§2.2). Partial chỉ là phản hồi sớm cho câu dài.
Bật lại trong `config.yaml` nếu muốn thấy transcript hiện dần.

## Yêu cầu hệ thống

```bash
brew install llama.cpp portaudio
```

Piper **không có** formula Homebrew (`brew install piper` sẽ gợi ý `piphero`).
Nó được cài qua pip trong `requirements/macos.txt`, và `config.yaml` trỏ thẳng
vào `.venv/bin/piper` để `make run` chạy được mà không cần activate venv.

Đừng dùng `docker compose` trên máy này: Docker Desktop không cho container
truy cập Metal, nên cả STT lẫn LLM đều âm thầm rơi về CPU.

## Cài và chạy

```bash
bash scripts/setup-macos.sh        # brew deps + pip deps + tải model
bash scripts/make_test_audio.sh    # audio mẫu giọng thật cho benchmark
make run                           # http://localhost:8000
```

## Trạng thái đã kiểm chứng trên macOS

Toàn bộ pipeline chạy thật đầu-cuối: Silero VAD v5, Faster-Whisper, Qwen2.5-3B
trên Metal, Piper TTS, WebSocket, Barge-in.

| Phép đo | Đo được | Target | |
|---|---|---|---|
| B2 LLM TTFT | 80ms | 200ms | ĐẠT |
| B6 event-loop lag P95 dưới tải | 2.6ms | 50ms | ĐẠT |
| B8 Barge-in | < 0.1ms | 200ms | ĐẠT |
| B1 STT riêng lẻ | 644ms | 400ms | không |
| B2 LLM tổng sinh | 962ms | 500ms | không |
| B3 TTS time-to-first-audio | 560ms | 400ms | không |
| B5 E2E P95 | 3796ms | 1500ms | không |

Số E2E ở trên đo với Qwen2.5-3B trước khi đổi model. Gemma sinh chậm hơn ~33%
nhưng **TTFT gần như bằng nhau** (91 so với 85ms), và E2E "first useful result"
phụ thuộc TTFT chứ không phải tổng thời gian sinh — `translation` là trường đầu
tiên trong JSON nên nó xuất hiện sớm bất kể phần còn lại sinh xong lúc nào.

Các mục không đạt đều là latency thô, đúng như dự đoán cho STT chạy CPU và
`piper-tts` bản Python. Phần **kiến trúc** thì đạt: event-loop lag giữ 2.6ms
trong khi cả ba engine chạy ở 74% CPU, và Barge-in hủy thật 10/10 lần.

```bash
# chạy benchmark trên máy này (sẽ có cảnh báo "không phải build CUDA")
.venv/bin/python -m benchmarks.run_gate --skip-interactive
```

## Nhận thay đổi từ `main`

```bash
git checkout release-mac
git merge main
```

Code không phân kỳ nên merge gần như không bao giờ xung đột. Nếu có, gần
chắc chắn là ở `config.yaml` — giữ bản của nhánh này.
