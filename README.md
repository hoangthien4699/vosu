# vosu — AI Conversational Copilot

Trợ lý hội thoại thời gian thực chạy qua tai nghe: nghe người đối diện nói →
bóc băng → dịch + hiểu hàm ý → gợi ý câu trả lời → **đọc gợi ý qua tai nghe**.
Toàn bộ xử lý AI chạy **local/offline**.

> **Trạng thái:** 🟡 Kiến trúc *Conditionally Frozen* sau 5 vòng review.
> Đặc tả đầy đủ ở [`docs/`](docs/). Trước khi tin vào bất kỳ con số latency nào,
> hãy chạy [Benchmark Gate](#benchmark-gate) trên phần cứng thật.

---

## Hai build

Một codebase, hai cấu hình phần cứng. Mọi khác biệt đi qua `DeviceProfile`
([`backend/app/core/device.py`](backend/app/core/device.py)) — không nơi nào
khác được hardcode `"cuda"` hay `-ngl 36`.

| | **CUDA (NVIDIA)** | **macOS (Apple Silicon)** |
|---|---|---|
| Mục đích | Build mục tiêu, dùng để **chốt Benchmark Gate** | Phát triển & kiểm thử logic |
| STT | CUDA, `int8_float16` | **CPU, `int8`** |
| LLM | llama.cpp CUDA, `-ngl 36` | llama.cpp **Metal**, `-ngl 99` |
| TTS | Piper, CPU | Piper, CPU |
| VRAM ceiling | 5.5GB, enforce qua `nvidia-smi` | Không áp dụng (unified memory) |

**Vì sao STT chạy CPU trên macOS:** faster-whisper dựa trên CTranslate2, và
CTranslate2 **không có backend Metal/MPS**. Đây không phải thiếu sót cấu hình
mà là giới hạn thư viện. Hệ quả: latency STT trên macOS cao hơn build CUDA
đáng kể, nên **kết quả benchmark trên macOS không dùng để quyết định Gate**.
`run_gate.py` in cảnh báo đậm nếu bạn thử làm vậy.

### Cài đặt

```bash
# NVIDIA/CUDA
bash scripts/setup-cuda.sh

# macOS (Apple Silicon)
bash scripts/setup-macos.sh
```

Cả hai script đều kiểm tra phần cứng, cài Python deps, rồi tải model.
Yêu cầu ở tầng hệ thống (không cài qua pip được):

- **CUDA**: driver NVIDIA + CUDA 12.x + cuDNN 9; `llama-server` build với `-DGGML_CUDA=ON`
- **macOS**: `brew install llama.cpp portaudio` (Piper cài qua pip — Homebrew không có formula `piper`)

### Chạy

```bash
make run                      # http://localhost:8000
docker compose up             # chỉ build CUDA — xem ghi chú trong docker-compose.yml
```

Mở `http://localhost:8000` để dùng web test client: nói → thấy transcript,
bản dịch, hàm ý, gợi ý trả lời → nghe TTS → nói chen vào để thử Barge-in.

---

## Benchmark Gate

Đặc tả quy định **không code Pipeline trước khi Gate PASS** trên phần cứng
thật. Bộ đo đã sẵn sàng:

```bash
make gate                                          # chạy đủ B1-B10
python -m benchmarks.run_gate --skip-interactive   # bỏ B7/B9/B10 (cần người vận hành)
python -m benchmarks.b5_concurrent_e2e -n 50       # chạy riêng một mục
```

| | Phép đo | Target |
|---|---|---|
| B1 | STT riêng lẻ | < 400ms |
| B2 | LLM riêng lẻ | TTFT < 200ms · total < 500ms |
| B3 | TTS riêng lẻ (CPU) | < 400ms |
| B4 | VRAM (Whisper + LLM active) | < 5.5GB |
| **B5** | **Concurrent Inference E2E** | **P50<1.0s · P90<1.3s · P95<1.5s · Max<2.0s · lỗi<2%** |
| B6 | CPU stress (STT+LLM+TTS) | không nghẽn event loop |
| B7 | Audio feedback loop | ghi nhận false-trigger rate |
| B8 | Barge-in response time | < 200ms |
| B9 | Speaker-source validation | ghi nhận WER + false activation |
| B10 | Web mic vs Bluetooth mic | ghi nhận WER/latency |

**B5 là mục quan trọng nhất.** Nó đo từ VAD endpoint tới kết quả copilot hữu
ích *đầu tiên* — không phải tổng cộng dồn STT + LLM. Nếu STT 600ms chồng lấn
với LLM 700ms thì cộng dồn ra 1300ms nhưng E2E thật có thể chỉ ~700ms. Và
**P95 mới là con số cần theo dõi, không phải P50**: một hệ thống chạy 800ms ở
phần lớn trường hợp nhưng cứ 5 câu lại vọt lên 2.5s vẫn không đạt chuẩn realtime.

B7/B9/B10 phụ thuộc phần cứng thật và cần người vận hành theo hướng dẫn trên
màn hình. Chúng không tự PASS/FAIL, nhưng Gate chưa hoàn tất nếu thiếu dữ liệu
của chúng — đặc biệt B9, thứ sẽ quyết định có phải thiết kế lại audio routing
hay không.

Để đo có nghĩa, đặt file WAV **16kHz mono** vào `benchmarks/audio/`. Không có
thì script tự sinh tín hiệu tổng hợp: số đo latency vẫn hợp lệ, nhưng
transcript/WER thì không.

---

## Kiến trúc

```
Audio → VAD → Utterance State → STT (partial/final) → LLM (streaming)
                                                          ↓
                                              LLM Output Parser
                                                          ↓
                                              Semantic Events
                                              ├──────────────→ WebSocket → UI
                                              └──────────────→ TTS (CPU)
                                                                   ↓
                                                              Playback
                                                                   ↓
                                                            Barge-in ──┐
                                                                       │
                     ┌─────────────────────────────────────────────────┘
                     ↓  (đóng vòng lặp — hệ thống audio HAI CHIỀU)
                    VAD
```

```
backend/app/
├── core/      config.py · device.py · runtime.py · vram_manager.py
├── audio/     vad.py · chunker.py · session.py
├── ai/        stt.py · llm.py · json_stream.py · copilot.py · tts.py
├── protocol/  events.py · schemas.py
└── api/       websocket.py
```

### Năm quyết định đáng biết trước khi sửa code

**1. `1.5s` là ngưỡng partial, KHÔNG phải speech boundary.**
Final STT luôn do VAD endpoint kích hoạt, áp dụng cho mọi độ dài câu. Nếu chốt
final theo ngưỡng 1.5s thì câu ngắn ("Yes.", "Okay.") sẽ *không bao giờ* được
transcribe. `tests/test_chunker.py` khóa contract này.

**2. Frontend không bao giờ thấy JSON thô của LLM.**
Token stream và application event là hai abstraction khác nhau. Backend parse
token thành semantic event (`translation_delta`, `intent_done`, `reply_ready`);
frontend chỉ hiểu các event ngữ nghĩa đó. Nhờ vậy đổi model hay đổi prompt
format không làm vỡ client.

**3. Model là hạ tầng dùng chung, session là consumer.**
`shutdown()` chờ mọi *inference job* xong mới hạ model — thứ chặn shutdown là
số job, không phải số session. Client ngắt kết nối khi job còn dang dở là kịch
bản chính mà mô hình `active_clients == 0 → kill model` làm sai.

**4. TTS chạy CPU, và trong process riêng.**
Whisper + Qwen đã chiếm ~5.1GB/6GB. Thêm TTS lên GPU sẽ vượt trần hoặc tạo
tranh chấp SM ba chiều. Và "async" trên danh nghĩa là chưa đủ: gọi hàm blocking
thẳng trong async route vẫn chặn event loop.

**5. Hệ thống là audio hai chiều đóng vòng, không phải pipeline một chiều.**
Khi TTS phát qua earbud, audio output có thể trở thành input của chính hệ
thống. Barge-in không phải tính năng thêm vào — nó là điều kiện để hệ thống
không tự nói chuyện với chính mình.

---

## Phát triển

```bash
make test     # unit test — không cần model, không cần GPU
make lint     # ruff
```

Bộ test cố ý chạy được không cần model: VAD có backend `energy` thay thế,
STT/LLM/TTS đều có bản giả lập. Nếu phải tải 2GB model mới chạy được test thì
sẽ không ai chạy.

Những test đáng đọc trước khi sửa code:

| File | Khóa điều gì |
|---|---|
| `test_chunker.py` | contract 1.5s vs VAD endpoint |
| `test_json_stream.py` | parser chịu được output méo của model 4-bit |
| `test_copilot_parser.py` | không rò JSON thô ra frontend |
| `test_runtime.py` | shutdown chờ inference job |
| `test_tts.py` | Barge-in < 200ms |
| `test_websocket_e2e.py` | thứ tự event + envelope đầy đủ |

---

## Còn chưa giải quyết

Hai rủi ro chỉ mới có kế hoạch đo, chưa có lời giải kỹ thuật:

- **False Activation.** Mic của earbud nằm rất gần miệng *người dùng*, nhưng
  yêu cầu sản phẩm là nghe *người đối diện*. Đây là mismatch vật lý, không phải
  giả định cần lưu ý. B9 sẽ quyết định có phải thiết kế lại audio routing không.
- **Bluetooth duplex.** A2DP vs HFP/SCO khi mic đang active là vấn đề tầng OS,
  ảnh hưởng trực tiếp tới tính khả thi của Barge-in. B10 đo việc này.

Chưa làm cho production: auth cho WebSocket, đa client, fallback khi LLM timeout.

## Tài liệu

| File | Nội dung |
|---|---|
| [`docs/AI-Earbud-Copilot-Tong-Hop-Du-An.md`](docs/AI-Earbud-Copilot-Tong-Hop-Du-An.md) | Bản tổng hợp một trang |
| [`docs/AI-Earbud-Copilot-Tong-Hop.md`](docs/AI-Earbud-Copilot-Tong-Hop.md) | Đặc tả đầy đủ + lịch sử 5 vòng review |
| [`docs/AI-Earbud-Copilot-Ke-Hoach-Trien-Khai.md`](docs/AI-Earbud-Copilot-Ke-Hoach-Trien-Khai.md) | Module breakdown + task backlog |
