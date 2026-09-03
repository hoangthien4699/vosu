# vosu — AI Conversational Copilot

> **Nhánh `release-mac` — build macOS (Apple Silicon).** Xem [BUILD.md](BUILD.md).
> Không dùng nhánh này để chốt Benchmark Gate — build CUDA ở nhánh `release`.


Trợ lý hội thoại thời gian thực chạy qua tai nghe: nghe người đối diện nói →
bóc băng → dịch + hiểu hàm ý → gợi ý câu trả lời → **đọc gợi ý qua tai nghe**.
Toàn bộ xử lý AI chạy **local/offline**.

> **Trạng thái:** 🟡 Kiến trúc *Conditionally Frozen* sau 5 vòng review.
> Đặc tả đầy đủ ở [`docs/`](docs/). Trước khi tin vào bất kỳ con số latency nào,
> hãy chạy [Benchmark Gate](#benchmark-gate) trên phần cứng thật.

---

## Nhánh

| Nhánh | Dùng cho | Khác biệt so với `main` |
|---|---|---|
| `main` | Phát triển | Tự nhận diện phần cứng |
| `release` | Triển khai NVIDIA/CUDA | `config.yaml` ghim `platform: cuda` |
| `release-mac` | Triển khai macOS (Apple Silicon) | `config.yaml` ghim `platform: macos` |

**Code Python ở ba nhánh giống hệt nhau.** Khác biệt chỉ nằm ở đúng ba file:
`config.yaml`, `BUILD.md`, và một dòng trong `.gitignore`. Đây là chủ ý: toàn
bộ khác biệt phần cứng đã được xử lý trong `DeviceProfile` lúc chạy, nên tách
code theo nhánh sẽ tạo ra hai bản phải bảo trì song song mà không được gì.

Quy trình: sửa trên `main`, rồi `git merge main` vào cả hai nhánh release.
Vì code không phân kỳ nên merge gần như không bao giờ xung đột.

```bash
git checkout release-mac && git merge main    # hoặc release
```

**Chỉ merge một chiều: `main` → release.** Merge ngược, hoặc cherry-pick một
commit có đụng `config.yaml`, sẽ kéo cấu hình của một build sang `main` và phá
vỡ đúng thứ chiến lược này bảo vệ. Nếu phải sửa trên nhánh release rồi đưa lên
`main`, tách làm hai commit: một cho code, một cho `config.yaml` — rồi chỉ
cherry-pick commit code.

CI chặn cả hai kiểu hỏng: `main` không được chứa `config.yaml`/`BUILD.md`, và
hai nhánh release không được phân kỳ file `.py`/`.js`/`.html`/`.css`.

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

Không có ai nói tiếng Anh với bạn để thử? Bấm **"Phát file…"** và chọn một file
trong `benchmarks/audio/` — nó đi qua đúng đường mà micro đi.

**Hướng dẫn chạy thử đầy đủ: [TESTING.md](TESTING.md).**

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

Để đo có nghĩa, đặt file WAV **16kHz mono** vào `benchmarks/audio/`
(`bash scripts/make_test_audio.sh` sinh nhanh một bộ). Không có thì script tự
sinh tín hiệu tổng hợp: số đo latency vẫn hợp lệ, nhưng transcript/WER thì
không — Silero VAD cũng không nhận tín hiệu tổng hợp là speech.

### Số đo tham chiếu trên build macOS (M4, Whisper base, Qwen2.5-3B Metal)

Không phải kết quả Gate — Gate phải chạy trên NVIDIA 6GB. Đây là đường cơ sở
để so sánh, và để thấy phần nào của ngân sách latency đang bị tiêu ở đâu:

| | Đo được | Target | |
|---|---|---|---|
| B1 STT riêng lẻ | 644ms P50 | 400ms | ✗ CPU, không có Metal cho CTranslate2 |
| B2 LLM TTFT | 80ms P50 | 200ms | ✓ |
| B2 LLM tổng sinh | 962ms P50 | 500ms | ✗ |
| B3 TTS time-to-first-audio | 560ms P50 | 400ms | ✗ piper-tts bản Python |
| B5 E2E P95 | 3796ms | 1500ms | ✗ |
| B6 event-loop lag P95 dưới tải | 2.6ms | 50ms | ✓ |
| B8 Barge-in | < 0.1ms | 200ms | ✓ |

### Đã chạy thật đầu-cuối

Với Whisper base, Qwen2.5-3B trên Metal và Piper thật, qua WebSocket thật:

```
  0 session_started
  1 audio_started
  2 stt_final          "I think we should table this discussion for now."
  3 copilot_started
 12 translation_delta  full='Tôi nghĩ '
 24 translation_delta  full='Tôi nghĩ chúng ta nên'
 34 tts_started        translation  ← bắt đầu đọc khi LLM CÒN đang sinh
 35 intent_done        suggest postponing discussion
 36 reply_ready        [0] Agreed, let's discuss later.
 37 reply_ready        [1] Sounds good, let's set a new time.
 38 copilot_done       ttft=31.55ms tokens=43
 83 tts_done           44 chunk PCM
```

Hai điều đáng chú ý: `tts_started` ở seq 34 nghĩa là streaming TTS theo câu
hoạt động — không đợi cả JSON. Và `copilot_done` (38) về trước `tts_done` (83)
rất xa, đúng contract "text không phụ thuộc TTS hoàn tất".

B6 là kết quả đáng giá nhất ở đây: event-loop lag giữ ở 2.6ms P95 trong khi
STT + LLM + TTS chạy đồng thời ở 74% CPU. Đó là kiểm chứng thực tế cho luận
điểm của §2.4 — chạy inference trong worker riêng thật sự giữ được event loop
thông thoáng, chứ không chỉ là "async trên danh nghĩa". TTS chậm đi 60% dưới
tải, vẫn trong ngưỡng.

Điều đáng chú ý ở B5: TTFT vẫn giữ ~102ms trong lúc chồng lấn, nhưng E2E lên
tới 2.3s. Chênh lệch đó không nằm ở LLM — nó nằm ở STT (640ms mỗi câu, chạy
CPU) cộng thời gian xếp hàng khi các utterance chồng lên nhau qua executor một
thread. Trên build CUDA, STT được kỳ vọng xuống dưới 400ms và phần xếp hàng co
lại tương ứng.

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

### Dịch hai chiều, không gợi ý câu trả lời

Sản phẩm nghe **hai người**, và chiều dịch suy từ ngôn ngữ Whisper nhận diện —
không cần bấm nút, vì người dùng đang trong hội thoại thật:

| Ai nói | Làm gì | Đọc thế nào |
|---|---|---|
| **Đối phương** | dịch sang tiếng Việt để **hiểu** | giọng Việt, tốc độ thường |
| **Người dùng** | dịch sang tiếng đối phương để **nói theo** | giọng đối phương, **chậm** |

Ngôn ngữ đối phương lấy theo thực tế nghe được. Khi Whisper không chắc (câu
ngắn như "Yes.", "Ừ."), hệ thống giữ chiều của lượt trước thay vì đoán bừa —
đoán sai chiều thì người dùng nghe bản dịch ngược hướng và không hiểu chuyện gì.

Trường `intent` (§4.4) và `replies` đều đã gỡ. Output LLM còn đúng **một
trường** `translation`, từ ~110 token xuống ~20.

### Model LLM: Gemma 3 4B, không phải Qwen2.5-3B

Đặc tả v4.1.0 chọn Qwen2.5-3B. Qwen rò tiếng Trung vào bản dịch tiếng Việt khi
chạy thật (`"Tôi nghĩ chúng ta nên推迟这次讨论目前。"`); Gemma 3 hỗ trợ đa ngôn
ngữ tốt hơn đáng kể. Đổi model là một dòng trong `config.yaml` — prompt template
là dữ liệu, chọn theo `llm.prompt_template` (`auto` suy từ tên file GGUF).

| | Qwen2.5-3B | Gemma 3 4B |
|---|---|---|
| Template | ChatML | `<start_of_turn>`, **không có vai trò system** |
| Stop token | `<\|im_end\|>` | `<end_of_turn>` |
| Cần `--swa-full` | không | **có** |
| RSS khi nạp | 2008 MB | 2459 MB |

#### `--swa-full` là bắt buộc với Gemma

Gemma 3 dùng sliding-window attention, và llama.cpp **không tái dùng được
prefix cache một phần** với SWA nếu thiếu cờ này — mỗi utterance phải xử lý lại
toàn bộ system prompt.

| | Xử lý prompt mỗi câu |
|---|---|
| không có cờ | 241 token / 610ms |
| `--swa-full` | 15 token / 86ms |

#### GBNF grammar là bắt buộc, không phải tối ưu hóa

JSON Schema chỉ kiểm soát **cấu trúc**, không kiểm soát nội dung *bên trong*
chuỗi. `{`, `}` và dấu ngoặc kép cong đều hợp lệ trong chuỗi JSON, nên model
viết `”}` giữa chuỗi rồi lảm nhảm tiếp — JSON vẫn "hợp lệ", người dùng đọc thấy
rác.

Đo với prompt **có lịch sử hội thoại**, 6 câu:

| | Sạch |
|---|---|
| `json_schema` | **0/6** |
| GBNF cấm ký tự cấu trúc | **6/6** |

Không có lịch sử thì `json_schema` cũng sạch — lịch sử chứa nhiều chuỗi trong
ngoặc kép nên nó mồi cho model sinh đúng cái mẫu gây lỗi. **Test không có lịch
sử sẽ cho kết quả sạch giả.**

> **Cảnh báo VRAM cho build CUDA:** Gemma nặng hơn Qwen 451 MB, mà ngân sách ở
> §3.1 tính cho Qwen và đã sát mép 5.5GB. **Bắt buộc chạy lại B4 trên phần cứng
> NVIDIA** trước khi chốt.

### Bộ nhớ hội thoại

§10 của đặc tả xếp context hẹp là "không phải blocker ở MVP" nhưng nói rõ phải
giải quyết "khi mở rộng sang hội thoại nhiều lượt (rolling summary + short
context)". Đã làm phần đó.

Không có nó, mỗi câu được dịch biệt lập — đại từ không phân giải được và gợi ý
trả lời không bám mạch. Đo thật trên một hội thoại 3 lượt:

| Câu | Không lịch sử | Có lịch sử |
|---|---|---|
| *"What do you think about that?"* | *"I was just considering the options."* | *"I'm a little concerned about the delay."* |
| *"It would give the team more time to test it properly."* | *"Could you elaborate on what 'properly' means?"* | *"That's a valid point, let's discuss the testing plan."* |

Cửa sổ trượt 6 lượt / 1200 ký tự, ghi **cả hai phía**: người dùng nói thật nên
lượt của họ cũng đi qua STT, lịch sử phản ánh đúng hội thoại đã diễn ra.

Vị trí trong prompt quyết định hiệu quả cache:

```
[system prompt]   [lịch sử]      [câu hiện tại]
 cố định           chỉ mọc        đổi mỗi lượt
                   thêm ở cuối
```

Lịch sử chỉ nối thêm ở cuối nên tiền tố của lượt trước vẫn dùng lại được. Đặt
nó sau câu hiện tại, hoặc chèn vào giữa system prompt, sẽ phá cache và đẩy TTFT
lên nhiều lần. Tắt bằng `llm.history_turns: 0`.

### Bảy quyết định đáng biết trước khi sửa code

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

**5. TTS được đẩy ra theo nhịp thời gian thực, không dốc hết một lần.**
Piper tổng hợp xong cả câu rồi mới xuất PCM. Không pace thì server đẩy toàn bộ
audio sang client trong vài mili-giây — và khi người dùng chen lời, server
không còn gì để hủy vì tất cả đã nằm trong buffer của client. Contract "server
hủy được trong <200ms" của §2.4.1 khi đó chỉ đúng trên giấy. `B8` sẽ báo ERROR
nếu phát hiện mình chỉ đang đo những lần hủy no-op.

**6. Tiến trình con phải spawn bằng `posix_spawn`, không được fork.**
faster-whisper kéo theo OpenMP/OpenBLAS, thư viện này cài `pthread_atfork`
handler. `fork()` trong lúc Whisper đang transcribe khiến handler gọi
`pthread_join` lên worker đang tính và treo vĩnh viễn — cả tiến trình chết
đứng, không lỗi, không timeout. Mà "spawn Piper trong lúc Whisper transcribe"
chính là kịch bản thường ngày của pipeline. Vì vậy binary luôn được truyền
bằng đường dẫn tuyệt đối, `close_fds=False`, và không `start_new_session` —
ba điều kiện để CPython chọn `posix_spawn`.

**7. Hệ thống là audio hai chiều đóng vòng, không phải pipeline một chiều.**
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
