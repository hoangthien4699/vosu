# AI Conversational Copilot (Real-time Earbud Assistant)

> **Phiên bản tài liệu:** 4.1.0
> **Trạng thái:** 🟡 **Conditionally Frozen — Awaiting Benchmark Gate.** Kiến trúc đã đủ chi tiết để khóa lại có điều kiện: implementation bị **BLOCKED** cho đến khi Benchmark Gate (mục 7) PASS trên phần cứng thật. Không cần thêm vòng review kiến trúc lớn nào nữa sau bản này — chỉ theo dõi kết quả benchmark để quyết định tiếp tục hay tối ưu model/config.
> **Thay đổi so với v1.0.0:** Chuyển từ mô hình "buffer cố định → transcribe → LLM batch" sang kiến trúc **streaming/incremental**, bổ sung Benchmark Gate trước khi code pipeline/frontend, chuẩn hóa event protocol, tái cấu trúc thư mục theo 3 tầng (Audio / Intelligence / Presentation).
> **Thay đổi so với v2.0.0 (review vòng 2):** 4 điều chỉnh nhỏ về wording và Benchmark Gate (TTFT, first useful reply, MVP assumption về hướng thu âm).
> **Thay đổi so với v2.1.0 (review vòng 3):** bổ sung 3 rủi ro vật lý cấp phần cứng — Compute Contention, Sliding-Window overhead, VAD silence threshold cộng vào E2E.
> **Thay đổi so với v3.0.0 (bổ sung TTS Layer):** thêm Tầng 5 — Text-to-Speech, chạy CPU, tách khỏi critical path E2E.
> **Thay đổi so với v4.0.0 (review vòng 5 — 8 điểm P0/P1 bắt buộc trước khi freeze):** (1) sửa benchmark metric từ "STT+LLM cộng dồn" sang **Concurrent inference E2E theo percentile (P50/P95)**; (2) thêm `utterance_id` vào event protocol; (3) thêm `tts_cancelled`/`tts_error`; (4) thêm **Barge-in state machine** (TTS đang phát → phát hiện speech mới → cancel TTS → resume listening); (5) thêm benchmark test **audio feedback loop** (TTS playback + microphone cùng lúc); (6) LLM streaming chuyển thành **semantic events**, frontend không parse partial JSON thô; (7) Benchmark Gate đổi từ một con số E2E sang **P50/P90/P95/Max + error rate**; (8) sửa mâu thuẫn trạng thái tài liệu — không còn tuyên bố "FROZEN" tuyệt đối, đổi thành "Conditionally Frozen — Awaiting Benchmark Gate".

---

## 1. Mô tả Dự án

### 1.1. Mục tiêu
Xây dựng hệ thống hỗ trợ hội thoại trực tiếp theo thời gian thực (**Real-time Conversational Copilot**), hoạt động qua tai nghe/micro. Hệ thống sẽ:

- Thu nhận âm thanh trực tiếp từ tai nghe/micro của người dùng.
- Bóc băng (Speech-to-Text) và tự động nhận diện ngôn ngữ đang nói.
- Phân tích ngữ cảnh và hàm ý văn hóa của câu nói bằng LLM.
- Đề xuất các câu phản hồi tự nhiên, phù hợp ngữ cảnh cho người dùng lựa chọn nhanh.

### 1.2. Ràng buộc hạ tầng hiện tại
- **100% Offline / Local**, chạy trên GPU **6GB VRAM**.
- Có phương án dự phòng nâng cấp lên **Cloud GPU 24GB VRAM** khi cần mở rộng.

### 1.3. Target Component Latency (trước gọi là "Latency Budget")

> **Ghi chú review (v2.1):** Bảng dưới đây thể hiện **target độ trễ cho từng thành phần** (component-level), không nên hiểu là "Latency Budget" đảm bảo (guaranteed E2E). E2E thực tế còn cộng thêm: speech endpoint detection, STT windowing, LLM queue, serialization, WebSocket overhead... Tiêu chí chấp nhận (acceptance criterion) thực sự là dòng **E2E < 1.5s** trong Benchmark Gate (mục 7), không phải tổng cộng các con số bên dưới.

Mục tiêu phản xạ hội thoại tự nhiên: **tổng độ trễ ≤ 1.5 giây** (E2E, đo thật trong Benchmark Gate).

| Chặng xử lý | Công nghệ | Thời gian xử lý |
|---|---|---|
| Thu âm & Chốt câu | Silero VAD | ~150ms |
| Bóc băng (STT) | Faster-Whisper (beam_size=1) | ~350ms – 450ms |
| Inference LLM | llama-server (Greedy Search, temp=0.2) | ~300ms – 500ms |
| Truyền nhận mạng | WebSocket nội bộ (Localhost) | ~20ms |
| **Tổng độ trễ toàn trình** | | **~820ms – 1.12s** |

> ⚠️ **Ghi chú từ review kiến trúc (v2):** Ngân sách này chỉ khả thi nếu pipeline chạy theo mô hình **streaming/incremental**. Với cách triển khai "buffer 1 giây → transcribe → gọi LLM" ở bản v1, latency thực tế sẽ gần **1.8s** (1000ms buffer + 400ms STT + 400ms LLM), vượt xa mục tiêu. Xem mục **2.1 — Kiến trúc Streaming/Incremental** và mục **9 — Benchmark Gate** bên dưới.

---

## 2. Kiến trúc Tổng thể Hệ thống

Hệ thống hoạt động theo mô hình **Pipelined Streaming qua WebSocket song công (Full-duplex)**.

```text
[Microphone / Earbuds]
        │
        ▼ (PCM 16kHz Streaming qua WebSocket)
┌─────────────────────────────────────────────────────────────┐
│                      BACKEND SERVER                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Session & VRAM Manager                                    │
│    - Kiểm soát vòng đời kết nối                               │
│    - Quản lý cấp phát & giải phóng VRAM (Auto-cleanup)        │
│                                                                │
│ 2. VAD & Audio Chunker                                        │
│    - Silero VAD lọc im lặng & phát hiện dứt câu                │
│                                                                │
│ 3. Local STT Engine (GPU)                                      │
│    - Faster-Whisper (small/medium, int8_float16)               │
│    - Tự động nhận diện mã ngôn ngữ (LID)                       │
│                                                                │
│ 4. Local LLM Engine (GPU)                                      │
│    - llama-server (llama.cpp) chạy Qwen2.5-3B-Instruct          │
│    - Prompt chuyên biệt: dịch + giải thích intent + gợi ý       │
│                                                                │
│ 5. Local TTS Engine (CPU — mới ở v4.0)                          │
│    - Piper TTS (ONNX) hoặc engine CPU nhẹ tương đương            │
│    - Chuyển translation/intent (và reply nếu được chọn) thành   │
│      giọng nói, phát qua tai nghe                               │
└──────────────────────────────┬────────────────────────────────┘
                               │ (JSON Event Stream + Audio Stream)
                               ▼
               [Frontend: Mobile App / Web Client]
               ├── Giao diện hiển thị văn bản gốc & bản dịch
               ├── Thẻ giải thích ngữ cảnh / hàm ý đối phương
               ├── Danh sách gợi ý câu trả lời nhanh (Quick Replies)
               └── Phát audio TTS qua tai nghe (translation/intent tự động,
                   quick reply theo yêu cầu)
```

### 2.1. Thành phần chính
| Thành phần | Vai trò |
|---|---|
| Session & VRAM Manager | Quản lý vòng đời kết nối, cấp phát/giải phóng VRAM tự động |
| VAD & Audio Chunker | Lọc im lặng, phát hiện điểm dứt câu bằng Silero VAD |
| Local STT Engine | Bóc băng bằng Faster-Whisper, tự nhận diện ngôn ngữ |
| Local LLM Engine | Dịch + phân tích hàm ý + sinh gợi ý trả lời bằng Qwen2.5-3B |
| **Local TTS Engine (mới)** | **Chuyển translation/intent (và reply nếu người dùng chọn) thành giọng nói, chạy trên CPU** |
| Frontend | Hiển thị văn bản gốc, bản dịch, giải thích ngữ cảnh, quick replies, và phát audio TTS |

### 2.2. Kiến trúc Streaming / Incremental (thay cho buffer cố định)

Bản v1 dùng buffer cố định (`CHUNK_SIZE = 32000 * 2`, tương đương ~1 giây) trước khi transcribe — đây là **vấn đề kiến trúc realtime cốt lõi**, không phải rủi ro phụ. Kiến trúc đúng nên là:

```text
PCM stream
   ↓
VAD
   ↓
speech started
   ↓
partial audio
   ↓
Pseudo-streaming / Sliding-window STT
   ↓
partial transcript
   ↓
speech ended
   ↓
final transcript
   ↓
LLM (streaming)
```

Timeline ví dụ:

```text
0ms       speech starts
100ms     ├── audio
200ms     ├── audio
300ms     ├── audio
400ms     ├── partial STT
500ms     ├── partial STT
600ms     ├── partial STT
700ms     ├── partial STT
800ms     ├── VAD detects ending
900ms     final STT
          ├── LLM (streaming)
1200ms    reply
```

Chỉ với mô hình này thì mục tiêu 1.2s–1.5s mới có cơ sở thực tế.

> ⚠️ **Ghi chú review (v2.1) — về bản chất "STT incremental":** Faster-Whisper **không phải** một streaming stateful ASR engine theo nghĩa chặt (không giống một số kiến trúc streaming ASR chuyên dụng). Cách khả thi để có partial transcript là:
>
> ```text
> audio window
>     ↓
> transcribe
>     ↓
> partial
>     ↓
> audio window mở rộng
>     ↓
> transcribe lại
>     ↓
> partial update
> ```
>
> Đây là **pseudo-streaming / sliding-window incremental inference**, không phải streaming ASR thật. Không phải blocker, nhưng tài liệu và code cần gọi đúng tên để developer không hiểu nhầm khả năng của Faster-Whisper.

> ⚠️ **Ghi chú review (v3.0) — Overhead của Sliding-Window STT trên GPU 6GB:** Nếu cửa sổ trượt quá dày (ví dụ transcribe lại mỗi 200ms), Faster-Whisper phải liên tục nạp dữ liệu và tính lại encoder/decoder. Với tần suất cao (3–5 lần/giây) trên GPU 6GB, GPU sẽ luôn ở mức tải ~100%, gây nghẽn hàng đợi lệnh CUDA — ảnh hưởng trực tiếp đến latency của cả STT lẫn LLM chạy song song. **Giải pháp cho MVP:** không trượt quá dày — chỉ tích lũy speech chunk qua VAD và **chỉ kích hoạt Whisper khi đoạn nói dài hơn ~1.5 giây hoặc khi VAD phát hiện speech probability bắt đầu suy giảm** (dấu hiệu sắp dứt câu), thay vì transcribe liên tục theo mốc thời gian cố định.

> ⚠️ **Ghi chú review (v4.1) — làm rõ contract: 1.5 giây là minimum trigger, không phải speech boundary:** Nếu chỉ dựa cứng vào "speech > 1.5s mới chạy Whisper", các utterance ngắn (ví dụ chỉ nói "Yes.", "Okay.") sẽ không bao giờ đạt ngưỡng này và có hành vi latency/accuracy rất khác so với câu dài — đây là một trade-off ẩn cần làm rõ trong contract. Contract đúng nên là:
>
> ```text
> VAD speech_start
>         ↓
> accumulate audio
>         ↓
> if minimum inference window reached
>         → optional partial STT
>         ↓
> VAD endpoint (dứt câu thật sự)
>         ↓
> mandatory final STT
> ```
>
> Tức là **1.5s chỉ là ngưỡng tối thiểu để kích hoạt partial inference** (tùy chọn, để có phản hồi sớm cho câu dài), còn **VAD endpoint mới là điều kiện bắt buộc để chạy final STT** — áp dụng cho mọi độ dài câu, kể cả câu ngắn dưới 1.5s.

### 2.3. Kiến trúc 3 tầng (Audio / Intelligence / Presentation)

Thay vì để `websocket_routes.py` gọi thẳng Whisper rồi LLM (coupling chặt), nên tách rõ:

```text
Tầng 1 — Audio                Tầng 2 — Intelligence          Tầng 3 — Presentation
Mic                            Speech                          Events
 ↓                              ↓                               ↓
Audio capture                  STT                             WebSocket
 ↓                              ↓                               ↓
VAD                            Language → Translation          UI
 ↓                              ↓
Speech segment                 Intent → Reply
```

Luồng hệ thống:

```text
WebSocket
   ↓
Session
   ↓
Audio Pipeline
   ↓
Copilot Pipeline
   ↓
Event Bus
   ↓
WebSocket
```

Tách như vậy giúp thay Whisper/Qwen/WebSocket (ví dụ chuyển sang WebRTC) mà không phá vỡ pipeline.

> ⚠️ **Ghi chú (v2.1):** "Event Bus" trong diagram trên **không nên** trở thành một message broker thật (RabbitMQ/Kafka/Redis). Ở MVP, đây chỉ cần là một abstraction nội bộ đơn giản — `asyncio.Queue` hoặc callback/event dispatcher là đủ. Thêm broker thật ở giai đoạn này là over-engineering không cần thiết.

### 2.4. Tầng 5 — Text-to-Speech (mới ở v4.0, mở rộng ở v4.1)

Với form-factor tai nghe, người dùng không thể vừa nghe hội thoại trực tiếp vừa đọc màn hình để tiếp nhận gợi ý — **audio là kênh đầu ra chính**, không phải phụ trợ. Đây là lý do TTS cần được thiết kế như một tầng kiến trúc riêng, không phải tính năng gắn thêm cuối cùng.

> ⚠️ **Ghi chú review (v4.1) — TTS biến hệ thống thành closed-loop audio system:** Mô hình hóa TTS đơn giản là "LLM → UI + LLM → TTS" (một chiều) chưa đủ. Khi TTS thực sự phát qua earbud, **audio output có thể trở thành input của chính hệ thống** — vì micro của earbud có thể thu lại cả tiếng TTS đang phát:
>
> ```text
>              ┌───────────────┐
>              │ Microphone    │
>              └───────┬───────┘
>                      ↓
>                     VAD
>                      ↓
>                     STT
>                      ↓
>                     LLM
>                      ↓
>                     TTS
>                      ↓
>                  Earbuds
>                      │
>                      │
>                      └──────────► Microphone (vòng lặp ngược)
> ```
>
> Đây là kiến trúc **Mic ↔ AI ↔ Earbud** (hai chiều), không còn là **Mic → AI → UI** (một chiều) như các phần khác của hệ thống. Hệ quả trực tiếp: nếu người đối diện nói tiếp trong lúc TTS đang đọc, micro có thể thu lẫn `human speech + TTS playback + room noise`, khiến Whisper transcribe sai.

**Vấn đề tài nguyên:** Hệ thống hiện đã dùng ~5.1–5.5GB/6GB VRAM cho Whisper + Qwen, và review vòng 3 đã xác định rủi ro **Compute Contention** giữa 2 model này khi chạy chồng lấn trên GPU. Thêm một TTS model chạy trên GPU sẽ:
- Gần như chắc chắn vượt trần VRAM 6GB, hoặc
- Tạo ra tranh chấp SM **3 chiều** (Whisper + LLM + TTS) — nghiêm trọng hơn nhiều so với tranh chấp 2 chiều đã cảnh báo ở v3.0.

**Quyết định kiến trúc: chạy TTS trên CPU, không dùng GPU.**

| Lựa chọn | Ưu điểm | Nhược điểm |
|---|---|---|
| TTS trên GPU (cùng card với Whisper/LLM) | Nhanh hơn nếu đủ tài nguyên | Vượt trần VRAM 6GB, tăng compute contention 3 chiều — **không khả thi với ràng buộc hiện tại** |
| **TTS trên CPU (đề xuất)** | Không cạnh tranh VRAM/SM với Whisper & LLM, tận dụng CPU thường dư dả hơn GPU trong bài toán này | Cần chọn engine đủ nhẹ để đạt latency chấp nhận được trên CPU |

**Công nghệ đề xuất:** engine TTS nhẹ, tối ưu cho CPU, dạng ONNX — ví dụ **Piper TTS** (hỗ trợ tiếng Việt và tiếng Anh, độ trễ thấp cho câu ngắn, footprint nhỏ). Không cần model TTS chất lượng studio — mục tiêu là đọc rõ, đủ tự nhiên, không phải lồng tiếng chuyên nghiệp.

> ⚠️ **Ghi chú review (v4.1) — "0GB VRAM" không nên viết như một physical guarantee:** Piper/ONNX chạy CPU vẫn tiêu tốn RAM, memory-mapped model, và runtime buffers — không ảnh hưởng đến 6GB VRAM nhưng không phải "0" theo nghĩa vật lý tuyệt đối. Wording đúng: **"GPU VRAM impact: negligible / excluded — chạy CPU-only"**, thay vì khẳng định "0GB VRAM".

**Latency: tách TTS khỏi critical path của E2E.** Text gợi ý (translation, intent, quick replies) vẫn hiển thị ngay khi LLM streaming xong — đạt target hiện tại mà **không phụ thuộc vào TTS**. TTS chạy như một nhánh xử lý song song, không chặn phần hiển thị text:

```text
LLM streaming (translation/intent/replies)
        │
        ├──► Frontend hiển thị text ngay (đạt E2E hiện tại)
        │
        └──► TTS Engine (CPU, async)
                   │
                   ▼
             Audio stream → phát qua tai nghe
             (có thể trễ hơn text 200-500ms, chấp nhận được)
```

> ⚠️ **Ghi chú review (v4.1) — "TTS async" chưa đủ, cần phân biệt 2 loại contention:**
> - **Event-loop contention:** nếu TTS gọi hàm blocking trong Python trực tiếp trong async route, event loop sẽ bị nghẽn dù được gọi "async" trên danh nghĩa → phải chạy TTS trong worker/thread/process riêng, không gọi blocking call thẳng trong event loop.
> - **CPU contention:** dù TTS không block event loop, nó vẫn cạnh tranh CPU với các thành phần khác (Whisper CPU-side components, Python runtime, network I/O) và có thể làm các task khác chậm đi. Cần benchmark riêng: CPU utilization, CPU saturation, event-loop lag, TTS latency under load, WebSocket jitter (xem mục 7).

**Streaming TTS theo câu/cụm từ:** thay vì đợi toàn bộ JSON (translation + intent + replies) hoàn chỉnh rồi mới tổng hợp giọng nói, nên bắt đầu synthesize ngay khi có đủ một câu/cụm hoàn chỉnh — giảm time-to-first-audio.

### 2.4.1. Barge-in / TTS State Machine (P0 — mới ở v4.1)

> ⚠️ **Ghi chú review (v4.1):** Đây là phần bổ sung quan trọng nhất của v4 — không thể bỏ qua khi TTS thực sự phát qua earbud. Không cần xây AEC (Acoustic Echo Cancellation) phức tạp ngay ở MVP, nhưng **bắt buộc phải có contract rõ ràng**: khi phát hiện speech mới trong lúc TTS đang phát, hệ thống phải có khả năng **cancel/stop TTS hiện tại và quay lại lắng nghe**.

TTS cần một state machine tường minh:

```text
IDLE → SYNTHESIZING → PLAYING → DONE
                          │
                          ▼
                  (speech mới phát hiện)
                          │
                          ▼
                     INTERRUPTED
                          │
                          ▼
                   stop TTS ngay
                          │
                          ▼
                   resume listening
```

Luồng Barge-in cụ thể:

```text
TTS đang PLAYING
      ↓
VAD phát hiện speech mới (người đối diện nói tiếp)
      ↓
BARGE_IN
      ↓
stop TTS playback ngay lập tức
      ↓
resume listening (VAD/STT tiếp tục bình thường)
```

**MVP scope cho contract này:** chưa cần audio echo cancellation thật, chỉ cần cơ chế phát hiện + hủy phát kịp thời để tránh Whisper nhận nhầm audio TTS làm speech input mới.

**MVP scope — chốt phạm vi đọc tự động (cập nhật v4.1, cần thiết để tránh audio dồn dập gây rối tai nghe):**

> ⚠️ **Ghi chú review (v4.1) — sửa lại quyết định "tự động đọc translation + intent" của v4.0:** Nếu người đối diện nói liên tục nhiều câu, mỗi câu đều tự động sinh TTS cho cả translation lẫn intent sẽ nhanh chóng gây **audio overload** — giọng AI chồng lấp lên hội thoại thật đang diễn ra. Intent thường là metadata hỗ trợ người dùng hiểu hơn là thứ cần *nghe* ngay lập tức.
>
> **MVP assumption (v4.1):**
> - **Translation → tự động đọc (AUTO)** — đây là giá trị cốt lõi khi đang nghe live hội thoại.
> - **Intent → hiển thị UI, TTS tùy chọn (optional)** — không tự động đọc mặc định.
> - **Quick reply → chỉ đọc khi người dùng chọn thủ công (manual)**.

**Benchmark target bổ sung cho TTS (thêm vào Benchmark Gate — mục 7):**

| Test | Target |
|---|---|
| TTS synthesis latency (câu ngắn, <10 từ, trên CPU) | < 300–400ms |
| TTS không làm tăng E2E của text suggestion | Text vẫn đạt E2E target, không phụ thuộc TTS hoàn tất |
| CPU utilization khi TTS + STT + LLM cùng chạy | Không gây nghẽn event loop / WebSocket — đo riêng CPU saturation, event-loop lag, TTS latency under load |
| **Audio feedback loop test (P0, mới v4.1)** | **Phát TTS qua loa/tai nghe trong lúc mic đang thu — đo mức độ Whisper nhận nhầm audio TTS thành speech input mới (false trigger rate)** |
| **Barge-in response time (P0, mới v4.1)** | **Từ lúc VAD phát hiện speech mới trong lúc TTS đang PLAYING → TTS ngừng phát: < 200ms** |

**Event protocol bổ sung (mở rộng mục 4.3):**

```text
tts_started
tts_audio_chunk    (binary audio stream, gửi theo chunk để phát dần)
tts_done
tts_cancelled       (mới v4.1 — khi bị Barge-in hủy giữa chừng)
tts_error           (mới v4.1 — khi synthesis thất bại)
```

Ví dụ luồng khi có Barge-in:

```text
tts_started
   ↓
tts_audio_chunk ...
   ↓
(speech mới được phát hiện qua VAD)
   ↓
tts_cancelled
```

---

## 3. Hạ tầng: Chiến lược Quản lý VRAM (6GB)

Để tránh lỗi tràn bộ nhớ (CUDA Out of Memory) khi chạy đồng thời 2 mô hình AI:

| Thành phần | Công nghệ / Mô hình | Mức cấu hình | VRAM tiêu thụ |
|---|---|---|---|
| STT Engine | faster-whisper | Model `small` (hoặc `medium` int8_float16) | ~1.5GB – 1.8GB |
| LLM Engine | llama-server (C++) | Qwen2.5-3B-Instruct-Q4_K_M.gguf | ~2.3GB – 2.5GB |
| Context Buffer | KV Cache | Giới hạn `num_ctx = 2048` | ~0.5GB |
| Hệ thống & CUDA | PyTorch Context & OS | Overhead dự phòng | ~0.8GB |
| **TTS Engine (mới, v4.0)** | Piper TTS (ONNX) hoặc tương đương | Chạy trên **CPU**, không dùng GPU | **0GB VRAM** (theo dõi CPU riêng, không tính vào ngân sách 6GB) |
| **Tổng cộng (GPU)** | | | **~5.1GB / 6.0GB (chỉ mang tính tham khảo)** |

> ⚠️ **Ghi chú review (v2):** Không nên lấy 5.1GB/6GB làm "design contract an toàn". VRAM thực tế còn phụ thuộc CUDA context, PyTorch allocator, backend Whisper, KV cache, backend llama.cpp, batch size, context size, kiến trúc GPU, phiên bản CUDA, và memory fragmentation. Khi Whisper + llama.cpp + PyTorch + CUDA + KV cache cùng tồn tại trong một process, 5.1/6GB là quá sát mép. Cần **benchmark thật** trên phần cứng mục tiêu (xem mục 9 — Benchmark Gate) và thiết kế theo ngưỡng động thay vì một con số cố định:

| Mức | VRAM |
|---|---|
| Expected (kỳ vọng thực tế) | ~4.5GB – 5.0GB |
| Hard ceiling (giới hạn cứng) | ~5.5GB |
| Vượt ceiling | Degrade chất lượng / reject phiên mới / unload model |

Ngoài ra, cho MVP nên **ưu tiên Faster-Whisper Small** (không mặc định lên Medium). Bài toán ở đây là "transcription đủ nhanh và đủ chính xác để LLM hiểu intent", không phải "transcription chính xác tuyệt đối" — đây là hai mục tiêu tối ưu khác nhau. Chỉ nâng lên Medium nếu benchmark cho thấy accuracy Small thực sự không đủ.

### 3.1. Cơ chế Thu hồi VRAM theo phiên (Session-based Lifecycle)

**Khi Client kết nối WebSocket:**
- Khởi động tiến trình con `llama-server` (`-ngl 36`).
- Nạp `WhisperModel`.

**Trong phiên hội thoại:**
- Duy trì mô hình trong VRAM để loại bỏ độ trễ khởi động lại (Cold Start).

**Khi Client ngắt kết nối (hoặc tắt app):**
1. Gửi tín hiệu `SIGTERM` tắt tiến trình `llama-server`.
2. Xóa đối tượng Whisper (`del model`).
3. Thực thi `gc.collect()` và `torch.cuda.empty_cache()` để trả VRAM về mức 0MB.

> ⚠️ **Ghi chú review (v2) — Coupling giữa Client lifecycle và Model lifecycle:** Logic hiện tại (`active_clients += 1` → start model, `active_clients == 0` → kill model) tạo ra coupling: *vòng đời client = vòng đời model*. Vấn đề nảy sinh khi Client A ngắt kết nối trong lúc Client B vẫn đang xử lý — cần đảm bảo **model lifetime > tất cả inference job đang chạy**, không chỉ dựa vào `active_clients > 0`. MVP có thể chưa cần phức tạp hóa, nhưng nên định nghĩa rõ lifecycle contract ngay từ đầu theo hướng:
>
> ```text
> Session Manager
>       │
>       ├── active_sessions
>       ├── inference_jobs
>       └── model_runtime
> ```
>
> **Lưu ý (v2.1):** Không cần implement ngay cả 3 abstraction này ở MVP. Điều quan trọng nhất là ghi rõ **contract**: *"Model runtime không được bị terminate khi vẫn còn inference job đang sử dụng model."* Implementation MVP có thể vẫn đơn giản — tránh xây hẳn một orchestration framework chỉ vì lifecycle issue này.
>
> **Ghi chú review (v4.1) — mental model đúng nên là "Model là shared infrastructure, Session là consumer":** Code hiện tại tư duy theo hướng `Session → Model Runtime` (session điều khiển vòng đời model). Mental model đúng hơn nên đảo lại:
>
> ```text
> Application
>    │
>    ├── Model Runtime
>    │      ├── Whisper
>    │      └── LLM
>    │
>    └── Sessions
>           ├── Session A
>           └── Session B
> ```
>
> MVP vẫn có thể enforce đơn giản `max_concurrent_sessions = 1` mà không cần xây orchestration phức tạp — chỉ cần đúng mental model để mở rộng sau này (nhiều session) không phải viết lại từ đầu.
>
> **Về xử lý lỗi khi acquire hardware:** trong code mẫu, `await hw_manager.acquire_hardware()` nằm **trước** khối `try`, nên nếu lỗi xảy ra giữa chừng lúc acquire, `finally: release_hardware()` sẽ không chạy → rò rỉ VRAM. Cấu trúc đúng nên là:
>
> ```python
> await websocket.accept()
> try:
>     await hw_manager.acquire_hardware()
>     # ... xử lý audio/STT/LLM
> finally:
>     await hw_manager.release_hardware()
> ```
>
> và acquire/release cần có trạng thái rõ ràng để tránh trường hợp "acquire fail → release sai state".
>
> **Về khởi động `llama-server`:** dùng `await asyncio.sleep(2)` rồi giả định server đã sẵn sàng là một assumption nguy hiểm. Nên thay bằng health-check polling thực sự:
>
> ```text
> start process
>    ↓
> poll health endpoint
>    ↓
> ready
> ```
>
> Đồng thời cần bổ sung xử lý: process thoát bất thường, timeout khi khởi động, kiểm tra port sẵn sàng, lỗi load model, lỗi cấp phát VRAM.

---

## 4. Định dạng Dữ liệu Truyền thông (WebSocket Protocol)

### 4.1. Audio Stream (Client → Server)
- **Định dạng:** Binary PCM 16-bit, Single-channel (Mono), Sample rate 16,000 Hz.
- **Kích thước gửi:** Chunk 100ms – 200ms qua kết nối binary WebSocket.

### 4.2. JSON Response (Server → Client)

**Giai đoạn 1 — Kết quả bóc băng tức thì (STT Event):**
```json
{
  "type": "stt_transcription",
  "data": {
    "language": "en",
    "text": "I think we should table this discussion for now."
  }
}
```

**Giai đoạn 2 — Phân tích & Gợi ý phản xạ (Copilot Event):**
```json
{
  "type": "copilot_analysis",
  "data": {
    "translation": "Tôi nghĩ chúng ta nên tạm gác lại cuộc thảo luận này.",
    "cultural_intent": "Đối phương muốn hoãn việc thảo luận lại, không phải muốn đưa ra bàn ngay.",
    "suggested_replies": [
      {
        "tone": "Chuyên nghiệp / Đồng thuận",
        "text": "Understood. When would be a good time to revisit this?",
        "meaning": "Đã hiểu. Khi nào thì thuận tiện để chúng ta bàn lại việc này?"
      },
      {
        "tone": "Thẳng thắn / Thăm dò",
        "text": "Is there a major blocker we need to resolve first?",
        "meaning": "Có trở ngại lớn nào chúng ta cần giải quyết trước không?"
      }
    ]
  }
}
```

### 4.3. Chuẩn hóa Event Protocol (v2, mở rộng v4.1)

Protocol ở v1 (`stt_transcription`, `copilot_analysis`) đủ cho MVP nhưng nên chuẩn hóa ngay để hỗ trợ streaming và debug dễ hơn. Bộ event types đề xuất:

```text
session_started
audio_started
stt_partial
stt_final
copilot_started
copilot_delta
copilot_done
tts_started
tts_audio_chunk
tts_done
tts_cancelled
tts_error
error
session_ended
```

Mỗi message nên có `session_id`, **`utterance_id`** và `sequence` (thứ tự message trong phiên) để client xử lý đúng thứ tự khi streaming:

```json
{
  "session_id": "sess_abc123",
  "utterance_id": "utt_007",
  "sequence": 42,
  "type": "stt_final",
  "timestamp": "2026-09-03T10:15:30Z",
  "data": {}
}
```

> ⚠️ **Ghi chú review (v4.1) — vì sao cần `utterance_id`, không chỉ `session_id` + `sequence`:** Một session có thể chứa nhiều utterance (mỗi lần người đối diện nói là một utterance riêng):
>
> ```text
> session
>  ├── utterance A
>  │    ├── stt_partial
>  │    ├── stt_final
>  │    ├── copilot
>  │    └── tts
>  │
>  ├── utterance B
>  │    ├── stt_partial
>  │    └── ...
> ```
>
> TTS càng làm `utterance_id` trở nên cần thiết: khi utterance B xuất hiện trong lúc TTS của utterance A còn đang phát, hệ thống cần biết chính xác **cancel TTS của utterance nào** (`utt_007` bị `tts_cancelled` khi `utt_008` xuất hiện). Không có `utterance_id`, việc quản lý output bất đồng bộ (đặc biệt với Barge-in ở mục 2.4.1) sẽ nhanh chóng rối loạn.

### 4.4. LLM nên streaming, không nên batch request

Cách gọi LLM ở v1 (`await http_client.post(...)` rồi chờ `res.json()`) là **batch request** — chờ LLM sinh xong toàn bộ JSON rồi mới gửi cho client, không tối ưu cho cảm nhận độ trễ (perceived latency). Nên chuyển sang streaming theo từng field:

```json
{ "type": "copilot_start" }
```
```json
{ "type": "copilot_delta", "field": "translation", "text": "Tôi nghĩ..." }
```
```json
{ "type": "copilot_delta", "field": "reply", "text": "Understood..." }
```
```json
{ "type": "copilot_done" }
```

> ⚠️ **Ghi chú review (v4.1) — không để frontend parse partial JSON của LLM trực tiếp:** LLM output (token stream) và application event (semantic event) là **hai abstraction khác nhau**. Nếu frontend nhận trực tiếp các mảnh JSON đang được LLM sinh dở (`copilot_delta` chứa fragment của JSON thô), frontend buộc phải hiểu cách Qwen đang cấu trúc JSON — rất dễ vỡ khi đổi model hoặc đổi prompt format. Kiến trúc đúng nên có một lớp parser trung gian:
>
> ```text
> llama.cpp
>    ↓
> token stream
>    ↓
> LLM output parser   (backend, không phải frontend)
>    ↓
> semantic events
>    ↓
> WebSocket
> ```
>
> Ví dụ: token stream thô từ Qwen được backend parse thành các semantic event như `translation_delta`, `intent_done`, `reply_ready` — frontend chỉ cần biết các event ngữ nghĩa này (`translation_delta`, `intent`, `reply`), không cần biết cấu trúc JSON nội bộ của LLM. Đây là thay đổi nhỏ về code nhưng quan trọng cho khả năng bảo trì lâu dài.

**Về prompt:** giữ nguyên việc gộp translation + intent + reply trong **một lần inference** (hợp lý cho MVP, không cần 3 lần gọi LLM riêng), nhưng output phải cực ngắn gọn — bỏ trường `meaning` (bản dịch cho từng reply) vì mỗi reply thêm bản dịch sẽ làm tăng token → tăng latency. Ví dụ output rút gọn:

```json
{
  "translation": "...",
  "intent": "delay discussion",
  "replies": ["...", "..."]
}
```

---

## 5. Cấu trúc Thư mục Dự án (Repository Structure)

### 5.1. Cấu trúc gốc (v1)

```text
ai-earbud-copilot/
├── docker-compose.yml
├── docs/
│   └── architecture.png
├── models/
│   ├── qwen2.5-3b-instruct-q4_k_m.gguf
│   └── llama-server
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── vram_manager.py
│   │   ├── services/
│   │   │   ├── vad_service.py
│   │   │   ├── stt_service.py
│   │   │   └── llm_service.py
│   │   ├── api/
│   │   │   └── websocket_routes.py
│   │   └── main.py
│   ├── requirements.txt
│   └── Dockerfile
└── client/
    ├── web_test/
    │   ├── index.html
    │   └── app.js
    └── mobile_app/
```

### 5.2. Cấu trúc đề xuất (v2 — theo 3 tầng Audio / Intelligence / Presentation)

`services/` (vad_service, stt_service, llm_service — gọi thẳng model từ route) được thay bằng cấu trúc tách bạch hơn, dễ thay model/transport mà không phá pipeline:

```text
backend/
└── app/
    ├── api/
    │   └── websocket.py
    │
    ├── core/
    │   ├── config.py
    │   ├── runtime.py
    │   └── vram_manager.py
    │
    ├── audio/
    │   ├── vad.py
    │   ├── chunker.py
    │   └── session.py
    │
    ├── ai/
    │   ├── stt.py
    │   ├── llm.py
    │   ├── copilot.py
    │   └── tts.py          # mới, v4.0 — chạy trên CPU
    │
    ├── protocol/
    │   ├── events.py
    │   └── schemas.py
    │
    └── main.py
```

`docker-compose.yml`, `docs/`, `models/`, và `client/` giữ nguyên như cấu trúc v1.

---

## 6. Triển khai Mã nguồn Backend Mẫu

File tham chiếu: `server_with_vram_manager.py`

> ⚠️ **Ghi chú review (v2):** Đoạn code dưới đây là **baseline v1**, minh họa ý tưởng nhưng còn 3 vấn đề cần sửa trước khi dùng làm nền triển khai thật: (1) dùng buffer cố định 1 giây thay vì VAD-driven segmentation — xem mục 2.1; (2) `acquire_hardware()` gọi trước `try` nên lỗi giữa chừng sẽ không được cleanup — xem mục 3.1; (3) LLM gọi theo kiểu batch (`await http_client.post` rồi chờ JSON đầy đủ) thay vì streaming — xem mục 4.4. Giữ code này làm tài liệu tham khảo cấu trúc tổng thể, không nên copy nguyên trạng vào production.
>
> 🔴 **Ghi chú review (v4.1) — cảnh báo mạnh hơn:** *"Code sample này chỉ mang tính tham khảo kiến trúc (architectural reference only) và không được coi là executable baseline."* Ngoài 3 vấn đề đã nêu, code mẫu còn thiếu toàn bộ các cơ chế sau — developer copy đoạn này rồi sửa dần rất dễ kế thừa sai kiến trúc:
> - Blocking Whisper inference chạy trực tiếp trong async route (không có worker/thread riêng)
> - Blocking `process.wait()` trong async context
> - Blocking model initialization (`sleep(2)` thay vì health-check)
> - Không có backpressure khi client gửi audio nhanh hơn tốc độ xử lý
> - Không có cơ chế cancellation
> - Không có utterance state machine (partial/final)
> - Không có `sequence` / `utterance_id`
> - Không có TTS, không có Barge-in, không có structured streaming parser (semantic events)
>
> Toàn bộ các phần này đã được đặc tả ở các mục tương ứng (2.1, 2.4, 2.4.1, 3.1, 4.3, 4.4) — code mẫu dưới đây **không** phản ánh các đặc tả đó, chỉ minh họa ý tưởng WebSocket + FastAPI + subprocess ở mức khung sườn.

```python
import asyncio
import os
import signal
import subprocess
import gc
import torch
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

app = FastAPI()

class LocalHardwareManager:
    def __init__(self):
        self.llama_process = None
        self.whisper_model = None
        self.active_clients = 0
        self.lock = asyncio.Lock()

    async def acquire_hardware(self):
        async with self.lock:
            self.active_clients += 1
            if self.active_clients == 1:
                cmd = [
                    "./llama-server",
                    "-m", "qwen2.5-3b-instruct-q4_k_m.gguf",
                    "-c", "2048",
                    "-ngl", "36",
                    "--port", "8080",
                    "-cb"
                ]
                self.llama_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                await asyncio.sleep(2)
                self.whisper_model = WhisperModel("small", device="cuda", compute_type="int8_float16")

    async def release_hardware(self):
        async with self.lock:
            self.active_clients = max(0, self.active_clients - 1)
            if self.active_clients == 0:
                if self.llama_process:
                    self.llama_process.send_signal(signal.SIGTERM)
                    try:
                        self.llama_process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.llama_process.kill()
                    self.llama_process = None

                if self.whisper_model:
                    del self.whisper_model
                    self.whisper_model = None

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

hw_manager = LocalHardwareManager()
LLM_ENDPOINT = "http://127.0.0.1:8080/completion"
SYSTEM_PROMPT = """You are a real-time conversational copilot.
Analyze the input speech and respond strictly in compact JSON:
{"trans": "<Vietnamese translation>", "intent": "<speaker intent in 10 words>", "replies": ["<reply 1>", "<reply 2>"]}
No markdown wrap."""

@app.websocket("/ws/copilot")
async def copilot_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await hw_manager.acquire_hardware()
    audio_buffer = bytearray()
    CHUNK_SIZE = 32000 * 2

    async with httpx.AsyncClient() as http_client:
        try:
            while True:
                data = await websocket.receive_bytes()
                audio_buffer.extend(data)

                if len(audio_buffer) >= CHUNK_SIZE:
                    pcm_data = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_buffer.clear()

                    segments, info = hw_manager.whisper_model.transcribe(
                        pcm_data,
                        beam_size=1,
                        vad_filter=True
                    )
                    text = " ".join([s.text for s in segments]).strip()

                    if not text:
                        continue

                    await websocket.send_json({
                        "type": "stt",
                        "lang": info.language,
                        "text": text
                    })

                    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nLanguage: {info.language}\nSpeech: \"{text}\"<|im_end|>\n<|im_start|>assistant\n"
                    payload = {
                        "prompt": prompt,
                        "temperature": 0.2,
                        "n_predict": 120,
                        "stop": ["<|im_end|>"]
                    }

                    res = await http_client.post(LLM_ENDPOINT, json=payload, timeout=2.0)
                    if res.status_code == 200:
                        content = res.json().get("content", "").strip()
                        await websocket.send_json({
                            "type": "copilot_suggestion",
                            "payload": content
                        })

        except WebSocketDisconnect:
            pass
        finally:
            await hw_manager.release_hardware()
```

---

## 7. Benchmark Gate (bắt buộc — chạy trước khi xây pipeline/frontend)

> ⚠️ **Ghi chú review (v2):** Không nên bắt đầu ngay bằng "Ngày 1-3: VRAM setup → Ngày 4-7: Pipeline". Ngày 1 nên có ngay một **Benchmark Gate**: đo latency + VRAM thật trên phần cứng mục tiêu. Nếu fail, dừng lại và tối ưu model — chưa xây tiếp frontend/mobile.

| Test | Target |
|---|---|
| VAD endpoint | < 150ms |
| STT 1s speech (chạy riêng lẻ) | < 400ms |
| LLM 30–50 tokens, total generation (chạy riêng lẻ) | < 500ms |
| **LLM TTFT (time-to-first-token)** | **< 200ms** |
| VRAM (khi cả 2 model cùng active) | < 5.5GB |
| **Concurrent inference E2E (mới v4.1 — thay cho "STT+LLM cộng dồn")** | **speech endpoint → first useful copilot result, xem bảng percentile bên dưới** |
| **Audio feedback loop test (P0, mới v4.1)** | **TTS playback + mic thu đồng thời — false trigger rate của Whisper** |
| **Barge-in response time (P0, mới v4.1)** | **VAD phát hiện speech mới khi TTS đang PLAYING → TTS ngừng phát: < 200ms** |

> ⚠️ **Ghi chú review (v4.1) — vì sao bỏ metric "STT + LLM cộng dồn < 1.1s":** Nếu STT và LLM chạy đồng thời và có overlap (ví dụ STT 600ms chồng lấn một phần với LLM 700ms), tổng cộng dồn `600 + 700 = 1300ms` **không phản ánh đúng E2E thực tế**, có thể chỉ khoảng 700ms nếu chúng overlap tốt. Metric đúng cần đo là **khoảng thời gian từ VAD endpoint đến khi có kết quả copilot đầu tiên hữu ích**, không phải tổng cộng dồn thời gian xử lý riêng lẻ của từng thành phần:
>
> ```text
> VAD endpoint
>      ↓
> STT final
>      ↓
> LLM first useful output
> ```
>
> #### Bảng percentile (thay cho một con số E2E duy nhất)
>
> Realtime system không nên chỉ đo một sample đơn lẻ. Cần benchmark trên **20–50 utterances** và báo cáo theo percentile:
>
> | Metric | Target |
> |---|---|
> | P50 (first useful reply) | < 1.0s |
> | P90 | < 1.3s |
> | P95 | < 1.5s |
> | Max | < 2.0s |
> | Error rate | < 2% |
>
> **P95 là con số quan trọng nhất để theo dõi**, không phải P50 — một hệ thống chạy 800ms ở phần lớn trường hợp nhưng cứ 5 câu lại có 1 câu vọt lên 2.5s vẫn không đạt chuẩn "realtime tốt".

> **Ghi chú review (v3.0) — vì sao cần Simultaneous Stress Test riêng:** Các target STT/LLM ở trên đo khi chạy **riêng lẻ**. Trên GPU 6GB (thường chỉ 16–28 SMs), khi VAD báo dứt câu và Whisper bắt đầu chạy final transcript đúng lúc llama-server đang xử lý câu trước hoặc chuẩn bị sinh token đầu — hai tiến trình độc lập sẽ **tranh chấp Streaming Multiprocessors (Compute Contention)**. Hậu quả: tốc độ inference của cả hai có thể tụt 30–50%. Benchmark Gate **bắt buộc** phải đo cả kịch bản kích hoạt đồng thời (đo bằng Concurrent inference E2E percentile ở trên), không chỉ đo đơn lẻ từng model.

> **Ghi chú review (v3.0) — VAD silence threshold cộng thẳng vào E2E:** Để Silero VAD khẳng định người nói đã dứt câu, cần quan sát một khoảng im lặng (thường cấu hình 300–500ms). Ngưỡng quá ngắn (150ms) khiến hệ thống chốt câu nhầm khi người nói chỉ ngập ngừng ("uhm", "well") — gãy ngữ cảnh. Ngưỡng an toàn hơn (~400ms) là cần thiết, nhưng **400ms này cộng thẳng vào E2E ngay từ đầu** — nghĩa là ngân sách thực tế còn lại cho STT + LLM chỉ khoảng **800ms – 900ms** để đạt tổng < 1.5s, không phải 1.5s đầy đủ cho riêng phần AI compute.

> **Ghi chú review (v2.1) — vì sao tách riêng TTFT:** Streaming LLM giảm *perceived latency* (time-to-first-useful-result), không nhất thiết giảm *compute latency* tổng. Ví dụ LLM tổng generation vẫn 450ms, nhưng nếu token đầu tiên xuất hiện ở 120ms thì người dùng đã thấy phản hồi từ rất sớm. Vì vậy "LLM total < 500ms" không nói lên được UX có nhanh hay không — cần đo riêng TTFT và first-useful-reply.

> **Ghi chú review (v2.1) — Web mic vs Bluetooth mic:** Nên thêm một test nhỏ vào Benchmark Gate ngay từ đầu (chưa cần xây mobile app): thu cùng một audio sample qua Web mic và qua Bluetooth mic, so sánh **WER**, **latency**, và **noise robustness**. Chỉ cần một prototype audio path, không cần app hoàn chỉnh — xem thêm mục 9.2.

```text
Nếu fail:
   STOP
     ↓
   Optimize model
     (chưa xây tiếp frontend/mobile)
```

### 7.1. Thứ tự thực hiện Benchmark Gate (Ngày 1)

```text
DAY 1
─────
1. NVIDIA / CUDA check
2. Whisper benchmark (riêng lẻ)
3. Qwen benchmark (riêng lẻ, đo cả TTFT)
4. TTS benchmark trên CPU (riêng lẻ)
5. Combined VRAM (Whisper + Qwen active — TTS không tính, chạy CPU)
6. Concurrent inference E2E (Whisper + llama-server chạy chồng lấn — đo Compute Contention, báo cáo P50/P90/P95/Max trên 20-50 utterances)
7. CPU stress test (STT + LLM + TTS cùng chạy — đảm bảo TTS không nghẽn event loop)
8. Audio feedback loop test (TTS playback + mic thu đồng thời — đo false trigger rate)
9. Barge-in response time (VAD phát hiện speech mới khi TTS đang phát → thời gian TTS ngừng)
          ↓
       GATE
    PASS / FAIL

PASS → Ngày 2-3: runtime manager, tiếp tục pipeline
FAIL → không code tiếp → đổi model/config → benchmark lại
```

### 7.2. MVP Assumption về hướng thu âm (bắt buộc chốt trước khi code)

> ⚠️ **Ghi chú review (v2.1) — False Activation vẫn là vấn đề lớn nhất chưa giải quyết, chỉ mới được nhận diện ở v2.** Trước khi bắt đầu code, dự án cần **chốt rõ một assumption MVP**:
>
> **"MVP giả định microphone thu chủ yếu giọng người đối diện (không phải giọng người dùng)."**
>
> Nếu assumption này đúng với setup phần cứng thực tế (vị trí micro, hướng thu, AEC), có thể triển khai tiếp theo kiến trúc hiện tại. Nếu **chưa chắc chắn** điều này đúng, thì việc phân biệt nguồn âm thanh (audio routing / mic placement — xem mục 9.1) phải được nâng lên thành **P0 validation**, tức đưa vào ngay Benchmark Gate hoặc đầu Giai đoạn 2, chứ không phải một task phụ ở cuối dự án.

---

## 8. Kế hoạch Triển khai (Milestones)

### Giai đoạn 1: Benchmark Gate & Hạ tầng Core VRAM (Ngày 1 - 3)
- [ ] **Chạy Benchmark Gate (mục 7) trước tiên** — nếu không đạt target, dừng và tối ưu model trước khi tiếp tục.
- [ ] Tải model Qwen2.5-3B-Instruct-Q4_K_M và binary llama-server.
- [ ] Kiểm thử lệnh chạy llama-server với cờ tối ưu: `-c 2048 -ngl 36`.
- [ ] Viết module `vram_manager.py` với cơ chế start/kill tiến trình, health-check polling (thay cho `sleep(2)`), và dọn dẹp CUDA cache.
- [ ] Kiểm tra dung lượng VRAM thực tế trên card 6GB thông qua `nvidia-smi` (đảm bảo ≤ 5.5GB hard ceiling khi cả Whisper và LLM cùng hoạt động — xem mục 3.1).

### Giai đoạn 2: Xây dựng Pipeline Xử lý Thời gian thực (Ngày 4 - 7)
- [ ] Xây dựng WebSocket endpoint trên FastAPI để nhận binary PCM stream.
- [ ] **Triển khai VAD-driven speech segmentation** (thay cho buffer cố định 1 giây) — xem mục 2.1.
- [ ] Tích hợp Faster-Whisper Small bóc băng theo speech segment (chế độ `beam_size=1`, streaming partial/final transcript).
- [ ] Thiết kế và khóa cố định System Prompt cho Qwen-3B để ép sinh JSON chuẩn, output rút gọn (bỏ trường `meaning` lặp lại — xem mục 4.4).
- [ ] Triển khai **LLM streaming** (`copilot_start` → `copilot_delta` → `copilot_done`) thay vì chờ response đầy đủ.
- [ ] Chuẩn hóa event protocol với `session_id` + `sequence` (xem mục 4.3).
- [ ] Đo đạc độ trễ toàn trình (Target: dứt lời → có gợi ý trong vòng 1.2 giây, dựa trên timeline streaming ở mục 2.1).
- [ ] **Tích hợp TTS Engine (CPU, mới v4.0)** — Piper TTS hoặc tương đương, xử lý song song (async), không chặn E2E của text suggestion — xem mục 2.4.
- [ ] Triển khai streaming TTS theo câu/cụm (bắt đầu đọc translation ngay khi có, không đợi intent/replies hoàn tất).
- [ ] Thêm event protocol cho audio (`tts_started`, `tts_audio_chunk`, `tts_done`).

### Giai đoạn 3: Dựng Client Thử nghiệm (Ngày 8 - 10)
- [ ] Viết giao diện Web kiểm thử (Web Audio API) thu âm trực tiếp từ microphone máy tính và gửi qua WebSocket.
- [ ] Hiển thị kết quả bóc băng, bản dịch và danh sách câu trả lời gợi ý lên màn hình (bao gồm partial transcript khi đang nói).
- [ ] **Phát thử audio TTS qua loa/tai nghe máy tính** — xác nhận translation/intent tự động đọc, quick replies chỉ đọc khi chọn (theo MVP assumption mục 2.4).
- [ ] Kiểm thử thực tế với các đoạn video hội thoại đa ngôn ngữ (tiếng Anh, tiếng Trung, tiếng Nhật).

### Giai đoạn 4: Hoàn thiện Sản phẩm Di động & Định tuyến Âm thanh (Ngày 11 - 15)
- [ ] **Test Bluetooth SCO / audio routing sớm hơn dự kiến** (không để dồn về cuối) — xem mục 9.2, vì microphone Bluetooth earbuds đi qua codec/AEC/AGC/noise suppression khác hẳn microphone web, ảnh hưởng trực tiếp đến accuracy của Whisper.
- [ ] Khởi tạo ứng dụng di động kết nối tai nghe Bluetooth (Bluetooth SCO Profile).
- [ ] Xác định rõ **cơ chế phân biệt nguồn âm thanh** (người dùng nói vs. người đối diện nói) — xem mục 9.1.
- [ ] Tích hợp tính năng hiển thị dạng thẻ phản xạ nhanh (Quick Response Cards).
- [ ] Bổ sung cơ chế ngắt nhàn rỗi (Idle Timeout): nếu người dùng không nói trong 3 phút, tự động giải phóng VRAM.

---

## 9. Vấn đề UX/Hạ tầng Âm thanh Cần Xử Lý Sớm

### 9.1. False Activation — Phân biệt nguồn âm thanh (nâng thành P0 ở v4.1)
Đây là vấn đề UX lớn với earbud assistant chưa được đề cập ở bản v1: nếu microphone là microphone của tai nghe người dùng, hệ thống cần xác định **AI đang nghe người dùng hay người đối diện**. Nếu mục tiêu là "đối phương nói tiếng Anh → AI dịch + gợi ý trả lời", thì **microphone placement / audio routing phải là một phần của kiến trúc**, không chỉ là chi tiết frontend.

> ⚠️ **Ghi chú review (v4.1) — assumption này nghiêm trọng hơn đánh giá trước đó:** MVP assumption "microphone thu chủ yếu giọng người đối diện" (mục 7.2) có rủi ro vật lý cụ thể: microphone của earbud thường nằm **rất gần miệng người dùng**, trong khi yêu cầu sản phẩm lại là nghe người đối diện — đây là một **mismatch vật lý tiềm tàng**, không chỉ là một giả định cần "lưu ý". Không cần giải quyết hoàn toàn ở architecture stage, nhưng **Speaker-source validation nên được nâng thành P0 hardware validation, ngang hàng với Bluetooth testing** (mục 9.2), đưa vào Benchmark Gate với các kịch bản:
>
> | Kịch bản test | Đo |
> |---|---|
> | User speech (chỉ người dùng nói) | Ai được detect là speaker? |
> | Other-person speech (chỉ đối phương nói) | WER, có nhận đúng là "cần dịch" không? |
> | Both speakers nói xen kẽ | False activation rate |
> | TTS playback đang phát | Miss rate / false trigger (liên quan Barge-in, mục 2.4.1) |
> | Background noise | WER dưới điều kiện nhiễu |

### 9.2. Bluetooth SCO cần benchmark sớm, không để cuối
Bản v1 để việc test Bluetooth SCO Profile ở Giai đoạn 4 (cuối cùng). Nên đưa lên sớm hơn vì:

```text
16kHz PCM
   ↓
Bluetooth codec
   ↓
OS audio routing
   ↓
AEC / AGC
   ↓
noise suppression
   ↓
microphone
   ↓
WebSocket
```

Web microphone và Bluetooth earbuds là hai môi trường rất khác nhau. Nếu chất lượng audio đầu vào kém, Whisper accuracy giảm trước khi LLM kịp làm gì — nên cần biết sớm để tránh phải thiết kế lại pipeline ở giai đoạn cuối.

> ⚠️ **Ghi chú review (v4.1) — Bluetooth không chỉ là vấn đề "audio quality", còn là vấn đề duplex behavior:** Với earbud, cần test **mic input + earbud output đồng thời** (duplex), không chỉ chất lượng thu âm đơn lẻ. Cụ thể cần kiểm tra hành vi **A2DP output vs HFP/HSP/SCO** khi microphone đang active — đây có thể là vấn đề ở tầng OS/audio profile, không phải backend, nhưng ảnh hưởng trực tiếp đến khả thi của Barge-in (mục 2.4.1). Benchmark phần cứng nên có đủ 4 kịch bản:
>
> ```text
> Web mic (baseline)
> Bluetooth mic
> Bluetooth mic + playback (không TTS)
> Bluetooth mic + TTS playback (kịch bản thực tế nhất — liên quan Audio feedback loop test ở mục 7)
> ```

---

## 10. Rủi ro & Điểm cần lưu ý (Ghi chú bổ sung)

| Rủi ro | Mô tả | Đề xuất |
|---|---|---|
| Giới hạn đa người dùng | 6GB VRAM chỉ đủ chạy tốt cho ~1 client đồng thời | Cân nhắc hàng đợi (queue) hoặc giới hạn số kết nối |
| ~~Cắt câu theo buffer cố định~~ | *(Đã xử lý ở v2 — xem mục 2.1)* Chunk cố định 1 giây (32000 samples) làm gãy câu và đẩy latency lên ~1.8s | Chuyển hẳn sang VAD-driven speech segmentation, không chỉ "cân nhắc" |
| Không có fallback khi LLM timeout | Timeout 2.0s không có cơ chế dự phòng | Thêm phản hồi mặc định hoặc rút ngắn `n_predict` khi gần timeout |
| Bảo mật WebSocket | Chưa có auth/token cho endpoint `/ws/copilot` | Bổ sung xác thực trước khi public cho mobile app |
| Context window hẹp | `num_ctx = 2048` | **Không phải blocker ở MVP** — với use case "nghe câu A → hiểu → trả lời", context nhỏ + prompt nhỏ + output nhỏ còn giúp đạt latency tốt hơn. Chỉ cần giải quyết khi mở rộng sang hội thoại nhiều lượt (rolling summary + short context) |
| Session/Model lifecycle coupling | Vòng đời client đang gắn chặt với vòng đời model | Tách Session Manager thành `active_sessions` / `inference_jobs` / `model_runtime` — xem mục 3.1 |
| `sleep(2)` giả định server sẵn sàng | Không đảm bảo llama-server đã thực sự load xong | Poll health endpoint thay vì sleep cố định — xem mục 3.1 |
| False activation (mic nghe nhầm nguồn) | Chưa phân biệt được AI đang nghe ai | Thiết kế audio routing/microphone placement như một phần kiến trúc — xem mục 9.1 |
| Bluetooth SCO chưa benchmark sớm | Audio pipeline Bluetooth khác hẳn web mic, ảnh hưởng accuracy STT | Test Bluetooth routing ngay từ Giai đoạn 2-3, không để cuối — xem mục 9.2 |
| **TTS cạnh tranh tài nguyên với Whisper/LLM (mới, v4.0)** | Nếu chạy TTS trên GPU sẽ vượt trần VRAM 6GB và tạo compute contention 3 chiều | Chạy TTS trên CPU (Piper TTS hoặc tương đương), tách khỏi critical path E2E — xem mục 2.4 |
| **Audio dồn dập nếu tự động đọc hết mọi gợi ý** | Đọc cả translation + intent + toàn bộ quick replies sẽ chồng lấp với hội thoại thật đang diễn ra | Chỉ tự động đọc translation/intent; quick replies đọc theo yêu cầu người dùng — xem MVP assumption mục 2.4 |

---

## 11. Tổng kết Review (5 vòng)

**Đánh giá tổng quan (v2.1 — review vòng 2):** Bản v2 đã xử lý hầu hết blocker kiến trúc của v1. Điểm đánh giá lại (thay cho điểm 6.5/10 ở vòng review đầu, không còn phản ánh đúng bản v2):

| Tiêu chí | Điểm |
|---|---|
| Architecture | 8/10 |
| MVP implementation readiness | 7/10 |
| Production readiness | 4/10 |

**Đánh giá tổng quan (v3.0 — review vòng 3):**

| Tiêu chí | Điểm | Nhận xét |
|---|---|---|
| Tính khả thi kỹ thuật (Feasibility) | 8.5/10 | Rất khả thi nếu tuân thủ chặt chẽ model quant 4-bit và prompt rút gọn |
| Kiến trúc & Thiết kế hệ thống | 9.0/10 | Rõ ràng, đúng chuẩn decoupled architecture, nhận diện đúng rủi ro coupling |
| Mức độ sẵn sàng triển khai (Readiness) | 9.5/10 | Đã có Benchmark Gate, tiêu chí Pass/Fail định lượng rõ ràng, sẵn sàng code |

**Đánh giá tổng quan (v4.1 — review vòng 5, sau khi thêm TTS ở v4.0):**

| Hạng mục | Điểm |
|---|---|
| Kiến trúc tổng thể | 8.5/10 |
| Tính khả thi MVP | 8/10 |
| Realtime pipeline | 7.5/10 |
| Audio architecture | 6.5/10 → cải thiện sau khi thêm Barge-in + feedback loop test |
| TTS integration | 7/10 → cải thiện sau khi thêm state machine + cancellation |
| Benchmark design | 7/10 → cải thiện sau khi đổi sang percentile-based metric |
| Implementation readiness | 7.5/10 |
| Production readiness | 4/10 |

**Kết luận review vòng 5:** Việc thêm TTS không chỉ là "thêm một module" — nó biến hệ thống từ pipeline một chiều (**Microphone → AI → UI**) thành hệ thống audio hai chiều đóng vòng (**Microphone ↔ AI ↔ Earbud**). 4 bổ sung quan trọng nhất của v4.1 để xử lý đúng bản chất này: **Barge-in, TTS cancellation, audio feedback loop, và utterance identity** (mục 2.4.1, 4.3). Sau khi sửa 8 điểm P0/P1 này, tài liệu đạt trạng thái **Conditionally Frozen** — không cần thêm vòng review kiến trúc lớn nào nữa, chỉ cần chạy Benchmark Gate thật.

**Kết luận review vòng 3 (không đổi):** Bản kế hoạch đủ điều kiện để khóa lại (freeze specification) có điều kiện và chuyển sang thực thi.

Trạng thái hiện tại: đã chuyển từ **"một architecture nghe có vẻ hợp lý"** thành **"một architecture có thể đem đi benchmark thực tế"** (Benchmark-ready). Đây không cần một hệ thống phức tạp kiểu compiler/agent/RAG — chỉ cần một **realtime stateful pipeline nhỏ gọn**:

```text
          ┌───────────────┐
Audio ───►│ VAD + Segment │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │     STT       │
          └───────┬───────┘
                  │
             final text
                  │
                  ▼
          ┌───────────────┐
          │     LLM       │
          │ Translation   │
          │ Intent        │
          │ Reply         │
          └───────┬───────┘
                  │
                  ▼
              WebSocket
                  │
                  ▼
                UI
```

### Đã giải quyết (từ v1 → v2)
✅ Fixed 1s buffer → VAD-driven · ✅ Streaming/incremental architecture · ✅ Benchmark Gate · ✅ VRAM không còn coi 5.1GB là safe · ✅ Health check · ✅ Lifecycle failure handling · ✅ Session/model coupling được nhận diện · ✅ Event protocol chuẩn hóa (`session_id`, `sequence`) · ✅ LLM streaming · ✅ Prompt output nhỏ (bỏ `meaning`) · ✅ Repository separation (3 tầng) · ✅ Bluetooth được kéo lên sớm · ✅ False activation được nhận diện · ✅ Context 2048 hạ từ blocker → non-blocker

### Đã giải quyết thêm ở v2.1 (4 điều chỉnh nhỏ theo review vòng 2)
✅ "STT incremental" → "Pseudo-streaming / Sliding-window incremental STT" (đúng bản chất Faster-Whisper) · ✅ "Latency Budget" → "Target Component Latency" (E2E < 1.5s là acceptance criterion thật) · ✅ Thêm `LLM TTFT < 200ms` và `First useful reply < 1.2s` vào Benchmark Gate · ✅ Chốt MVP assumption về hướng thu âm (mục 7.2)

### Đã giải quyết thêm ở v3.0 (3 rủi ro vật lý cấp phần cứng theo review vòng 3)
✅ **Compute Contention** — thêm Simultaneous Stress Test vào Benchmark Gate để đo tranh chấp SM khi Whisper + llama-server chạy chồng lấn (mục 7) · ✅ **Sliding-Window overhead** — giới hạn tần suất transcribe lại, chỉ kích hoạt Whisper khi speech đủ dài hoặc VAD phát hiện speech probability suy giảm (mục 2.2) · ✅ **VAD silence threshold** — làm rõ 300–500ms cộng thẳng vào E2E, ngân sách AI compute thực tế chỉ còn ~800–900ms (mục 7)

### Đã giải quyết thêm ở v4.1 (8 điểm P0/P1 theo review vòng 5, sau khi thêm TTS)
✅ **Benchmark metric đúng** — đổi "STT+LLM cộng dồn" thành Concurrent inference E2E theo percentile P50/P90/P95/Max + error rate (mục 7) · ✅ **`utterance_id`** thêm vào event protocol (mục 4.3) · ✅ **TTS cancellation** — thêm `tts_cancelled`/`tts_error` (mục 2.4.1, 4.3) · ✅ **Barge-in state machine** — TTS PLAYING → speech mới → cancel → resume listening (mục 2.4.1) · ✅ **Audio feedback loop benchmark** — TTS playback + mic thu đồng thời (mục 7) · ✅ **LLM streaming → semantic events** — backend parse token stream thành event ngữ nghĩa, frontend không đọc JSON thô (mục 4.4) · ✅ **Benchmark percentile** thay cho một con số E2E duy nhất (mục 7) · ✅ **Sửa mâu thuẫn trạng thái tài liệu** — "Conditionally Frozen — Awaiting Benchmark Gate" thay vì tuyên bố FROZEN tuyệt đối

### Giữ nguyên từ v1
✅ Local/offline architecture · ✅ WebSocket full-duplex · ✅ FastAPI · ✅ Faster-Whisper · ✅ llama.cpp · ✅ Qwen 3B · ✅ 6GB VRAM target · ✅ Web client trước mobile · ✅ 1 client trước · ✅ JSON event protocol · ✅ VRAM lifecycle manager · ✅ Thứ tự milestone tổng thể (infra → pipeline → web test → mobile)

### Vẫn cần theo dõi (không phải blocker, nhưng chưa "xong")
🟡 **False activation** — nay đã nâng thành P0 hardware validation (mục 9.1), nhưng vẫn chưa có giải pháp kỹ thuật thật (chỉ có kế hoạch benchmark). Kết quả benchmark sẽ quyết định có cần thiết kế lại audio routing hay không.
🟡 **Production readiness (4/10)** — auth WebSocket, đa client, error fallback khi LLM timeout vẫn cần thiết kế thêm trước khi lên production (không cần giải quyết ngay ở giai đoạn MVP/benchmark).
🟡 **Bluetooth duplex behavior** — A2DP vs HFP/HSP/SCO khi mic active là vấn đề tầng OS, cần benchmark thật trước khi khẳng định khả thi (mục 9.2).

### Bắt buộc sửa (đã đưa vào tài liệu — baseline v1 không nên copy nguyên trạng)
🔴 Không dùng fixed 1-second buffer làm speech boundary — chuyển sang VAD-driven speech segmentation
🔴 Benchmark VRAM thật thay vì coi 5.1GB là "an toàn"
🔴 LLM nên streaming, không batch request
🔴 Có partial STT / final STT (không đợi đủ 1 giây)
🔴 Không `sleep(2)` để giả định llama-server ready — dùng health-check polling
🔴 Thêm xử lý lỗi khi process khởi động thất bại / VRAM allocation fail
🔴 Chuẩn hóa event protocol kèm `session_id` + `utterance_id` + `sequence`
🔴 Test Bluetooth audio routing sớm hơn (không để cuối)
🔴 **(mới v4.1)** Barge-in + TTS cancellation là contract bắt buộc, không phải tùy chọn

---

## 12. Thứ tự thực hiện tiếp theo (Verdict cuối — review vòng 5, Conditionally Frozen)

> 🟡 Bản v4.1 hiện tại đạt mức **"có thể triển khai"**, khóa lại có điều kiện (**Conditionally Frozen**) — implementation bị BLOCKED cho đến khi Benchmark Gate PASS. Không cần thêm vòng review kiến trúc lớn nào nữa. Đặc biệt: **không code mobile trước**.

### 12.1. Kiến trúc pipeline cuối cùng (sau khi tích hợp TTS + Barge-in)

```text
Audio
  ↓
VAD
  ↓
Utterance State
  ↓
STT (partial / final)
  ↓
LLM (streaming)
  ↓
Semantic Events
  ├──────────────► UI
  │
  └──────────────► TTS
                         ↓
                      Playback
                         ↓
                    Barge-in ──► quay lại Audio State (đóng vòng lặp)
```

> Điểm khác biệt quan trọng nhất so với v3.0: **Audio Output (TTS) quay ngược lại ảnh hưởng Audio State (Barge-in)** — đây là bản chất "hai chiều" mà v4 buộc phải xử lý, thay vì tư duy một chiều Microphone → AI → UI.

### 12.2. Quy trình Benchmark Gate → Freeze → Thực thi

```text
                    ┌──────────────┐
                    │  6GB GPU     │
                    └──────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Benchmark Gate  │
                  │                 │
                  │ VRAM            │
                  │ Whisper         │
                  │ Qwen            │
                  │ TTFT            │
                  │ Concurrent E2E  │
                  │ (P50/P90/P95)   │
                  │ Feedback loop   │
                  │ Barge-in        │
                  └────────┬────────┘
                           │
                     PASS / FAIL
                      /         \
                   FAIL         PASS
                    ↓             ↓
                 Optimize       Freeze
                                  ↓
                               Backend
                                  ↓
                             Web Test
                                  ↓
                        Bluetooth (duplex + feedback)
                                  ↓
                               Mobile
```

Nếu Benchmark Gate (mục 7) **PASS** trên toàn bộ chỉ số (bao gồm Concurrent inference percentile, Audio feedback loop, và Barge-in response time), kiến trúc v4.1 được đánh giá đủ tốt để chuyển từ "Conditionally Frozen" sang thực thi MVP đầy đủ.

### 12.3. Hành động cụ thể cho Ngày 1 (Benchmark Gate)

1. **Đo STT riêng lẻ:** Viết script Python đơn giản đo thời gian thực thi của `faster-whisper small` với file mẫu 2 giây.
2. **Đo LLM riêng lẻ:** Khởi chạy `llama-server` với model Qwen2.5-3B-Q4_K_M, gửi request qua `curl`/`httpx` để đo **TTFT** và tổng thời gian sinh 40 token.
3. **Đo TTS riêng lẻ:** Benchmark Piper TTS (hoặc engine tương đương) trên CPU với câu ngắn <10 từ.
4. **Đo Concurrent Inference E2E (bắt buộc):** Kích hoạt đồng thời Whisper + llama-server (mô phỏng đúng kịch bản thực tế), chạy 20–50 utterances, báo cáo **P50/P90/P95/Max** thay vì một con số duy nhất — so sánh với kết quả đo riêng lẻ ở bước 1–2 để xác định mức độ suy giảm do Compute Contention.
5. **Đo Audio Feedback Loop (P0, mới v4.1):** Phát TTS qua loa/tai nghe trong lúc mic đang thu, đo tỷ lệ Whisper nhận nhầm audio TTS thành speech input mới.
6. **Đo Barge-in Response Time (P0, mới v4.1):** Từ lúc VAD phát hiện speech mới khi TTS đang PLAYING đến khi TTS thực sự ngừng phát — target < 200ms.
7. **Đo Web mic vs Bluetooth mic (mục 9.2):** So sánh WER, latency, noise robustness trên cùng một audio sample qua 4 kịch bản (Web mic / Bluetooth mic / Bluetooth mic + playback / Bluetooth mic + TTS playback).
8. Nếu tất cả các chỉ số ở mục 7 đều PASS → chuyển từ "Conditionally Frozen" sang thực thi, bắt đầu Ngày 2–3 (runtime manager, tiếp tục pipeline theo Giai đoạn 2). Nếu FAIL → không code tiếp, quay lại tối ưu model/config, benchmark lại.
