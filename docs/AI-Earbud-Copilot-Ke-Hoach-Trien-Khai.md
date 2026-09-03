# AI Conversational Copilot — Kế hoạch Triển khai (Module & Task)

> **Phiên bản:** 1.0.0
> **Tài liệu tham chiếu:** `AI-Earbud-Copilot-Tong-Hop.md` v4.1.0 (Conditionally Frozen — Awaiting Benchmark Gate)
> **Mục đích:** Chuyển đặc tả kiến trúc thành danh sách module cụ thể và task backlog có thể giao việc, theo dõi tiến độ, và tick "Definition of Done" — không lặp lại phần phân tích/lý do kiến trúc đã có ở tài liệu tham chiếu.
> **Nguyên tắc bắt buộc:** Không code Giai đoạn N+1 khi Giai đoạn N chưa đạt Definition of Done. Benchmark Gate (Task nhóm B) là điều kiện tiên quyết cho toàn bộ Pipeline (nhóm C trở đi).

---

## 0. Sơ đồ Module & Phụ thuộc (Dependency Graph)

```text
core/config.py ──────────────┐
                              │
core/runtime.py ◄─────────── core/vram_manager.py
   │  (Model Runtime, shared infra — không coupling theo session)
   │
   ├──► ai/stt.py ───────────┐
   ├──► ai/llm.py ───────────┤
   └──► ai/tts.py (CPU) ─────┤
                              ▼
                        ai/copilot.py (orchestrator: STT → LLM → semantic events → TTS)
                              │
audio/vad.py ──► audio/chunker.py ──► audio/session.py (Utterance State Machine)
                              │
                              ▼
                  protocol/events.py + protocol/schemas.py
                              │
                              ▼
                        api/websocket.py
                              │
                              ▼
                     client/web_test/ (Giai đoạn 3)
                              │
                              ▼
                     client/mobile_app/ (Giai đoạn 4, sau Bluetooth validation)
```

**Thứ tự triển khai bắt buộc:** `core` → `ai` (benchmark riêng lẻ trước) → `audio` → `protocol` → `api` → `client web` → `Bluetooth validation` → `client mobile`.

---

## 1. Danh sách Module

### 1.1. `core/config.py`
| | |
|---|---|
| **Trách nhiệm** | Load cấu hình tập trung: đường dẫn model, ngưỡng VRAM, tham số VAD, benchmark targets |
| **Input** | File `.env` / `config.yaml` |
| **Output** | Config object dùng chung cho toàn bộ module khác |
| **Phụ thuộc** | Không |
| **Tham chiếu spec** | Mục 3 (VRAM), mục 7 (Benchmark targets) |

### 1.2. `core/runtime.py`
| | |
|---|---|
| **Trách nhiệm** | **Model Runtime** — quản lý vòng đời Whisper/LLM/TTS như hạ tầng dùng chung (shared infrastructure), tách biệt khỏi Session |
| **Input** | Config từ `core/config.py` |
| **Output** | Instance model sẵn sàng phục vụ nhiều session (MVP: `max_concurrent_sessions = 1`) |
| **Contract bắt buộc** | *"Model runtime không được terminate khi vẫn còn inference job đang dùng model."* |
| **Phụ thuộc** | `core/vram_manager.py` |
| **Tham chiếu spec** | Mục 3.1 — mental model `Application → Model Runtime + Sessions` |

### 1.3. `core/vram_manager.py`
| | |
|---|---|
| **Trách nhiệm** | Cấp phát/giải phóng VRAM theo lifecycle; health-check polling khi khởi động `llama-server` (không dùng `sleep()`); xử lý lỗi khởi động/OOM |
| **Input** | Tín hiệu start/stop từ `core/runtime.py` |
| **Output** | Trạng thái VRAM hiện tại, sự kiện lỗi (process crash, OOM, startup timeout) |
| **Phụ thuộc** | `core/config.py` |
| **Tham chiếu spec** | Mục 3.1 |

### 1.4. `audio/vad.py`
| | |
|---|---|
| **Trách nhiệm** | Silero VAD — phát hiện speech start/end, tính speech probability để hỗ trợ sliding-window trigger và Barge-in |
| **Input** | PCM stream 16kHz mono |
| **Output** | Sự kiện `speech_started`, `speech_probability`, `speech_ended` (VAD endpoint) |
| **Phụ thuộc** | Không |
| **Tham chiếu spec** | Mục 2.2 (timeline), mục 7 (VAD silence threshold), mục 2.4.1 (Barge-in trigger) |

### 1.5. `audio/chunker.py`
| | |
|---|---|
| **Trách nhiệm** | Tích lũy audio theo VAD (không dùng buffer cố định); quyết định khi nào đạt "minimum inference window" để trigger partial STT |
| **Input** | PCM stream + sự kiện từ `audio/vad.py` |
| **Output** | Audio segment sẵn sàng cho STT (partial hoặc final) |
| **Contract bắt buộc** | 1.5s chỉ là ngưỡng tối thiểu cho partial inference, **không phải** speech boundary — VAD endpoint mới là điều kiện bắt buộc cho final STT |
| **Phụ thuộc** | `audio/vad.py` |
| **Tham chiếu spec** | Mục 2.2 (Sliding-Window contract, review v4.1) |

### 1.6. `audio/session.py`
| | |
|---|---|
| **Trách nhiệm** | **Utterance State Machine** — quản lý trạng thái từng utterance (partial → final → copilot → tts → done/cancelled), sinh `utterance_id` |
| **Input** | Sự kiện từ `audio/chunker.py`, `ai/copilot.py`, `ai/tts.py` |
| **Output** | `utterance_id`, trạng thái hiện tại của utterance |
| **Phụ thuộc** | `audio/chunker.py` |
| **Tham chiếu spec** | Mục 4.3 (vì sao cần `utterance_id`) |

### 1.7. `ai/stt.py`
| | |
|---|---|
| **Trách nhiệm** | Wrapper Faster-Whisper — chạy trong worker/thread riêng (không block event loop), hỗ trợ cả partial và final transcription |
| **Input** | Audio segment từ `audio/chunker.py` |
| **Output** | Text + ngôn ngữ (LID), gắn `partial`/`final` flag |
| **Phụ thuộc** | `core/runtime.py` |
| **Tham chiếu spec** | Mục 2.2 (Pseudo-streaming / Sliding-window incremental STT — tên gọi chính xác, không phải streaming ASR thật) |

### 1.8. `ai/llm.py`
| | |
|---|---|
| **Trách nhiệm** | Wrapper `llama-server` — gọi streaming completion, đo TTFT |
| **Input** | Final transcript + system prompt |
| **Output** | Token stream thô từ Qwen2.5-3B |
| **Phụ thuộc** | `core/runtime.py` |
| **Tham chiếu spec** | Mục 4.4 |

### 1.9. `ai/copilot.py` (Orchestrator)
| | |
|---|---|
| **Trách nhiệm** | **LLM Output Parser** — nhận token stream thô từ `ai/llm.py`, parse thành **semantic events** (`translation_delta`, `intent_done`, `reply_ready`); điều phối gọi `ai/tts.py` cho phần translation |
| **Input** | Token stream từ `ai/llm.py` |
| **Output** | Semantic events gửi tới `protocol/events.py` |
| **Contract bắt buộc** | Frontend không bao giờ nhận JSON thô đang được LLM sinh dở — chỉ nhận semantic event đã parse |
| **Phụ thuộc** | `ai/llm.py`, `ai/tts.py` |
| **Tham chiếu spec** | Mục 4.4 (review v4.1 — semantic events) |

### 1.10. `ai/tts.py`
| | |
|---|---|
| **Trách nhiệm** | Wrapper Piper TTS (CPU, chạy trong worker/process riêng để tránh event-loop contention); state machine `IDLE → SYNTHESIZING → PLAYING → INTERRUPTED/DONE`; xử lý Barge-in cancellation |
| **Input** | Text (translation, hoặc intent/reply nếu được yêu cầu) từ `ai/copilot.py`; tín hiệu `speech_started` từ `audio/vad.py` (để Barge-in) |
| **Output** | Audio stream (chunk), sự kiện `tts_started`/`tts_audio_chunk`/`tts_done`/`tts_cancelled`/`tts_error` |
| **Contract bắt buộc** | Khi `audio/vad.py` phát hiện speech mới trong lúc state = `PLAYING` → chuyển `INTERRUPTED`, dừng phát trong < 200ms |
| **Phụ thuộc** | `core/runtime.py` (không dùng GPU), `audio/vad.py` (Barge-in signal) |
| **Tham chiếu spec** | Mục 2.4, 2.4.1 |

### 1.11. `protocol/events.py`
| | |
|---|---|
| **Trách nhiệm** | Định nghĩa toàn bộ event type enum (`session_started`...`tts_cancelled`...`error`); Event Bus nội bộ (chỉ cần `asyncio.Queue`, không cần message broker thật) |
| **Input/Output** | Event object có `session_id`, `utterance_id`, `sequence`, `timestamp`, `type`, `data` |
| **Phụ thuộc** | Không |
| **Tham chiếu spec** | Mục 4.3 |

### 1.12. `protocol/schemas.py`
| | |
|---|---|
| **Trách nhiệm** | Pydantic schema validate cấu trúc JSON cho từng event type, đảm bảo output LLM rút gọn (không có trường `meaning` dư thừa) |
| **Phụ thuộc** | `protocol/events.py` |
| **Tham chiếu spec** | Mục 4.4 |

### 1.13. `api/websocket.py`
| | |
|---|---|
| **Trách nhiệm** | Endpoint `/ws/copilot` — nhận PCM binary, điều phối `audio/session.py` → `ai/copilot.py` → `protocol/events.py`, gửi event ra client; xử lý `acquire_hardware()`/`release_hardware()` đúng try/finally |
| **Phụ thuộc** | Toàn bộ module trên |
| **Tham chiếu spec** | Mục 3.1, mục 6 (code baseline — không copy nguyên trạng) |

---

## 2. Task Backlog

> Mỗi task có: **ID**, **Module**, **Mô tả**, **Ưu tiên (P0/P1/P2)**, **Acceptance Criteria**, **Phụ thuộc**.
> P0 = chặn Benchmark Gate hoặc chặn tiến độ Phase. P1 = cần cho MVP nhưng không chặn benchmark. P2 = có thể lùi sau MVP.

### Nhóm A — Setup hạ tầng (Ngày 1, trước Benchmark Gate)

| ID | Module | Mô tả | Ưu tiên | Acceptance Criteria | Phụ thuộc |
|---|---|---|---|---|---|
| A1 | — | Kiểm tra NVIDIA/CUDA, driver version, `nvidia-smi` hoạt động | P0 | GPU nhận diện đúng, CUDA version tương thích llama.cpp + faster-whisper | — |
| A2 | — | Tải model: Qwen2.5-3B-Instruct-Q4_K_M.gguf, llama-server binary, Faster-Whisper Small, Piper TTS voice (VI + EN) | P0 | Tất cả file model tồn tại đúng path trong `models/` | A1 |
| A3 | `core/config.py` | Viết config loader (đường dẫn model, ngưỡng VRAM, benchmark targets) | P0 | Load thành công từ `.env`/`config.yaml` | A2 |

### Nhóm B — Benchmark Gate (Ngày 1, BẮT BUỘC trước khi code Pipeline)

| ID | Module | Mô tả | Ưu tiên | Acceptance Criteria (target) | Phụ thuộc |
|---|---|---|---|---|---|
| B1 | — | Script đo STT riêng lẻ (Faster-Whisper Small, audio mẫu 1-2s) | P0 | < 400ms | A3 |
| B2 | — | Script đo LLM riêng lẻ: TTFT + tổng thời gian 30-50 token | P0 | TTFT < 200ms, total < 500ms | A3 |
| B3 | — | Script đo TTS riêng lẻ (Piper, câu <10 từ, CPU) | P0 | < 300–400ms | A3 |
| B4 | — | Đo VRAM khi Whisper + Qwen cùng active (`nvidia-smi`) | P0 | < 5.5GB hard ceiling | B1, B2 |
| B5 | — | **Concurrent Inference E2E**: chạy 20-50 utterance với Whisper + llama-server chồng lấn, đo speech-endpoint → first-useful-result | P0 | P50 < 1.0s, P90 < 1.3s, P95 < 1.5s, Max < 2.0s, error rate < 2% | B1, B2, B4 |
| B6 | — | **CPU stress test**: STT + LLM + TTS cùng chạy, đo CPU saturation, event-loop lag, WebSocket jitter | P0 | Không nghẽn event loop; TTS latency không tăng đáng kể dưới tải | B1, B2, B3 |
| B7 | — | **Audio feedback loop test**: phát TTS qua loa/tai nghe trong lúc mic đang thu, đo false-trigger rate của Whisper | P0 | Ghi nhận baseline false-trigger rate (chưa cần target cứng ở lần đo đầu — dùng làm cơ sở thiết kế Barge-in) | B3 |
| B8 | — | **Barge-in response time**: mô phỏng VAD phát hiện speech mới khi TTS đang phát, đo thời gian dừng phát | P0 | < 200ms | B3 |
| B9 | — | **Speaker-source validation**: test 5 kịch bản (user only / other only / both / TTS playback / background noise) | P0 | Ghi nhận WER + false activation rate cho từng kịch bản — làm input quyết định thiết kế audio routing | B7 |
| B10 | — | **Web mic vs Bluetooth mic**: so sánh WER, latency, noise robustness trên 4 kịch bản (Web / BT / BT+playback / BT+TTS) | P0 | Ghi nhận số liệu — không cần đạt target cứng, dùng để quyết định có cần audio routing riêng cho Bluetooth | B3, B7 |
| **GATE** | — | **Quyết định PASS/FAIL dựa trên B1–B10** | — | Tất cả P0 đạt target → PASS, chuyển Nhóm C. Bất kỳ mục nào FAIL → dừng, tối ưu model/config, đo lại | B1–B10 |

### Nhóm C — Core Runtime & VRAM (chỉ bắt đầu sau khi Benchmark Gate PASS)

| ID | Module | Mô tả | Ưu tiên | Acceptance Criteria | Phụ thuộc |
|---|---|---|---|---|---|
| C1 | `core/vram_manager.py` | Health-check polling khi start `llama-server` (thay `sleep(2)`) | P0 | Server chỉ được coi "ready" sau khi poll endpoint health trả 200 | GATE |
| C2 | `core/vram_manager.py` | Xử lý lỗi: process crash, startup timeout, port not ready, OOM | P0 | Mỗi lỗi có log rõ ràng + cleanup VRAM đúng | C1 |
| C3 | `core/runtime.py` | Model Runtime tách khỏi Session lifecycle (`Application → Model Runtime + Sessions`) | P0 | Model không bị kill khi 1 session disconnect nhưng vẫn còn inference job từ session khác (kể cả `max_concurrent_sessions=1` vẫn phải theo đúng contract) | C1, C2 |
| C4 | `api/websocket.py` | Sửa `acquire_hardware()` vào trong `try` (không phải trước `try`) | P0 | Lỗi giữa chừng khi acquire vẫn trigger `finally: release_hardware()` | C3 |

### Nhóm D — Audio Pipeline (VAD-driven, không buffer cố định)

| ID | Module | Mô tả | Ưu tiên | Acceptance Criteria | Phụ thuộc |
|---|---|---|---|---|---|
| D1 | `audio/vad.py` | Tích hợp Silero VAD, expose `speech_started`/`speech_probability`/`speech_ended` | P0 | VAD endpoint latency < 150ms (khớp B-benchmark) | GATE |
| D2 | `audio/chunker.py` | Audio accumulation theo VAD; trigger partial STT ở minimum window (~1.5s), mandatory final STT ở VAD endpoint | P0 | Utterance ngắn (<1.5s, ví dụ "Yes.") vẫn có final STT đúng qua VAD endpoint, không phụ thuộc ngưỡng 1.5s | D1 |
| D3 | `audio/session.py` | Utterance State Machine + sinh `utterance_id` | P0 | Mỗi utterance có `utterance_id` duy nhất, trạng thái transition đúng thứ tự | D2 |

### Nhóm E — AI Engines (đã qua Benchmark riêng lẻ ở Nhóm B)

| ID | Module | Mô tả | Ưu tiên | Acceptance Criteria | Phụ thuộc |
|---|---|---|---|---|---|
| E1 | `ai/stt.py` | Wrapper Faster-Whisper chạy trong thread/worker riêng, hỗ trợ partial + final | P0 | Không block event loop (đo bằng event-loop lag test) | GATE, D2 |
| E2 | `ai/llm.py` | Wrapper `llama-server` streaming completion, đo TTFT runtime | P0 | Token stream trả về đúng thứ tự, không mất token | GATE |
| E3 | `ai/copilot.py` | Parser: token stream → semantic events (`translation_delta`, `intent_done`, `reply_ready`) | P0 | Frontend chỉ nhận semantic event, không có JSON thô của LLM lọt ra ngoài | E2 |
| E4 | `ai/llm.py` + prompt | System prompt rút gọn: bỏ trường `meaning`, output tối giản | P1 | Token count output giảm rõ rệt so với baseline v1 (đo trước/sau) | E2 |
| E5 | `ai/tts.py` | Wrapper Piper TTS chạy CPU, trong worker/process riêng | P0 | Không block event loop (đo bằng CPU stress test B6) | GATE |
| E6 | `ai/tts.py` | Streaming TTS theo câu/cụm (bắt đầu đọc translation ngay khi có, không đợi cả JSON) | P1 | Time-to-first-audio giảm so với đợi full JSON | E5, E3 |
| E7 | `ai/tts.py` | State machine `IDLE → SYNTHESIZING → PLAYING → INTERRUPTED/DONE` | P0 | Transition đúng khi có Barge-in signal | E5, D1 |
| E8 | `ai/tts.py` | Barge-in: nhận `speech_started` từ VAD khi state=`PLAYING` → cancel, phát `tts_cancelled` | P0 | Response time < 200ms (khớp B8) | E7, D1 |
| E9 | `ai/copilot.py` | MVP scope đọc tự động: Translation → AUTO, Intent → optional (UI only mặc định), Reply → manual | P0 | Test thực tế: hội thoại liên tục nhiều câu không gây audio overload (đánh giá định tính + feedback) | E3, E7 |

### Nhóm F — Protocol & WebSocket

| ID | Module | Mô tả | Ưu tiên | Acceptance Criteria | Phụ thuộc |
|---|---|---|---|---|---|
| F1 | `protocol/events.py` | Định nghĩa đầy đủ event type enum (bao gồm `tts_cancelled`, `tts_error`) | P0 | Danh sách khớp mục 4.3 của spec | — |
| F2 | `protocol/events.py` | Event Bus nội bộ bằng `asyncio.Queue` (không dùng broker ngoài) | P0 | Message pass đúng thứ tự trong 1 session | F1 |
| F3 | `protocol/schemas.py` | Schema validate `session_id` + `utterance_id` + `sequence` + `timestamp` bắt buộc trên mọi event | P0 | Event thiếu field bắt buộc bị reject ở tầng validate | F1 |
| F4 | `api/websocket.py` | Endpoint `/ws/copilot` nhận PCM, điều phối toàn bộ pipeline, emit event | P0 | E2E hoạt động end-to-end với audio thật (không phải benchmark script) | C4, D3, E3, E8, F2, F3 |
| F5 | `api/websocket.py` | Backpressure: xử lý khi client gửi audio nhanh hơn tốc độ xử lý | P1 | Không tràn buffer/memory khi test với audio liên tục dài | F4 |

### Nhóm G — Web Client Thử nghiệm (Giai đoạn 3)

| ID | Module | Mô tả | Ưu tiên | Acceptance Criteria | Phụ thuộc |
|---|---|---|---|---|---|
| G1 | `client/web_test/` | Web Audio API thu mic, gửi PCM qua WebSocket | P0 | Audio gửi đúng format (16kHz mono PCM16) | F4 |
| G2 | `client/web_test/` | Hiển thị partial/final transcript, translation, intent, quick replies | P0 | UI cập nhật realtime theo semantic event | F4 |
| G3 | `client/web_test/` | Phát audio TTS qua loa máy tính, xử lý `tts_cancelled` (dừng phát ngay khi nhận event) | P0 | Nghe được audio; khi Barge-in xảy ra, audio dừng ngay | F4, E8 |
| G4 | `client/web_test/` | Test đa ngôn ngữ (Anh, Trung, Nhật) qua video mẫu | P1 | Ghi nhận WER định tính cho từng ngôn ngữ | G1, G2 |

### Nhóm H — Bluetooth & Speaker Validation (đưa sớm, không để cuối)

| ID | Module | Mô tả | Ưu tiên | Acceptance Criteria | Phụ thuộc |
|---|---|---|---|---|---|
| H1 | — | Prototype audio path Bluetooth SCO (chưa cần mobile app hoàn chỉnh) | P0 | Thu được PCM từ Bluetooth mic gửi qua cùng WebSocket endpoint | F4 |
| H2 | — | Test duplex: A2DP output vs HFP/HSP/SCO khi mic đang active | P0 | Ghi nhận hành vi OS/audio profile — xác định có cần audio routing riêng | H1 |
| H3 | — | Test Speaker-source validation trên phần cứng Bluetooth thật (không chỉ giả lập ở B9) | P0 | So sánh false activation rate giữa mic web (B9) và mic Bluetooth thật | H1, H2 |

### Nhóm I — Mobile App (chỉ sau khi Nhóm H hoàn tất)

| ID | Module | Mô tả | Ưu tiên | Acceptance Criteria | Phụ thuộc |
|---|---|---|---|---|---|
| I1 | `client/mobile_app/` | Khởi tạo app kết nối Bluetooth earbuds | P1 | Kết nối ổn định, audio path khớp kết quả Nhóm H | H1, H2, H3 |
| I2 | `client/mobile_app/` | Quick Response Cards UI | P1 | UI hiển thị đúng semantic event | I1, G2 |
| I3 | `client/mobile_app/` | Idle Timeout: giải phóng VRAM sau 3 phút không nói | P2 | VRAM về 0 sau timeout, khôi phục đúng khi có audio mới | I1, C3 |

---

## 3. Definition of Done theo từng Phase

| Phase | Điều kiện hoàn thành |
|---|---|
| **Benchmark Gate (Nhóm A, B)** | Toàn bộ B1–B10 đạt target hoặc có quyết định tối ưu rõ ràng nếu FAIL. Không chuyển Phase tiếp nếu GATE = FAIL. |
| **Core + Audio + AI (Nhóm C, D, E)** | Pipeline chạy được end-to-end trên audio giả lập (không cần UI), Barge-in hoạt động đúng contract, event-loop không bị nghẽn dưới CPU stress test. |
| **Protocol + WebSocket (Nhóm F)** | `/ws/copilot` hoạt động với client thật, mọi event có đủ `session_id`/`utterance_id`/`sequence`. |
| **Web Client (Nhóm G)** | Demo được đầy đủ: nói → thấy transcript/translation → nghe TTS → Barge-in hoạt động khi ngắt lời. |
| **Bluetooth Validation (Nhóm H)** | Có số liệu WER/false-activation trên phần cứng Bluetooth thật, quyết định rõ có cần thiết kế lại audio routing hay không trước khi vào Mobile. |
| **Mobile (Nhóm I)** | App kết nối ổn định qua Bluetooth, tái sử dụng toàn bộ backend không đổi kiến trúc. |

---

## 4. Bảng theo dõi tiến độ tổng hợp (Master Checklist)

- [ ] Nhóm A — Setup hạ tầng (A1–A3)
- [ ] Nhóm B — Benchmark Gate (B1–B10) → **GATE: PASS / FAIL**
- [ ] Nhóm C — Core Runtime & VRAM (C1–C4)
- [ ] Nhóm D — Audio Pipeline (D1–D3)
- [ ] Nhóm E — AI Engines & TTS/Barge-in (E1–E9)
- [ ] Nhóm F — Protocol & WebSocket (F1–F5)
- [ ] Nhóm G — Web Client (G1–G4)
- [ ] Nhóm H — Bluetooth & Speaker Validation (H1–H3)
- [ ] Nhóm I — Mobile App (I1–I3)

> Thứ tự thực hiện tuân thủ nghiêm ngặt dependency graph ở mục 0 — không bắt đầu nhóm sau khi nhóm trước chưa đạt Definition of Done tương ứng (mục 3).
