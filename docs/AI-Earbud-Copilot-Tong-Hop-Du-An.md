# AI Conversational Copilot (Real-time Earbud Assistant)
## Tài liệu Tổng hợp Dự án

> **Phiên bản:** Final 1.0 — hợp nhất từ Đặc tả kiến trúc v4.1.0 (5 vòng review) + Kế hoạch triển khai v1.0
> **Trạng thái dự án:** 🟡 **Conditionally Frozen — Awaiting Benchmark Gate.** Kiến trúc đã khóa lại có điều kiện; implementation bị chặn cho đến khi Benchmark Gate (mục 6) chạy PASS trên phần cứng thật.
> **Tài liệu này thay thế cho việc đọc riêng lẻ:** `AI-Earbud-Copilot-Tong-Hop.md` (đặc tả, 5 vòng review) + `AI-Earbud-Copilot-Ke-Hoach-Trien-Khai.md` (module/task). Hai file gốc vẫn giữ để tra cứu chi tiết lịch sử quyết định.

---

## 1. Tổng quan Dự án

### 1.1. Mục tiêu
Xây dựng một trợ lý hội thoại thời gian thực chạy qua tai nghe: nghe người đối diện nói → bóc băng → dịch + hiểu hàm ý → gợi ý câu trả lời → **đọc gợi ý qua tai nghe**. Toàn bộ xử lý AI chạy **local/offline** trên GPU 6GB VRAM.

### 1.2. Ràng buộc cứng
| Ràng buộc | Chi tiết |
|---|---|
| Hạ tầng | 100% offline/local, GPU 6GB VRAM (có phương án nâng cấp Cloud GPU 24GB) |
| Độ trễ | End-to-end (nói xong → có gợi ý) mục tiêu P95 < 1.5s |
| Số người dùng | 1 client đồng thời cho MVP |
| Đầu ra | Text (hiển thị) **và** Audio (TTS qua tai nghe) |

### 1.3. Bản chất kỹ thuật thật sự của hệ thống
Đây **không phải** một chatbot pipeline một chiều đơn giản. Sau khi bổ sung TTS, hệ thống trở thành **audio hệ thống hai chiều đóng vòng**:

```text
Microphone → AI (VAD → STT → LLM → TTS) → Earbuds
     ▲                                         │
     └─────────────── (audio feedback) ────────┘
```

Điều này kéo theo 3 lớp vấn đề kiến trúc phải giải quyết đồng thời — không thể bỏ qua bất kỳ lớp nào:
1. **Latency thời gian thực** dưới ràng buộc GPU 6GB (Compute Contention giữa Whisper/LLM)
2. **Tài nguyên đa engine** (Whisper + LLM trên GPU, TTS trên CPU để tránh vượt trần VRAM)
3. **Vòng lặp audio** (Barge-in — TTS đang phát phải tự hủy khi người đối diện nói tiếp, tránh Whisper nhận nhầm audio TTS làm input mới)

---

## 2. Kiến trúc Tổng thể (bản cuối)

```text
Audio
  ↓
VAD (Silero, endpoint detection)
  ↓
Utterance State (utterance_id, partial/final)
  ↓
STT — Faster-Whisper Small (pseudo-streaming / sliding-window, GPU)
  ↓
LLM — Qwen2.5-3B-Instruct Q4_K_M (streaming, GPU)
  ↓
LLM Output Parser → Semantic Events (translation_delta, intent_done, reply_ready)
  ├──────────────────────────────► WebSocket → UI (text, hiển thị ngay)
  │
  └──────────────────────────────► TTS — Piper (CPU, async)
                                          ↓
                                       Playback qua tai nghe
                                          ↓
                                       Barge-in ──► quay lại VAD (đóng vòng lặp)
```

**Nguyên tắc thiết kế xuyên suốt:**
- Text hiển thị **không phụ thuộc** vào TTS hoàn tất (TTS là nhánh song song, không chặn E2E của phần text).
- TTS chạy **CPU**, không chạm GPU — tránh compute contention 3 chiều với Whisper + LLM vốn đã sát trần VRAM.
- Mọi audio phát ra tai nghe đều phải có khả năng **bị hủy tức thì** khi phát hiện người đối diện nói tiếp (Barge-in).

### 2.1. Kiến trúc 3 tầng

| Tầng | Thành phần |
|---|---|
| **Audio** | Mic capture → VAD → Speech segmentation → Utterance State |
| **Intelligence** | STT → LLM (streaming) → Output Parser → TTS |
| **Presentation** | Semantic Events → WebSocket → UI / Audio playback |

### 2.2. Mô hình vòng đời (Model Runtime vs Session)
```text
Application
   │
   ├── Model Runtime (shared infra — Whisper, LLM, TTS)
   │      Contract: không terminate khi còn inference job đang dùng
   │
   └── Sessions (consumer)
          MVP: max_concurrent_sessions = 1
```

---

## 3. Ngân sách Tài nguyên

### 3.1. VRAM (GPU 6GB)
| Thành phần | VRAM | Ghi chú |
|---|---|---|
| STT (Faster-Whisper Small, int8_float16) | ~1.5–1.8GB | |
| LLM (Qwen2.5-3B-Q4_K_M, `num_ctx=2048`) | ~2.3–2.5GB + ~0.5GB KV cache | |
| Hệ thống/CUDA overhead | ~0.8GB | |
| **TTS (Piper, CPU)** | **Negligible/excluded — không dùng GPU** | Không viết "0GB" như một guarantee vật lý tuyệt đối |
| **Expected** | **~4.5–5.0GB** | Không coi ~5.1GB là "an toàn" — chỉ là tham khảo |
| **Hard ceiling** | **~5.5GB** | Vượt ngưỡng → degrade/reject/unload |

### 3.2. Compute Contention (rủi ro vật lý cốt lõi)
Trên GPU 6GB (16–28 SMs), Whisper và LLM chạy chồng lấn sẽ tranh chấp Streaming Multiprocessors, có thể làm tốc độ mỗi bên tụt 30–50%. **Đây là lý do bắt buộc phải benchmark kịch bản đồng thời, không chỉ đo riêng lẻ** (xem mục 6).

### 3.3. CPU (TTS)
TTS chạy CPU tránh được xung đột VRAM/SM, nhưng vẫn cần benchmark riêng: CPU saturation, event-loop lag khi STT+LLM+TTS chạy cùng lúc.

---

## 4. Target Component Latency (không phải "ngân sách đảm bảo")

| Thành phần | Target |
|---|---|
| VAD endpoint | < 150ms |
| STT (chạy riêng lẻ) | < 400ms |
| LLM total generation (chạy riêng lẻ) | < 500ms |
| LLM TTFT | < 200ms |
| TTS synthesis (câu <10 từ, CPU) | < 300–400ms |
| Barge-in response time | < 200ms |

> Các con số trên là **target theo thành phần**, không phải tổng cộng dồn = E2E. VAD silence threshold (300–500ms, để tránh cắt câu khi người nói ngập ngừng) cộng thẳng vào E2E — ngân sách thực tế cho STT+LLM chỉ còn ~800–900ms trong tổng 1.5s. **E2E thực chỉ được xác nhận qua đo Concurrent Inference (mục 6), báo cáo theo percentile, không phải cộng các con số thành phần.**

---

## 5. Event Protocol

### 5.1. Event types
```text
session_started · audio_started
stt_partial · stt_final
copilot_started · copilot_delta · copilot_done
tts_started · tts_audio_chunk · tts_done · tts_cancelled · tts_error
error · session_ended
```

### 5.2. Cấu trúc message bắt buộc
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
`utterance_id` bắt buộc vì một session có nhiều utterance, và TTS của utterance cũ cần bị `tts_cancelled` chính xác khi utterance mới xuất hiện (Barge-in).

### 5.3. Nguyên tắc streaming
- LLM sinh token stream thô → **backend parse thành semantic event** (`translation_delta`, `intent_done`, `reply_ready`) → frontend **không bao giờ** nhận JSON thô đang dở dang của LLM.
- Output LLM tối giản: bỏ trường `meaning` lặp lại cho từng reply để giảm token → giảm latency.

### 5.4. Barge-in / TTS State Machine
```text
IDLE → SYNTHESIZING → PLAYING → DONE
                          │
                 (speech mới phát hiện)
                          ▼
                    INTERRUPTED → stop TTS → resume listening
```

---

## 6. Benchmark Gate (bắt buộc — Ngày 1, trước khi code Pipeline)

| # | Test | Target |
|---|---|---|
| 1 | VAD endpoint | < 150ms |
| 2 | STT riêng lẻ | < 400ms |
| 3 | LLM riêng lẻ (TTFT / total) | < 200ms / < 500ms |
| 4 | TTS riêng lẻ (CPU) | < 300–400ms |
| 5 | VRAM (Whisper + LLM active) | < 5.5GB |
| 6 | **Concurrent Inference E2E** (20–50 utterance, Whisper+LLM chồng lấn) | P50 < 1.0s · P90 < 1.3s · P95 < 1.5s · Max < 2.0s · error rate < 2% |
| 7 | CPU stress test (STT+LLM+TTS đồng thời) | Không nghẽn event loop / WebSocket |
| 8 | **Audio feedback loop** (TTS phát trong lúc mic thu) | Đo false-trigger rate của Whisper (baseline, dùng thiết kế Barge-in) |
| 9 | **Barge-in response time** | < 200ms |
| 10 | **Speaker-source validation** (user/other/both/TTS/noise) | Ghi nhận WER + false activation rate từng kịch bản |
| 11 | Web mic vs Bluetooth mic (4 kịch bản, gồm BT+TTS playback) | Ghi nhận WER/latency/noise robustness |

```text
Nếu bất kỳ mục nào FAIL:
   STOP → tối ưu model/config → đo lại
   (chưa code tiếp Pipeline/Frontend/Mobile)
```

---

## 7. Quyết định MVP Scope (đã chốt)

| Quyết định | Nội dung |
|---|---|
| Hướng thu âm | **Assumption:** micro ưu tiên thu giọng người đối diện. Nếu Benchmark Gate mục 10 cho thấy assumption sai → audio routing/mic placement trở thành P0 thiết kế lại, không phải task phụ. |
| Đọc tự động qua TTS | **Translation → AUTO.** **Intent → hiển thị UI, TTS optional (không tự động).** **Quick reply → chỉ đọc khi người dùng chọn thủ công.** (Tránh audio TTS chồng lấp hội thoại thật khi đối phương nói liên tục nhiều câu.) |
| Sliding-window STT | 1.5s là **ngưỡng tối thiểu để trigger partial inference**, không phải speech boundary. VAD endpoint mới là điều kiện bắt buộc cho final STT — áp dụng cho cả câu ngắn dưới 1.5s. |
| Session/Model coupling | Model Runtime là hạ tầng dùng chung, Session là consumer — không để vòng đời model gắn chặt vòng đời 1 kết nối client. |
| Event Bus | `asyncio.Queue` nội bộ — không dùng message broker thật (Kafka/RabbitMQ/Redis) ở MVP. |

---

## 8. Module Backend (tóm tắt — chi tiết đầy đủ ở tài liệu Kế hoạch Triển khai)

```text
backend/app/
├── core/        config.py · runtime.py (Model Runtime) · vram_manager.py
├── audio/       vad.py · chunker.py (sliding-window contract) · session.py (utterance state)
├── ai/          stt.py · llm.py · copilot.py (semantic event parser) · tts.py (CPU, barge-in)
├── protocol/    events.py (Event Bus) · schemas.py
└── api/         websocket.py (/ws/copilot)
```

`docker-compose.yml`, `docs/`, `models/`, `client/web_test/`, `client/mobile_app/` giữ nguyên cấu trúc gốc.

---

## 9. Rủi ro & Trạng thái Xử lý (Risk Register hợp nhất)

| Rủi ro | Trạng thái | Giải pháp |
|---|---|---|
| Fixed 1-second buffer làm gãy câu | ✅ Đã xử lý | VAD-driven speech segmentation |
| VRAM 5.1GB coi là "an toàn" | ✅ Đã xử lý | Benchmark thật, Expected/Hard-ceiling động |
| LLM batch request (chờ full JSON) | ✅ Đã xử lý | Streaming + semantic event parser |
| `sleep(2)` giả định server ready | ✅ Đã xử lý | Health-check polling |
| Compute Contention GPU 6GB | ✅ Đã đưa vào Benchmark Gate | Concurrent Inference E2E test (percentile) |
| Sliding-window STT gây GPU tải 100% | ✅ Đã xử lý | Giới hạn tần suất trigger theo VAD, không transcribe liên tục |
| TTS cạnh tranh VRAM/SM với Whisper+LLM | ✅ Đã xử lý | TTS bắt buộc chạy CPU |
| Audio feedback loop (TTS lọt vào mic) | ✅ Đã đưa vào Benchmark Gate + thiết kế | Barge-in state machine, test riêng (mục 6.8) |
| Utterance quản lý bất đồng bộ khi có nhiều utterance | ✅ Đã xử lý | `utterance_id` trong mọi event |
| Audio dồn dập nếu tự động đọc mọi gợi ý | ✅ Đã xử lý | MVP scope: chỉ auto-đọc translation |
| **False Activation** (mic gần miệng user hơn đối phương) | 🟡 P0 validation, chưa có giải pháp kỹ thuật thật | Benchmark mục 6.10 sẽ quyết định có cần thiết kế lại audio routing |
| **Bluetooth duplex** (A2DP vs HFP/SCO khi mic active) | 🟡 P0 validation | Benchmark mục 6.11, có thể là vấn đề tầng OS |
| Production readiness (auth WebSocket, đa client, error fallback) | 🟡 Chưa giải quyết — không chặn MVP | Thiết kế thêm trước khi lên production, không cần ngay ở giai đoạn benchmark |

---

## 10. Roadmap & Milestones

```text
Ngày 1        Ngày 2-3       Ngày 4-7          Ngày 8-10      Ngày 11-15
Benchmark  →  Core Runtime → Audio/AI/         Web Client  →  Bluetooth      →  Mobile
Gate          + VRAM          Protocol                        Validation
   │
   └─ FAIL → tối ưu model/config → đo lại (không tiến Phase tiếp theo)
```

| Giai đoạn | Nội dung chính | Điều kiện bắt đầu |
|---|---|---|
| **Ngày 1 — Benchmark Gate** | Chạy đủ 11 test ở mục 6, quyết định PASS/FAIL | Model + hardware sẵn sàng |
| **Ngày 2–3 — Core** | Model Runtime, VRAM manager, health-check | Benchmark Gate PASS |
| **Ngày 4–7 — Pipeline** | VAD-driven audio pipeline, STT/LLM/TTS streaming, semantic events, Barge-in | Core hoàn thành |
| **Ngày 8–10 — Web Client** | Demo đầy đủ: nói → thấy gợi ý → nghe TTS → Barge-in hoạt động | Pipeline hoàn thành |
| **Ngày 11–15 — Bluetooth + Mobile** | Duplex test, speaker validation trên phần cứng thật, sau đó mới build mobile app | Web Client demo ổn định |

**Chi tiết task-level (ID, module, acceptance criteria) xem tài liệu `AI-Earbud-Copilot-Ke-Hoach-Trien-Khai.md`.**

---

## 11. Đánh giá Tổng thể (điểm số cuối cùng sau 5 vòng review)

| Hạng mục | Điểm |
|---|---|
| Kiến trúc tổng thể | 8.5/10 |
| Tính khả thi MVP | 8/10 |
| Realtime pipeline | 7.5/10 |
| Audio architecture (sau khi thêm Barge-in) | 7.5/10 |
| TTS integration (sau khi thêm state machine + cancellation) | 8/10 |
| Benchmark design (sau khi chuyển sang percentile) | 8.5/10 |
| Implementation readiness | 8/10 |
| Production readiness | 4/10 (chưa cần giải quyết ở giai đoạn MVP) |

**Kết luận:** Kiến trúc đã đủ chi tiết, đã xử lý các bẫy kỹ thuật phổ biến của dự án AI realtime (nhầm lẫn streaming thật/giả, VRAM fragmentation, cold start, audio feedback loop, false activation). Không cần thêm vòng review kiến trúc. Bước tiếp theo duy nhất: **chạy Benchmark Gate trên phần cứng thật** — nếu PASS, chuyển thẳng sang implementation theo roadmap ở mục 10 và task backlog chi tiết.

---

## 12. Tài liệu liên quan

| Tài liệu | Nội dung |
|---|---|
| `AI-Earbud-Copilot-Tong-Hop.md` (v4.1.0) | Đặc tả kiến trúc đầy đủ, lịch sử 5 vòng review, phân tích chi tiết từng quyết định kỹ thuật |
| `AI-Earbud-Copilot-Ke-Hoach-Trien-Khai.md` | Module breakdown (13 module) + Task backlog (~50 task có ID, acceptance criteria, dependency) |
| Tài liệu này | Bản tổng hợp một trang cho toàn dự án — dùng làm điểm khởi đầu tham chiếu nhanh |
