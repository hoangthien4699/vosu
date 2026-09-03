# Hướng dẫn chạy thử

Ba mức, từ nhẹ tới nặng. Mức 1 không cần gì cả; mức 3 cần model đã tải.

---

## Mức 1 — Unit test (30 giây, không cần model, không cần GPU)

```bash
make test
```

101 test. Cố ý chạy được **không cần model**: VAD có backend `energy` thay thế,
STT/LLM/TTS đều có bản giả lập. Nếu phải tải 2GB mới chạy được test thì sẽ
không ai chạy.

Đáng đọc trước khi sửa code:

| File | Khóa điều gì |
|---|---|
| `test_chunker.py` | contract 1.5s vs VAD endpoint |
| `test_json_stream.py` | parser chịu được output méo của model 4-bit |
| `test_copilot_parser.py` | không rò JSON thô ra frontend |
| `test_prompt_template.py` | Gemma vs ChatML, stop token |
| `test_runtime.py` | shutdown chờ inference job |
| `test_tts.py` | Barge-in < 200ms, posix_spawn |
| `test_websocket_e2e.py` | thứ tự event, envelope đầy đủ |
| `test_web_client.py` | UI phục vụ đúng, không hỏng im lặng |

---

## Mức 2 — Chạy UI (cần model)

### Chuẩn bị một lần

```bash
bash scripts/setup-macos.sh        # hoặc setup-cuda.sh
bash scripts/make_test_audio.sh    # audio mẫu giọng thật
```

### Chạy

```bash
make run
```

Mở **http://localhost:8000** (tự chuyển sang `/app/`).

Kiểm tra server sẵn sàng trước khi thao tác:

```bash
curl -s localhost:8000/health | python3 -m json.tool
# ready: true, llama_server: true, stt_loaded: true
```

### Hai cách thử trên UI

**Cách A — phát file (thử một mình được):**

1. Bấm **"Phát file…"**
2. Chọn một file trong `benchmarks/audio/` (ví dụ `sample_03.wav`)
3. Xem transcript → bản dịch → hàm ý → gợi ý trả lời hiện dần

File được phát qua **đúng đường mà micro đi** — cùng resample về 16kHz, cùng
chunk 100ms, cùng WebSocket — nên nó kiểm chứng pipeline thật, không phải
đường tắt.

#### Chế độ "nghe lại từng câu" (bật mặc định)

Với file có nhiều câu, phát liên tục sẽ khiến câu sau đến khi câu trước còn
đang xử lý — backend hủy utterance cũ để nhường utterance mới, và kết quả trên
màn hình bị thay giữa chừng. Đo trên file 3 câu:

| | Phát liên tục | Từng câu một |
|---|---|---|
| Câu nhận diện | 3 | 3 |
| **TTS bị hủy giữa chừng** | **1** | **0** |
| Thời gian | 14s | 26s |

Khi bật, hệ thống **tự dừng phát ngay khi chốt được một câu**, và chỉ phát tiếp
sau khi câu đó xử lý xong hoàn toàn — nghĩa là LLM sinh xong **và** không còn
lượt TTS nào đang đọc. Chờ mỗi `copilot_done` là chưa đủ: một bản dịch nhiều
câu sẽ có nhiều lượt `tts_started`/`tts_done`.

Với mỗi câu, hệ thống phát lại **bốn bước theo thứ tự**:

| | Nghe gì | Giọng |
|---|---|---|
| 1 | **Âm thanh gốc** — cắt đúng đoạn của câu đó từ chính file bạn chọn | (file gốc) |
| 2 | **Bản dịch** | tiếng Việt |
| 3 | **Hàm ý** | tiếng Việt |
| 4 | **Gợi ý trả lời**: *"Có hai lựa chọn cho bạn. Một là: … Mục đích là … Hai là: …"* | xen kẽ |

Bước 4 đổi giọng theo từng đoạn: khung dẫn và mục đích đọc giọng Việt, còn câu
gợi ý đọc giọng **Anh** — đó là ngôn ngữ bạn sẽ nói ra. Một giọng đọc cả hai
thì phần tiếng Anh nghe rất khó hiểu.

Client cắt được đúng đoạn audio gốc nhờ trường `start_s` trong `stt_final`:
server đếm vị trí câu trên chính dòng byte mà client gửi, nên hai bên khớp
tuyệt đối. Đo trên file 3 câu: `start_s` = 0.416 / 3.552 / 7.104s, khớp đúng
cách ghép file (0.4 / 3.5 / 7.0s).

Ở chế độ này client đặt server sang **`set_tts_mode: manual`** — server không
tự đọc gì, client quyết thứ tự. Nếu để server tự đọc theo §2.4.1 thì bản dịch
sẽ phát *trước cả* âm thanh gốc, vì streaming TTS bắt đầu ngay khi có câu đầu.

Nút **"Tạm dừng"** cho bạn kiểm soát tay bất cứ lúc nào. Bấm "Tiếp tục" sẽ hủy
chuỗi nghe lại đang dở và phát tiếp ngay.

Thời lượng thực tế mỗi câu: khoảng **20 giây** (3s bản dịch + 2s hàm ý + 14s
gợi ý trả lời qua 5 đoạn TTS).

> **Chi phí của trường "mục đích".** §4.4 của đặc tả cố ý bỏ trường mô tả cho
> từng reply vì token thêm làm tăng latency. Đo lại trên Gemma 3 4B / M4:
> first-useful-result 198ms → 282ms, tổng thời gian sinh 1759ms → 2972ms
> (+69%). Tắt bằng `llm.reply_purpose: false` nếu ưu tiên tốc độ cho hội thoại
> trực tiếp.

Tạm dừng an toàn nhờ một tính chất của kiến trúc: **VAD phía server chạy theo
frame, không theo đồng hồ thực**. Ngừng gửi audio thì trạng thái VAD đóng băng
nguyên vẹn, gửi tiếp là chạy đúng chỗ cũ — không mất câu, không cắt nhầm biên.
Đã kiểm chứng: cả hai chế độ đều nhận diện đúng 3/3 câu.

**Cách B — micro thật:**

1. Bấm **"Bắt đầu nghe"**, cho phép trình duyệt dùng micro
2. Nói một câu **tiếng Anh** (sản phẩm nghe *người đối diện* nói ngoại ngữ)
3. Dừng nói ~0.5 giây để VAD chốt câu

### Thử Barge-in

Trong lúc AI đang đọc bản dịch, **nói chen vào**. TTS phải dừng ngay, và nhật
ký hiện `tts_cancelled` kèm `reason: barge_in` và thời gian phản hồi.

Hoặc bấm **"Dừng đọc"** để thử đường `client_request`.

### Đọc gì trên màn hình

| Khu vực | Ý nghĩa |
|---|---|
| **Đang nghe** | transcript tạm (xám nghiêng) rồi transcript cuối |
| **Bản dịch** | hiện dần theo từng ký tự — đây là `translation_delta` |
| **Hàm ý** | người nói thực sự muốn gì |
| **Gợi ý trả lời** | **bấm vào để nghe đọc** — đúng ngôn ngữ người đối diện dùng |
| **Chỉ số** | E2E, TTFT, STT, Barge-in — đo trực tiếp |
| **Nhật ký** | semantic event thô, để debug |

Nhật ký cố ý hiển thị **semantic event**, không phải JSON của LLM — nếu bạn
thấy dấu ngoặc nhọn hay tên trường JSON ở đó thì đó là lỗi.

---

## Mức 3 — Benchmark Gate

```bash
make gate                                          # đủ B1-B10
python -m benchmarks.run_gate --skip-interactive   # bỏ B7/B9/B10
python -m benchmarks.b5_concurrent_e2e -n 30       # chạy riêng
```

So sánh hai model LLM trên chính prompt của dự án:

```bash
python -m benchmarks.compare_models \
    --model models/gemma-3-4b-it-q4_k_m.gguf \
    --model models/qwen2.5-3b-instruct-q4_k_m.gguf
```

> Trên macOS, `run_gate` in cảnh báo đậm: đặc tả yêu cầu chốt Gate trên GPU
> NVIDIA 6GB. Kết quả trên Mac chỉ để phát triển.

---

## Khi có gì đó không chạy

| Triệu chứng | Nguyên nhân thường gặp |
|---|---|
| `/health` trả `ready: false` | Xem lý do trong JSON trả về; thường là thiếu model |
| Trang trắng, không CSS | Đang mở sai đường dẫn — dùng `/` hoặc `/app/` |
| Nói mà không có gì xảy ra | Thiếu `models/silero_vad.onnx` → VAD rơi về backend `energy`, kém nhạy hơn nhiều. Chạy `scripts/download_models.sh` |
| Có transcript nhưng không có bản dịch | `llama-server` chưa chạy. `curl localhost:8080/health` |
| Có bản dịch nhưng không nghe thấy gì | Thiếu Piper. Nhật ký sẽ có `tts_error` nói rõ |
| `Không tìm thấy binary llama-server` | `brew install llama.cpp` rồi chạy lại `scripts/download_models.sh` |
| Port 8080 bị chiếm | `pkill -f llama-server` |

Log server nói khá rõ — chạy với `VOSU_SERVER__LOG_LEVEL=DEBUG` nếu cần chi tiết.
