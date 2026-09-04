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

#### Cắt câu: dựa vào nội dung, không chỉ độ dài khoảng lặng

Chỉ đo khoảng lặng thì **không** tách được "ngập ngừng giữa câu" với "đã nói
xong". Đo thật: khoảng ngập ngừng giữa câu của người nói chậm (800ms) còn dài
hơn khoảng nghỉ giữa hai câu của người nói bình thường (700ms).

Nên sau khi STT xong, hệ thống xét luôn transcript có trông như một câu trọn
nghĩa không (`backend/app/ai/completeness.py`). Ba tín hiệu, lấy miễn phí:

| tín hiệu | ví dụ | bắt được |
|---|---|---|
| không có dấu kết câu | `"So what I am trying to say is"` | 25% |
| dấu ba chấm — bỏ lửng | `"Tôi nghĩ là chúng ta nên..."` | +33% |
| từ cuối không đứng cuối câu được | `"Before we sign anything I want to."` | +34% |

Riêng dấu câu là không đủ: Whisper vẫn chấm câu cho mảnh dở. Kết hợp cả ba,
đo trên bộ thử ngập ngừng thật:

```
bắt được câu dở          11/12  (92%)
không giữ nhầm câu trọn  32/32  (100%)
```

Nghe ra câu dở thì giữ lại, chờ tối đa `stt.merge_window_ms` (1200ms). Có nói
tiếp thì **ghép audio rồi nghe lại trên đoạn liền** — nghe lại chứ không nối
hai transcript, vì Whisper trên audio liền mạch cho ra câu đúng ngữ pháp hơn.
Không ai nói tiếp thì vẫn dịch nguyên câu dở: người ta có quyền bỏ lửng câu.

Cửa sổ chờ đo bằng **đồng hồ audio**, không phải đồng hồ thật, và neo vào chỗ
đoạn audio kết thúc. Đo bằng đồng hồ thật thì hỏng ở đúng chế độ dừng-từng-câu:
client dừng file ngay tại endpoint nên trong lúc chờ không có audio nào chạy,
cửa sổ tự hết giờ trước khi đoạn nói tiếp kịp tới. Ngoài ra không bỏ chờ khi
VAD đang thu tiếng nói, cũng không bỏ khi đoạn nói tiếp đã chốt endpoint và
đang xếp hàng — cả ba đều đã đo thấy làm câu bị cắt đôi.

Đo A/B trên file 6 câu trọn: bật hay tắt đều ~6.0s, chênh lệch nằm trong nhiễu
— tính năng không cộng gì vào câu bình thường.

Thử bằng `benchmarks/audio/ngat_giua_cau.wav` (dựng bằng script trong
`benchmarks/`): hai câu, mỗi câu bị ngắt 0.9s giữa chừng.

#### Chế độ "dừng từng câu để đọc bản dịch" (bật mặc định)

File **phát ra loa** đúng nhịp thời gian thực, đồng bộ với dòng byte gửi lên
server. Phát tới hết câu nào thì **dừng ngay tại đó**, đọc bản dịch của chính
câu đó, xong mới phát tiếp:

```
… file phát …
[câu 1 dứt]  -> DỪNG -> đọc bản dịch câu 1 -> phát tiếp
[câu 2 dứt]  -> DỪNG -> đọc bản dịch câu 2 -> phát tiếp
```

Không phát lại âm thanh gốc: bạn vừa nghe câu đó lúc file chạy tới rồi.

**Câu bị ngắt giữa chừng thì file phát tiếp, không dừng.** Nếu server nghe ra
một câu còn dở ("So what I am trying to say is…") thì nó phát
`utterance_continued`, client phát tiếp cho tới chỗ dứt thật, rồi hai đoạn được
ghép lại và dịch một lần. Không có cái này thì file dừng vĩnh viễn — audio cần
để quyết định sẽ không bao giờ tới.

**Mốc dừng lấy từ `utterance_endpoint`, không phải `stt_final`.** Server phát
`utterance_endpoint` ngay khi VAD chốt câu, trước cả khi Whisper chạy. Đợi
`stt_final` thì muộn hơn 1.8-2.1s (đo thật) — file đã phát lấn sang câu sau
rồi mới dừng, và người dùng nghe bản dịch câu trước chồng lên câu sau.

Đo trên file 6 câu: `utterance_endpoint` tới sau lúc người ta thật sự ngừng
nói khoảng 0.65-0.71s, tức là rơi gọn vào khoảng lặng giữa hai câu.

**Nhịp gửi lấy theo đồng hồ AudioContext, không phải `setTimeout`.** Sai số
vài phần trăm mỗi nhịp của `setTimeout` cộng dồn thành cả giây sau 20 giây,
làm cái nghe được lệch hẳn với cái đã gửi lên server — dừng sẽ dừng sai chỗ.

Bỏ tick thì phát liền mạch. Lúc đó bản dịch câu trước sẽ đọc chồng lên câu
sau đang phát, vì độ trễ đầu-cuối (~5.8s) dài hơn khoảng cách giữa hai câu.

Chiều dịch suy từ ngôn ngữ Whisper nhận diện, không cần bấm nút:

| Ai nói | Làm gì | Giọng đọc |
|---|---|---|
| **Đối phương** (tiếng Anh…) | dịch sang tiếng Việt để bạn **hiểu** | Việt, tốc độ thường |
| **Bạn** (tiếng Việt) | dịch sang tiếng đối phương để bạn **nói theo** | Anh, **đọc chậm** |

Ngôn ngữ đối phương lấy theo thực tế nghe được, không cứng là tiếng Anh — họ
nói tiếng Nhật thì chiều ngược dịch sang tiếng Nhật.

Câu ngắn ("Yes.", "Ừ.") thì Whisper hay nhận nhầm ngôn ngữ, nên khi không chắc
hệ thống **giữ chiều của lượt trước** thay vì đoán bừa.

Ở chế độ này client đặt server sang **`set_tts_mode: manual`** — server không
tự đọc gì, client quyết khi nào đọc. Nếu để server tự đọc theo §2.4.1 thì nó
bắt đầu đọc ngay khi có câu dịch đầu tiên, không đợi file dừng.

Phát tiếp xong thì **tua nhanh qua đoạn im lặng**. File phát tiếp từ GIỮA
khoảng nghỉ giữa hai câu — người nói thật nghỉ 2-4 giây mà điểm dừng chỉ ăn
0.7s đầu, nên phần còn lại là im lặng phải ngồi nghe. Đo trên file nghỉ 2.5s:

```
không tua : 1877ms chết sau mỗi lượt đọc
tua 8x    :  283ms
```

Gửi nhanh hơn thời gian thực KHÔNG ảnh hưởng VAD vì nó đếm theo mẫu audio chứ
không theo đồng hồ — nhưng chỉ đúng khi vẫn gửi ĐỦ MỌI MẪU, chỉ là gửi nhanh.
Gặp chunk có tiếng thì lập tức về đúng nhịp thật và phát ra loa, nên không bỏ
sót đoạn nào cần nghe. Kiểm trên cả file nghỉ dài lẫn nghỉ ngắn: transcript
đủ câu, không câu nào cụt.

Khoảng dừng ở mỗi câu dài bao lâu, đo trên file 6 câu:

```
dừng file -> nghe xong (Whisper)   ~2.1s
          -> dịch xong (LLM)       ~2.1s
          -> có tiếng đọc          ~0.5s   (Piper, sau khi hâm nóng sẵn)
          đọc bản dịch             ~2.6-3.5s
          -> phát tiếp             ~0ms
```

Tức im lặng ~4.7s rồi mới nghe bản dịch. Đây là giá của pipeline, không phải
lỗi — trừ một khoản đã cắt được: Piper vốn spawn tiến trình mới và nạp lại
model cho TỪNG câu, mất ~500ms mỗi lần. Nay sau khi đọc xong một câu, hệ
thống spawn sẵn tiến trình cho lượt sau trong lúc còn rảnh. Trường
`prewarmed` trong `tts_done` cho biết lượt nào dùng được (câu đầu thì không,
vì chưa có gì để hâm).

Nút **"Tạm dừng"** cho bạn kiểm soát tay bất cứ lúc nào. Bấm "Tiếp tục" sẽ hủy
lượt đọc đang dở và phát tiếp ngay.

Tốc độ đọc chiều ngược chỉnh bằng `tts.coach_length_scale` (mặc định 1.35 —
số càng lớn càng chậm).


> **GBNF grammar là bắt buộc, không phải tối ưu hóa.** JSON Schema chỉ kiểm
> soát *cấu trúc*, không cấm được `}` hay dấu ngoặc kép cong **bên trong**
> chuỗi — model viết `”}` giữa chuỗi rồi lảm nhảm tiếp mà JSON vẫn "hợp lệ".
> Đo với prompt có lịch sử hội thoại, 6 câu: `json_schema` **0/6 sạch**, GBNF
> **6/6**. Bật mặc định qua `llm.grammar`.

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

So sánh hai model LLM trên chính prompt của dự án. Dự án chỉ giữ MỘT model
(Qwen3.5-4B, xem `models/README.md`), nên muốn dùng công cụ này thì phải tải
thêm model để đối chiếu — nó nhận đường dẫn bất kỳ:

```bash
python -m benchmarks.compare_models \
    --model models/qwen3.5-4b-q4_k_m.gguf \
    --model <đường-dẫn-gguf-khác>
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
