# models/

Thư mục này **không được commit** (xem `.gitignore`) — model nặng hàng GB.

Chạy `scripts/download_models.sh` để tải đủ. Sau khi xong, cấu trúc phải là:

```
models/
├── llama-server                              # binary llama.cpp (tự build hoặc brew)
├── gemma-3-4b-it-q4_k_m.gguf                 # ~2.3 GB  <- model đang dùng
├── qwen2.5-3b-instruct-q4_k_m.gguf           # ~2.0 GB  <- giữ để đối chiếu
├── silero_vad.onnx                           # ~2 MB
└── piper/
    ├── vi_VN-vais1000-medium.onnx            # + .onnx.json
    └── en_US-lessac-medium.onnx              # + .onnx.json
```

Model Faster-Whisper **không** nằm ở đây — thư viện tự tải về cache của
HuggingFace (`~/.cache/huggingface`) theo tên trong `paths.whisper_model`.

## Vì sao Whisper Small, không phải Medium

§3.1 của đặc tả: bài toán là *"transcription đủ nhanh và đủ chính xác để LLM
hiểu intent"*, không phải *"transcription chính xác tuyệt đối"* — hai mục tiêu
tối ưu khác nhau. Chỉ nâng lên Medium nếu benchmark cho thấy Small thực sự
không đủ.

## Vì sao Qwen3.5-2B Q8_0

Đã thử ba model trên chính prompt của dự án. Qwen2.5-3B (đặc tả gốc) loại vì rò
tiếng Trung vào bản dịch tiếng Việt. Giữa Qwen3.5-2B và Gemma 3 4B, chất lượng
dịch ngang nhau sau khi sửa prompt — chọn Qwen3.5-2B vì bộ nhớ:

| | RSS | Ước tính tổng VRAM | Dự phòng dưới trần 5.5GB |
|---|---|---|---|
| Qwen3.5-2B Q8_0 | 2119 MB | 5.17 GB | **0.33 GB** |
| Gemma 3 4B Q4_K_M | 2617 MB | 5.50 GB | **0** |

**Q8_0 chứ không phải Q4.** Model 2B nhỏ nên lượng tử hóa 4-bit làm hụt chất
lượng dịch rõ hơn nhiều so với model lớn. Q8_0 gần như không mất gì, mà file
vẫn nhẹ hơn một model 4B nén Q4.

Đổi model là một dòng trong `config.yaml` — prompt template tự nhận diện theo
tên file. So sánh lại bằng `python -m benchmarks.compare_models`.
