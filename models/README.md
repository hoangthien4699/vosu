# models/

Thư mục này **không được commit** (xem `.gitignore`) — model nặng hàng GB.

Chạy `scripts/download_models.sh` để tải đủ. Sau khi xong, cấu trúc phải là:

```
models/
├── llama-server                              # binary llama.cpp (tự build hoặc brew)
├── qwen2.5-3b-instruct-q4_k_m.gguf           # ~2.0 GB
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
