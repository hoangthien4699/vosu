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

## Vì sao Gemma 3 4B thay cho Qwen2.5-3B

Đặc tả v4.1.0 chọn Qwen2.5-3B. Khi chạy thật, Qwen rò tiếng Trung vào bản dịch
tiếng Việt — quan sát trực tiếp:

    "Tôi nghĩ chúng ta nên推迟这次讨论目前。"

Model 3B lượng tử hóa 4-bit không giữ vững ngôn ngữ đích. Gemma 3 hỗ trợ đa
ngôn ngữ tốt hơn đáng kể.

Đánh đổi cần theo dõi:

| | Qwen2.5-3B | Gemma 3 4B |
|---|---|---|
| File Q4_K_M | 2.0 GB | 2.3 GB |
| Prompt template | ChatML | `<start_of_turn>`, KHÔNG có vai trò system |
| Stop token | `<\|im_end\|>` | `<end_of_turn>` |

**+0.36GB nghe thì nhỏ, nhưng ngân sách VRAM ở §3.1 tính cho Qwen và đã sát
mép 5.5GB.** Bắt buộc chạy lại B4 trên phần cứng NVIDIA trước khi coi Gemma là
lựa chọn chốt. Nếu vượt trần: hạ Whisper xuống `base`, giảm `n_ctx`, hoặc quay
lại Qwen — đổi model chỉ là một dòng trong `config.yaml`, cả hai template đều
đã có sẵn.
