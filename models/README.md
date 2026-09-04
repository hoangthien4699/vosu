# models/

Thư mục này **không được commit** (xem `.gitignore`) — model nặng hàng GB.

Chạy `scripts/download_models.sh` để tải đủ. Sau khi xong, cấu trúc phải là:

```
models/
├── llama-server                              # binary llama.cpp (tự build hoặc brew)
├── qwen3.5-4b-q4_k_m.gguf                    # ~2.5 GB  <- model duy nhất
├── silero_vad.onnx                           # ~2 MB
└── piper/
    ├── vi_VN-vais1000-medium.onnx            # + .onnx.json
    └── en_US-lessac-medium.onnx              # + .onnx.json
```

Model Faster-Whisper **không** nằm ở đây — thư viện tự tải về cache của
HuggingFace (`~/.cache/huggingface`) theo tên trong `paths.whisper_model`.

## Vì sao chỉ một model

Dự án chốt **Qwen3.5-4B Q4_K_M** và không giữ model dự phòng nào. Ưu tiên là
chất lượng dịch; các lựa chọn nhanh hơn đều đánh đổi bằng nghĩa:

| model | vì sao loại |
|---|---|
| Qwen2.5-3B Q4_K_M | model của đặc tả gốc — rò tiếng Trung vào bản dịch tiếng Việt |
| Gemma 3 4B Q4_K_M | dịch kém hơn trên chính prompt của dự án |
| Qwen3.5-2B Q8_0 | nhanh hơn ~1.7s mỗi câu, nhưng sai thành ngữ |

Ví dụ đo được với 2B: `"I think we should table this discussion"` ra
*"đặt cuộc thảo luận"* (hiểu "table" là "đặt"), còn 4B ra *"tạm gác lại"*.

Muốn so lại về sau thì tải thêm model rồi chạy
`python -m benchmarks.compare_models`; công cụ nhận đường dẫn bất kỳ.

## Vì sao Whisper Small, không phải Base hay Medium

Đo WER trên 20 câu Anh/Việt (`benchmarks/stt_wer.py`). `base` nghe nhầm kiểu
"thanh toán" → "thang tòa án", "bàn giao" → "bàn dào", và càng ồn càng giãn:

| SNR | base | small |
|---|---|---|
| sạch | 5.3% | 1.1% |
| 20dB (phòng họp thường) | 9.1% | **1.1%** |
| 10dB | 14.4% | 8.0% |
| 5dB | 27.8% | 16.6% |

Nghe sai một từ khoá rồi dịch sai hẳn nghĩa — tệ hơn nhiều so với chậm thêm
một giây. Chưa nâng lên Medium: §3.1 đặt mục tiêu *"đủ nhanh và đủ chính xác
để LLM hiểu intent"*, và Small chưa cho thấy là không đủ.

## Hệ quả VRAM khi chỉ còn 4B

Qwen3.5-4B (RSS ~2830 MB) + Whisper `small` (~1.8 GB) + KV (~0.5 GB) + CUDA
overhead (~0.8 GB) ≈ **5.9 GB**, vượt trần 5.5 GB của §3.1. Trước đây có hai
đường lui — hạ Whisper xuống `base`, hoặc quay về 2B — nay đã bỏ cả hai vì
đều đánh đổi bằng chất lượng. Nghĩa là build CUDA cần card **8 GB**, không
phải 6 GB. Chạy `benchmarks/b4_vram.py` để có số thật trên phần cứng NVIDIA.
