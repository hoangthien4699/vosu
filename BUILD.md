# Nhánh `release` — build NVIDIA/CUDA

Nhánh triển khai cho GPU NVIDIA. **Đây là build mục tiêu của đặc tả** và là
build duy nhất được dùng để chốt Benchmark Gate (§6).

## Khác gì so với `main`

Chỉ ba file. Code Python **giống hệt** `main`:

| File | Khác biệt |
|---|---|
| `config.yaml` | Được commit, ghim `platform: cuda`, `whisper_model: small` |
| `.gitignore` | Không còn bỏ qua `config.yaml`; cấu hình cục bộ chuyển sang `config.local.yaml` |
| `BUILD.md` | File này |

Toàn bộ khác biệt phần cứng nằm trong `DeviceProfile`
([`backend/app/core/device.py`](backend/app/core/device.py)) và được phân giải
lúc chạy — không có nhánh code riêng nào cần bảo trì.

## Cấu hình có hiệu lực

| | Giá trị | Vì sao |
|---|---|---|
| STT | `cuda` / `int8_float16` | GPU rời, có cuDNN |
| LLM | `-ngl 36` | Đủ offload Qwen2.5-3B lên 6GB |
| Whisper | `small` | §3.1 — không mặc định lên Medium |
| LLM model | Gemma 3 4B | Qwen trả reply sai ngôn ngữ 3/6 ca |
| `--swa-full` | bật | Bắt buộc với Gemma 3 (SWA), nếu không TTFT tệ 7 lần |
| VRAM hard ceiling | 5.5GB | §3.1, enforce qua `nvidia-smi` |

> **Rủi ro VRAM chưa được kiểm chứng.** Gemma 3 4B tốn thêm ~451 MB so với
> Qwen2.5-3B (RSS 2459 so với 2008 MB, đo trên Metal). Ngân sách ở §3.1 tính
> cho Qwen và đã sát mép 5.5GB, nên **B4 trên nhánh này là mục bắt buộc phải
> chạy trước tiên**. Nếu vượt trần, thứ tự hạ: `whisper_model: base` →
> `n_ctx: 1536` → quay lại Qwen.

## Yêu cầu hệ thống

Không cài được qua pip:

- NVIDIA driver + CUDA 12.x + cuDNN 9 (CTranslate2 cần)
- `llama-server` build với `-DGGML_CUDA=ON` → liên kết vào `models/llama-server`

## Cài và chạy

```bash
bash scripts/setup-cuda.sh     # kiểm tra GPU, cài deps, tải model
make gate                      # BẮT BUỘC trước khi làm gì khác
make run                       # nếu Gate PASS
```

Hoặc Docker:

```bash
docker compose up
```

## Benchmark Gate

Nhánh này là nơi Gate được chốt. Nếu bất kỳ mục P0 nào FAIL:

```
STOP -> tối ưu model/config -> đo lại
(chưa code tiếp Pipeline/Frontend/Mobile)
```

B7/B9/B10 cần người vận hành thao tác theo hướng dẫn trên màn hình. **B9
(speaker-source validation) là mục quyết định** có phải thiết kế lại audio
routing hay không — mic earbud nằm gần miệng người dùng, trong khi yêu cầu
sản phẩm là nghe người đối diện.

## Nhận thay đổi từ `main`

```bash
git checkout release
git merge main
```

Code không phân kỳ nên merge gần như không bao giờ xung đột. Nếu có, gần
chắc chắn là ở `config.yaml` — giữ bản của nhánh này.
