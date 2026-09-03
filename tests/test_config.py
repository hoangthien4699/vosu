"""Khóa hành vi của config loader — nơi lỗi âm thầm gây hại nhất."""
from __future__ import annotations

import pytest

from app.core.config import load_config


def test_vosu_config_la_bien_dieu_khien_khong_phai_khoa_config(tmp_path):
    """VOSU_CONFIG chọn FILE config, không được ánh xạ vào Config."""
    path = tmp_path / "c.yaml"
    path.write_text("llm:\n  port: 9999\n", encoding="utf-8")
    cfg = load_config(env={"VOSU_CONFIG": str(path)})
    assert cfg.llm.port == 9999


def test_ep_kieu_dung_cho_moi_kieu_du_lieu():
    cfg = load_config(env={
        "VOSU_LLM__PORT": "9090",              # int
        "VOSU_VAD__THRESHOLD": "0.65",         # float
        "VOSU_TTS__AUTO_READ_REPLIES": "true",  # bool
        "VOSU_STT__LANGUAGE": "en",            # Optional[str]
        "VOSU_LLM__N_GPU_LAYERS": "20",        # Optional[int]
        "VOSU_LLM__EXTRA_ARGS": "--a,--b",     # list[str]
    })
    assert cfg.llm.port == 9090 and isinstance(cfg.llm.port, int)
    assert cfg.vad.threshold == pytest.approx(0.65)
    assert cfg.tts.auto_read_replies is True
    assert cfg.stt.language == "en"
    assert cfg.llm.n_gpu_layers == 20
    assert cfg.llm.extra_args == ["--a", "--b"]


def test_khoa_sai_bi_tu_choi_thay_vi_am_tham_bo_qua():
    with pytest.raises(ValueError, match="VOSU_LLM__KHONG_CO"):
        load_config(env={"VOSU_LLM__KHONG_CO": "1"})


def test_device_profile_quyet_dinh_ngl_cho_tung_build():
    """Một codebase, hai build — khác biệt phải đi qua DeviceProfile."""
    assert load_config(env={"VOSU_PLATFORM": "cuda"}).llm_gpu_layers == 36
    assert load_config(env={"VOSU_PLATFORM": "macos"}).llm_gpu_layers == 99
    assert load_config(env={"VOSU_PLATFORM": "cpu"}).llm_gpu_layers == 0


def test_config_thang_device_profile_khi_dat_tuong_minh():
    cfg = load_config(env={"VOSU_PLATFORM": "cuda", "VOSU_LLM__N_GPU_LAYERS": "12"})
    assert cfg.llm_gpu_layers == 12


def test_stt_khac_nhau_giua_hai_build():
    cuda = load_config(env={"VOSU_PLATFORM": "cuda"}).device
    macos = load_config(env={"VOSU_PLATFORM": "macos"}).device
    assert (cuda.stt_device, cuda.stt_compute_type) == ("cuda", "int8_float16")
    # CTranslate2 không có backend Metal — STT buộc phải chạy CPU trên macOS
    assert (macos.stt_device, macos.stt_compute_type) == ("cpu", "int8")
    assert cuda.has_dedicated_vram and not macos.has_dedicated_vram


def test_platform_khong_hop_le_bao_loi_ro():
    with pytest.raises(ValueError, match="platform không hợp lệ"):
        _ = load_config(env={"VOSU_PLATFORM": "rocm"}).device


def test_config_example_yaml_nap_duoc():
    """File mẫu trong repo phải luôn nạp được — nếu không, hướng dẫn là sai."""
    from app.core.config import REPO_ROOT

    example = REPO_ROOT / "config.example.yaml"
    assert example.exists()
    cfg = load_config(env={"VOSU_CONFIG": str(example)})
    assert cfg.vram.hard_ceiling_gb == 5.5
    assert cfg.chunker.min_partial_window_s == 1.5
