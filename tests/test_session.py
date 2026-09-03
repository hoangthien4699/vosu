from __future__ import annotations

import pytest

from app.audio.session import (
    InvalidTransition,
    SessionState,
    UtteranceState,
)


def test_utterance_id_duy_nhat_va_tang_dan():
    s = SessionState("sess_x")
    ids = [s.begin_utterance().id for _ in range(3)]
    assert ids == ["utt_001", "utt_002", "utt_003"]
    assert len(set(ids)) == 3


def test_luong_trang_thai_binh_thuong():
    s = SessionState("sess_x")
    u = s.begin_utterance()
    for target in (
        UtteranceState.TRANSCRIBING,
        UtteranceState.COPILOT,
        UtteranceState.SPEAKING,
        UtteranceState.DONE,
    ):
        s.transition(u.id, target)
    assert u.is_terminal


def test_transition_sai_thu_tu_bi_tu_choi():
    s = SessionState("sess_x")
    u = s.begin_utterance()
    with pytest.raises(InvalidTransition):
        s.transition(u.id, UtteranceState.SPEAKING)  # nhảy cóc qua TRANSCRIBING


def test_khong_the_hoi_sinh_utterance_da_ket_thuc():
    s = SessionState("sess_x")
    u = s.begin_utterance()
    s.transition(u.id, UtteranceState.CANCELLED)
    with pytest.raises(InvalidTransition):
        s.transition(u.id, UtteranceState.TRANSCRIBING)


def test_utterance_moi_huy_utterance_dang_chay():
    """Kịch bản Barge-in §2.4.1: người đối diện nói tiếp khi TTS đang phát."""
    s = SessionState("sess_x")
    a = s.begin_utterance()
    s.transition(a.id, UtteranceState.TRANSCRIBING)
    s.transition(a.id, UtteranceState.COPILOT)
    s.transition(a.id, UtteranceState.SPEAKING)

    b = s.begin_utterance()

    assert a.state is UtteranceState.CANCELLED
    assert s.current is b
    assert s.active() == [b]


def test_stt_rong_ket_thuc_som_van_hop_le():
    s = SessionState("sess_x")
    u = s.begin_utterance()
    s.transition(u.id, UtteranceState.TRANSCRIBING)
    s.transition(u.id, UtteranceState.DONE)
    assert u.state is UtteranceState.DONE


def test_e2e_do_tu_vad_endpoint_khong_phai_tu_luc_bat_dau_noi():
    import time

    s = SessionState("sess_x")
    u = s.begin_utterance()
    time.sleep(0.02)          # thời gian người ta đang nói — KHÔNG tính vào E2E
    u.mark_endpoint()
    time.sleep(0.01)
    u.mark_first_useful()

    assert u.e2e_ms is not None
    assert 5 < u.e2e_ms < 40, f"E2E={u.e2e_ms}ms — có vẻ đang tính cả thời gian nói"
    u.mark_first_useful()     # gọi lại không được ghi đè
    assert u.e2e_ms < 40
