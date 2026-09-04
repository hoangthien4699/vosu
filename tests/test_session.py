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


def test_utterance_moi_khong_huy_utterance_dang_chay():
    """Câu mới KHÔNG được hủy câu đang xử lý dở.

    Trước đây `begin_utterance` đánh dấu câu trước là CANCELLED. Vì các câu
    được xếp hàng xử lý tuần tự, lúc câu thứ hai vừa dứt lời thì câu thứ nhất
    mới đang dịch dở — nên nó bị hủy oan và mọi thứ hạ nguồn bỏ qua.

    Đo thật: file 6 câu ra đủ 6 bản dịch trên màn hình nhưng chỉ HAI câu cuối
    được đọc thành tiếng. Với tai nghe phiên dịch, không nghe được nghĩa là
    mất hẳn câu đó.

    Cắt lời khi người dùng nói đè lên là việc của Barge-in, không phải của hàm
    mở utterance.
    """
    s = SessionState("sess_x")
    a = s.begin_utterance()
    s.transition(a.id, UtteranceState.TRANSCRIBING)
    s.transition(a.id, UtteranceState.COPILOT)
    s.transition(a.id, UtteranceState.SPEAKING)

    b = s.begin_utterance()

    assert a.state is UtteranceState.SPEAKING, "câu đang đọc dở bị hủy oan"
    assert s.current is b
    assert s.active() == [a, b]


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
