"""Bộ nhớ hội thoại nhiều lượt (§10 — "rolling summary + short context").

Đặc tả §10 xếp context hẹp là "không phải blocker ở MVP", nhưng nói rõ phải
giải quyết "khi mở rộng sang hội thoại nhiều lượt". Không có nó, mỗi câu được
dịch biệt lập: đại từ không phân giải được ("it", "that one"), câu tiếp nối
mất mạch, và gợi ý trả lời lặp lại thứ vừa nói.

Hai ràng buộc định hình thiết kế:

1. `n_ctx = 2048`. System prompt ~200 token, output ~110 token, nên ngân sách
   cho lịch sử chỉ khoảng 1.5k token. Phải cắt theo cửa sổ trượt, không thể
   giữ hết.

2. Prefix cache. Lịch sử được nối vào SAU system prompt và TRƯỚC câu hiện tại,
   và nó chỉ mọc thêm ở cuối — nên prompt của lượt N+1 vẫn dùng lại được phần
   lớn tiền tố đã cache của lượt N. Đặt lịch sử ở chỗ khác sẽ phá cache và
   đẩy TTFT lên (đo được: 15 token/86ms so với 241 token/610ms).

Ghi cả hai phía. Người dùng nói thật (bằng tiếng của họ) nên lượt của họ cũng
đi qua STT như lượt đối phương — lịch sử phản ánh đúng hội thoại đã diễn ra,
không phải suy đoán từ việc họ bấm nút nào.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Turn:
    """Một lượt của MỘT trong hai người.

    Trước đây chỉ ghi lượt đối phương, còn câu người dùng đáp lại phải suy từ
    việc họ bấm chọn gợi ý nào. Giờ người dùng nói thật nên cả hai phía đều đi
    qua STT — lịch sử phản ánh đúng hội thoại, không phải suy đoán.
    """

    utterance_id: str
    text: str
    language: str | None = None
    #: True nếu đây là lượt của NGƯỜI DÙNG.
    is_user: bool = False
    translation: str = ""

    def render(self) -> str:
        # Ghi bản dịch, không phải nguyên văn: model đọc lịch sử bằng ngôn ngữ
        # nó đang làm việc thì mạch lạc hơn là trộn hai thứ tiếng.
        who = "You" if self.is_user else "Them"
        content = self.translation.strip() or self.text
        return f'{who}: "{content}"' 


@dataclass
class ConversationHistory:
    """Cửa sổ trượt các lượt gần nhất, có trần ký tự."""

    max_turns: int = 6
    max_chars: int = 1200
    _turns: deque[Turn] = field(default_factory=deque)

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def add(
        self,
        utterance_id: str,
        text: str,
        language: str | None,
        *,
        is_user: bool = False,
    ) -> Turn:
        turn = Turn(utterance_id, text.strip(), language, is_user=is_user)
        self._turns.append(turn)
        while len(self._turns) > self.max_turns:
            self._turns.popleft()
        return turn

    def get(self, utterance_id: str) -> Turn | None:
        for turn in self._turns:
            if turn.utterance_id == utterance_id:
                return turn
        return None

    def set_translation(self, utterance_id: str, translation: str) -> None:
        turn = self.get(utterance_id)
        if turn is not None:
            turn.translation = translation.strip()

    def clear(self) -> None:
        self._turns.clear()

    def render(self, *, exclude: str | None = None) -> str:
        """Dựng khối lịch sử cho prompt. Rỗng nếu chưa có lượt nào trước đó.

        `exclude` bỏ đúng lượt đang xử lý — câu hiện tại đã nằm ở phần sau của
        prompt rồi, đưa vào đây nữa là model thấy nó hai lần.

        Cắt từ CŨ nhất khi vượt trần ký tự: lượt gần nhất mới là thứ giúp phân
        giải đại từ và giữ mạch, lượt xa thì để mất được.
        """
        blocks = [t.render() for t in self._turns if t.utterance_id != exclude]
        if not blocks:
            return ""

        while blocks and sum(len(b) + 1 for b in blocks) > self.max_chars:
            blocks.pop(0)
        if not blocks:
            return ""
        return "\n".join(blocks)
