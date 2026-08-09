"""한 대화에서 들고 있어야 하는 값들."""
from dataclasses import dataclass, field
from typing import Literal

from kava_api.graph_state import ImgKind

type Role = Literal["user", "assistant"] # 사용자와 AI의 답변만 사용


@dataclass(frozen=True, slots=True)
class Message:
    """대화의 한 메시지."""
    role: Role
    content: str

@dataclass(frozen=True, slots=True)
class ImageRef:
    """이미지 id와 그 경로."""
    id: str
    path: str

@dataclass(slots=True) # frozen=False: 속성값이 바뀜
class OcrCache:
    """OCR로 추출한 텍스트."""
    text: str

@dataclass(slots=True)
class ConversationState:
    """한 세션에서 정상적으로 완료된 대화 상태."""
    session_id: str
    messages: list[Message] = field(default_factory=list)
    img: ImageRef | None = None
    img_kind: ImgKind = ""
    ocr_cache: OcrCache | None = None

    @property
    def turn_count(self) -> int:
        """지금 history에 남아 있는 완결된 턴의 수. assistant의 발화 개수와 같음."""
        return sum(message.role == "assistant" for message in self.messages)

    def is_same_image(self, img: ImageRef | None) -> bool:
        """현재 대화에 연결된 이미지와 같은지 확인."""
        return img is not None and self.img is not None and img.id == self.img.id

    def reset_context(self, *, img: ImageRef | None) -> None:
        """단일 이미지를 지원하므로, 이미지가 바뀌면 대화와 이미지 관련 값을 모두 초기화한다."""
        self.messages.clear() # 리스트 초기화
        self.img = img # 새 이미지로 초기화
        self.img_kind = ""
        self.ocr_cache = None

    def derived_for(self, img: ImageRef | None) -> tuple[ImgKind, OcrCache | None]:
        """같은 이미지에 저장된 분류 결과와 OCR 내용을 반환."""
        if not self.is_same_image(img):
            return "", None
        return self.img_kind, self.ocr_cache

    def append_turn(self, question: str, answer: str, *, max_turns: int) -> None:
        """<user, assistant> 완결된 한 턴을 기록하고, 오래된 턴은 쌍 단위로 버림."""
        self.messages.extend([
            Message(role="user", content=question),
            Message(role="assistant", content=answer),
        ])
        overflow = self.turn_count - max_turns
        if overflow > 0:
            del self.messages[:overflow * 2]
