"""KAVA 에이전트 HTTP 요청, 응답 본문 스키마."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


MAX_QUESTION_LENGTH = 2048 # 질문 길이의 상한.
MAX_IMAGE_ID_LENGTH = 200 # mac OS의 파일 이름 길이 제한(255)에 따른 image_id 길이 상한.


class ImageSelector(BaseModel):
    """이번 턴에서 볼 이미지를 지목한다. POST /v1/assets가 돌려준 image_id를 그대로 쓴다."""
    image_id: str = Field(min_length=1, max_length=MAX_IMAGE_ID_LENGTH)


class TurnRequest(BaseModel):
    """POST /v1/sessions/{session_id}/turns 요청 본문"""
    request_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    image: ImageSelector | None = None
    clear_image: bool = False

    @model_validator(mode="after")
    def check_question_not_blank(self) -> "TurnRequest":
        """공백만 있는 질문을 거른다."""
        if not self.question.strip():
            raise ValueError("질문이 비어 있습니다.")
        return self


class ActiveImage(BaseModel):
    """지금 대화에 사용되는 이미지."""
    image_id: str
    kind: str | None # 아직 분류되지 않았으면 null


class TurnResponse(BaseModel):
    """POST /v1/sessions/{session_id}/turns 응답 본문"""
    session_id: str
    answer: str
    turn_count: int
    history_reset: bool
    active_image: ActiveImage | None


class MessageOut(BaseModel):
    """조회 응답에 실리는 한 발화."""
    role: Literal["user", "assistant"]
    content: str


class SessionResponse(BaseModel):
    """GET /v1/sessions/{session_id} 응답 본문"""
    session_id: str
    turn_count: int
    messages: list[MessageOut]
    active_image: ActiveImage | None
    has_ocr_cache: bool # OCR 원문은 싣지 않고 있고 없음만 알린다


class DeleteResponse(BaseModel):
    """DELETE /v1/sessions/{session_id} 응답 본문"""
    session_id: str
    deleted: bool


class AssetResponse(BaseModel):
    """POST /v1/assets 응답 본문"""
    image_id: str


def to_active_image(img_id: str | None, img_kind: str | None) -> ActiveImage | None:
    """한 턴의 결과(TurnView)의 img_id, img_kind를 받아 ActiveImage 객체로 반환.

    그래프에서 img_kind가 결정되지 않으면 빈 문자열("")이지만, 프론트엔드에서 그대로 렌더링되지 않게 None으로 바꿔야 한다."""
    if img_id is None:
        return None
    return ActiveImage(image_id=img_id, kind=img_kind or None)
