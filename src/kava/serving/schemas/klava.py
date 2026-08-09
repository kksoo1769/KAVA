from typing import Annotated, Literal
from pydantic import BaseModel, Field, model_validator


coord = Annotated[float, Field(ge=0., le=1.)] # [0, 1] 구간 실수로 정의


class KLaVAHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class KLaVARequest(BaseModel):
    """POST /generate 요청 본문"""
    img_path: str
    prompt: str
    history: list[KLaVAHistoryMessage] | None = None
    temperature: float = 0.1
    max_new_tokens: int = 4096
    enable_thinking: bool = False
    # True이면 응답을 SSE로 전달한다
    stream: bool = False

class KLaVAResponse(BaseModel):
    """POST /generate 응답 본문"""
    response: str

class KLaVARereadRequest(KLaVARequest):
    """POST /reread 요청 본문"""
    bbox: tuple[coord, coord, coord, coord] # 4개의 좌표로 한정

    @model_validator(mode="after") # 개별 필드 검증 후, 모델 전체에서 값 검증
    def check_bbox_order(self):
        """(x1, y1), (x2, y2)가 각각 좌측 상단과 우측 하단의 좌표인지 확인"""
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox는 (x1, y1, x2, y2) 순서로, 좌측 상단의 (x1, y1)과 우측 하단의 (x2, y2)가 되어야 합니다.")
        return self
