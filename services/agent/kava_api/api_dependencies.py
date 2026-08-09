"""엔드포인트가 FastAPI Depends로 받는 객체.

dependencies.py는 그래프가 쓰는 외부 의존성을, 이 파일은 요청 하나가 쓰는 의존성을 다룬다."""
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from kava_api.service import TurnService

NOT_READY_DETAIL = "KAVA 에이전트가 아직 준비되지 않았습니다."


def get_service(request: Request) -> TurnService:
    """lifespan에서 조립된 서비스를 요청마다 가져온다.

    초기화가 실패했거나 종료 중이면 서비스가 없으므로 ready도 함께 확인한다."""
    state = request.app.state
    if not getattr(state, "ready", False) or getattr(state, "service", None) is None:
        raise HTTPException(status_code=503, detail=NOT_READY_DETAIL)
    return state.service


ServiceDep = Annotated[TurnService, Depends(get_service)] # TurnService를 서버 시작시 한 번만 생성하고 get_service로 꺼내오기만 한다.
