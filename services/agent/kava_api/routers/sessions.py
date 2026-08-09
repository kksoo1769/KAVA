"""세션 턴 처리, 조회, 삭제와 이미지 업로드 엔드포인트."""
import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, File, Path, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from kava_api.api_dependencies import ServiceDep
from kava_api.errors import INTERNAL_DETAIL, KAVAError, public_detail
from kava_api.graph_backend import Delta, Progress
from kava_api.schemas import (
    AssetResponse,
    DeleteResponse,
    MessageOut,
    SessionResponse,
    TurnRequest,
    TurnResponse,
    to_active_image,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["sessions"])

type SessionId = Annotated[str, Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]


@router.post("/sessions/{session_id}/turns", response_model=TurnResponse)
async def create_turn(session_id: SessionId, body: TurnRequest, service: ServiceDep) -> TurnResponse:
    """서비스(TurnService)를 호출해 한 턴을 처리하고 한 턴의 결과(TurnView)를 받아 답변과 현재 이미지 상태를 반환."""
    view = await service.run_turn(
        session_id,
        body.question,
        body.image.image_id if body.image is not None else None,
        clear_image=body.clear_image,
        request_id=body.request_id,
    )
    return TurnResponse(
        session_id=session_id,
        answer=view.answer,
        turn_count=view.turn_count,
        history_reset=view.history_reset,
        active_image=to_active_image(view.img_id, view.img_kind),
    )


def sse(event: str, data: dict) -> str:
    """SSE 한 프레임을 만든다. event 줄, data 줄, 그리고 빈 줄로 끝난다."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/sessions/{session_id}/turns/stream")
async def stream_turn(session_id: SessionId, body: TurnRequest, service: ServiceDep) -> StreamingResponse:
    """한 턴을 SSE로 전달. 완료 이벤트에는 전체 답변을 담고 이후 오류는 이벤트로 처리."""
    async def gen() -> AsyncIterator[str]:
        # 세션 잠금이 스트림 소비 기간과 일치하도록 여기서 제너레이터를 소비한다
        try:
            async for item in service.stream_turn(
                session_id,
                body.question,
                body.image.image_id if body.image is not None else None,
                clear_image=body.clear_image,
                request_id=body.request_id,
            ):
                # 서비스가 반환한 순서대로 이벤트를 전달한다
                if isinstance(item, Progress):
                    yield sse("progress", {"label": item.label})
                    continue
                if isinstance(item, Delta):
                    # 델타에는 화면용 텍스트만 담는다
                    yield sse("delta", {"text": item.text})
                    continue
                yield sse("done", TurnResponse(
                    session_id=session_id,
                    answer=item.answer,
                    turn_count=item.turn_count,
                    history_reset=item.history_reset,
                    active_image=to_active_image(item.img_id, item.img_kind),
                ).model_dump())
        except KAVAError as exc:
            logger.warning("스트리밍 턴 실패 %s: %s(%s)", session_id, exc.code, exc.detail)
            yield sse("error", {"code": exc.code, "detail": public_detail(exc)})
        except Exception:
            # 내부 예외는 로그에만 남긴다
            logger.exception("스트리밍 턴에서 처리되지 않은 오류 %s", session_id)
            yield sse("error", {"code": "internal_error", "detail": INTERNAL_DETAIL})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        # 프록시가 응답을 모으지 않도록 버퍼링을 끈다
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def read_session(session_id: SessionId, service: ServiceDep) -> SessionResponse:
    """확정된 대화를 반환. 없는 세션은 SessionNotFound(404)가 된다."""
    view = await service.get_conversation(session_id)
    return SessionResponse(
        session_id=view.session_id,
        turn_count=view.turn_count,
        messages=[MessageOut(role=message.role, content=message.content) for message in view.messages],
        active_image=to_active_image(view.img_id, view.img_kind),
        has_ocr_cache=view.has_ocr_cache,
    )


@router.delete("/sessions/{session_id}", response_model=DeleteResponse)
async def delete_session(session_id: SessionId, service: ServiceDep) -> DeleteResponse:
    """세션을 멱등하게 삭제하고 처리 중이면 409를 반환."""
    deleted = await service.delete_conversation(session_id)
    return DeleteResponse(session_id=session_id, deleted=deleted)


@router.post("/assets", response_model=AssetResponse, status_code=201)
async def create_asset(service: ServiceDep, file: Annotated[UploadFile, File()]) -> AssetResponse:
    """file 필드의 이미지를 저장하고 image_id를 반환.

    이미지 검증과 파일 저장은 별도 스레드에서 처리한다.
    """
    contents = await file.read()
    ref = await asyncio.to_thread(service.assets.save, file.filename or "upload.png", contents)
    logger.info("이미지 업로드 완료: %s (%d바이트)", ref.id, len(contents))
    return AssetResponse(image_id=ref.id)


@router.get("/assets/{image_id}", response_class=FileResponse)
async def read_asset(image_id: str, service: ServiceDep) -> FileResponse:
    """검증된 image_id의 이미지를 웹 미리보기용으로 반환."""
    ref = await asyncio.to_thread(service.assets.resolve, image_id)
    return FileResponse(
        ref.path,
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )
