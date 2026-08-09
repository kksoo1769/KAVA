import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# 구체 런타임을 불러오지 않도록 Protocol을 사용한다
from kava.serving.protocol import SharedRuntimeProtocol, StreamResult
from kava.serving.images import save_bbox_crop
from kava.serving.klava_dependencies import RuntimeDep
from kava.serving.schemas import (
    ChatRequest, ChatResponse,
    KLaVARequest, KLaVAResponse, KLaVARereadRequest,
)


logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])


# 엔드포인트에서 사용하는 요청 본문과 런타임
@router.post("/generate", response_model=KLaVAResponse)
def generate(body: KLaVARequest, runtime: RuntimeDep):
    """이미지 전체를 보고 응답을 생성. body.stream=True면 SSE로 토큰을 흘려보낸다."""
    payload = to_vlm_payload(body)
    if body.stream:
        return sse_response(runtime, kind="vlm", payload=payload)
    return {"response": run_job(runtime, kind="vlm", payload=payload)}

@router.post("/reread", response_model=KLaVAResponse)
def reread(body: KLaVARereadRequest, runtime: RuntimeDep):
    """bbox 영역을 crop한 뒤, 그 부분만 보고 응답을 생성."""
    try:
        crop_path = save_bbox_crop(body.img_path, body.bbox)
    except FileNotFoundError as exc:
        # crop은 스트림이 시작되기 전이므로 요청 오류를 422로 반환한다
        raise HTTPException(status_code=422, detail=f"이미지를 찾을 수 없습니다: {body.img_path}") from exc
    payload = to_vlm_payload(body, crop_path)
    if body.stream:
        return sse_response(runtime, kind="vlm", payload=payload)
    return {"response": run_job(runtime, kind="vlm", payload=payload)}

@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, runtime: RuntimeDep):
    """텍스트만 주어졌을 때, LM base만으로 응답을 생성."""
    payload = to_chat_payload(body)
    if body.stream:
        return sse_response(runtime, kind="chat", payload=payload)
    return run_job(runtime, kind="chat", payload=payload)

def to_vlm_payload(body: KLaVARequest, img_path: str | None = None) -> dict:
    """KLaVARequest를 SharedRuntime.run_vlm 키워드 인자 dict로 변환.
    
    reread crop의 경우, img_path를 주면 body.img_path 대신 사용한다."""
    return {
        "img_path": img_path or body.img_path,
        "prompt": body.prompt,
        "history": [msg.model_dump() for msg in (body.history or [])],
        "temperature": body.temperature,
        "max_new_tokens": body.max_new_tokens,
        "enable_thinking": body.enable_thinking,
    }

def to_chat_payload(body: ChatRequest) -> dict:
    """ChatRequest를 SharedRuntime.run_chat 키워드 인자 dict로 변환."""
    return {
        "messages": [msg.model_dump(exclude_none=True) for msg in (body.messages)],
        "tools": body.tools,
        "temperature": body.temperature,
        "max_new_tokens": body.max_new_tokens,
        "enable_thinking": body.enable_thinking,
    }

def sse(event: str, data: dict) -> str:
    """event와 data 줄로 구성된 SSE 프레임을 생성."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def done_body(value: str | dict) -> dict:
    """일반 응답과 같은 스키마로 완료 이벤트 본문을 생성."""
    if isinstance(value, str):
        return KLaVAResponse(response=value).model_dump() # vlm은 문자열 하나
    return ChatResponse(**value).model_dump() # chat은 3개 키가 든 dict

def stream_events(runtime: SharedRuntimeProtocol, kind: str, payload: dict):
    """증분을 전달하고 마지막에 전체 응답을 보낸다. 시작 후 오류는 오류 이벤트로 처리."""
    try:
        for item in runtime.stream(kind=kind, payload=payload):
            if isinstance(item, StreamResult):
                yield sse("done", done_body(item.value))
            else:
                yield sse("delta", {"text": item})
    except TimeoutError:
        # 일반 응답의 504와 같은 경우다
        yield sse("error", {"detail": "모델 추론 시간이 너무 길어 timeout 되었습니다."})
    except FileNotFoundError:
        # 스트리밍 오류에는 이미지 경로를 포함하지 않는다
        logger.exception("스트리밍 중 이미지를 찾지 못했다")
        yield sse("error", {"detail": "이미지를 찾을 수 없습니다."})
    except ValueError as exc:
        # 검증 오류는 일반 응답과 같은 문구를 사용한다
        yield sse("error", {"detail": str(exc)})
    except Exception:
        # 처리하지 않은 예외는 내부 정보를 제외한다
        logger.exception("스트리밍 중 처리되지 않은 오류")
        yield sse("error", {"detail": "모델 추론 중 오류가 발생했습니다."})

def sse_response(runtime: SharedRuntimeProtocol, kind: str, payload: dict) -> StreamingResponse:
    """버퍼링을 끈 SSE 응답을 생성."""
    return StreamingResponse(
        stream_events(runtime, kind, payload),
        media_type="text/event-stream",
        headers={
            # 프록시가 응답을 모으지 않도록 버퍼링을 끈다
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

def run_job(runtime: SharedRuntimeProtocol, kind: str, payload: dict) -> str | dict:
    """runtime queue에 job을 넣고, 내부의 예외를 HTTP 오류로 반환"""
    try:
        return runtime.run(kind=kind, payload=payload)
    except TimeoutError as exc:
        # 504: 서버가 작업을 수행하다가 시간 초과
        raise HTTPException(status_code=504, detail="모델 추론 시간이 너무 길어 timeout 되었습니다.") from exc # HTTPException을 TimeOutError의 원인으로 연결
    except FileNotFoundError as exc:
        # 422: 요청 형식은 맞지만 유효하지 않은 데이터
        raise HTTPException(status_code=422, detail=f"이미지를 찾을 수 없습니다: {exc}") from exc # HTTPException을 FileNotFoundError의 원인으로 연결
    except ValueError as exc:
        # 422: 요청 형식은 맞지만 유효하지 않은 데이터
        raise HTTPException(status_code=422, detail=str(exc)) from exc
