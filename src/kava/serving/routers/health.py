from fastapi import APIRouter, HTTPException, Request

# torch를 불러오지 않도록 경량 Protocol을 사용한다
from kava.serving.protocol import SharedRuntimeProtocol


router = APIRouter(tags=["health"])

@router.get("/livez")
def livez():
    """HTTP 프로세스가 살아 있는지 확인."""
    return {"status": "alive"}

@router.get("/readyz")
def readyz(request: Request):
    """모델 로드, 워커 스레드, queue 작동 확인."""
    state = request.app.state
    runtime: SharedRuntimeProtocol | None = getattr(state, "runtime", None)
    if not getattr(state, "ready", False) or runtime is None:
        # 503: 서비스 사용 불가능(사바 과부하 등의 이유)
        raise HTTPException(status_code=503, detail="아직 런타임이 준비되지 않았습니다.")
    if not runtime.worker.is_alive():
        raise HTTPException(status_code=503, detail="모델의 워커 스레드가 살아 있지 않습니다.")
    return {"status": "ready", "queue_size": runtime.queue.qsize()}
