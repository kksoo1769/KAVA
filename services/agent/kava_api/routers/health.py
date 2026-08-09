"""프로세스와 서비스 준비 상태 확인. /v1 접두어 없이 루트에 둔다."""
from fastapi import APIRouter

from kava_api.api_dependencies import ServiceDep

router = APIRouter(tags=["health"])


@router.get("/livez")
def livez() -> dict[str, str]:
    """HTTP 프로세스가 살아 있는지 확인."""
    return {"status": "alive"}


@router.get("/readyz")
def readyz(service: ServiceDep) -> dict[str, str]:
    """서비스가 요청을 처리할 준비가 되었는지 확인."""
    return {"status": "ready"}
