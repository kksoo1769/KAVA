"""KAVA 에이전트 HTTP 서버."""
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kava_api.assets import LocalAssetStore
from kava_api.dependencies import AgentSettings, build_default_dependencies
from kava_api.errors import INTERNAL_DETAIL, KAVAError, public_detail
from kava_api.graph_backend import GraphBackend
from kava_api.routers.health import router as health_router
from kava_api.routers.sessions import router as sessions_router
from kava_api.service import TurnService
from kava_api.store import ConversationStore

logger = logging.getLogger(__name__)


def build_default_service(settings: AgentSettings) -> TurnService:
    """기본 의존성으로 TurnService를 생성."""
    deps = build_default_dependencies(settings)
    return TurnService(
        store=ConversationStore(),
        backend=GraphBackend(deps, recursion_limit=settings.recursion_limit),
        assets=LocalAssetStore(Path(settings.asset_dir)),
        max_turns=settings.max_turns,
    )


def create_app(
    settings: AgentSettings | None = None,
    service_factory: Callable[[AgentSettings], TurnService] | None = None,
) -> FastAPI:
    """설정에 맞춰 FastAPI 앱을 생성."""
    resolved = settings or AgentSettings.from_env()
    factory = service_factory or build_default_service

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.service = factory(resolved)
            app.state.ready = True
            logger.info("에이전트 서비스를 조립했습니다. 자산 디렉터리: %s", resolved.asset_dir)
        except Exception:
            # 조립 실패는 준비 상태로 알리고 프로세스는 유지한다
            logger.exception("에이전트 서비스 조립에 실패했습니다.")

        try:
            yield
        finally:
            app.state.ready = False
            app.state.service = None

    app = FastAPI(title="KAVA Agent API", lifespan=lifespan)
    # lifespan 실행 전의 준비 상태
    app.state.service = None
    app.state.ready = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["content-type"],
        # 쿠키를 사용하지 않으므로 credential은 허용하지 않는다
    )

    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(sessions_router)
    return app


def register_exception_handlers(app: FastAPI) -> None:
    """모든 KAVA 오류를 같은 응답 형식으로 등록."""

    @app.exception_handler(KAVAError)
    async def handle_kava_error(request: Request, exc: KAVAError) -> JSONResponse:
        """KAVA 오류를 코드와 공개 문구로 반환."""
        # 서버 오류의 내부 정보는 public_detail에서 제거한다
        detail = public_detail(exc)
        logger.warning(
            "요청 실패 %s %s: %s(%s)", request.method, request.url.path, exc.code, exc.detail
        )
        return JSONResponse(status_code=exc.status_code, content={"code": exc.code, "detail": detail})

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """요청 검증 오류를 공통 형식으로 반환."""
        return JSONResponse(
            status_code=422,
            content={"code": "invalid_request", "detail": summarize_validation_error(exc)},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """처리하지 않은 예외를 내부 오류로 반환."""
        logger.exception("처리되지 않은 오류 %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "detail": INTERNAL_DETAIL},
        )


def summarize_validation_error(exc: RequestValidationError) -> str:
    """검증 오류 리스트를 사람이 읽을 문자열 한 줄로 요약."""
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()) if item != "body")
        message = error.get("msg", "잘못된 값입니다.")
        parts.append(f"{location}: {message}" if location else message)
    if not parts:
        return "요청 형식이 올바르지 않습니다."
    return "요청 형식이 올바르지 않습니다. " + "; ".join(parts)


app = create_app()
