"""KLaVA 추론 서버"""
from collections.abc import Callable
from contextlib import asynccontextmanager
from fastapi import FastAPI

from kava.serving.klava_dependencies import KLaVASettings, build_shared_runtime
from kava.serving.routers.health import router as health_router
from kava.serving.routers.inference import router as inference_router


def create_app(
    settings: KLaVASettings | None = None,
    runtime_factory: Callable[[KLaVASettings], object] | None = None,
) -> FastAPI:
    resolved = settings or KLaVASettings.from_env()
    factory = runtime_factory or build_shared_runtime

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = factory(resolved)
        app.state.runtime = runtime
        app.state.ready = True

        try:
            yield
        finally:
            app.state.ready = False
            runtime.shutdown()

    app = FastAPI(title="KLaVA", lifespan=lifespan)

    app.state.settings = resolved
    app.state.ready = False # lifespan 이전 ready 초기화
    
    app.include_router(health_router)
    app.include_router(inference_router)
    return app


app = create_app()