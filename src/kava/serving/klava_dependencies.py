"""Endpoint에서 필요한 객체 정리"""
import os
from dataclasses import dataclass

from typing import Annotated
from fastapi import Depends, HTTPException, Request

from kava.serving.protocol import SharedRuntimeProtocol
from kava.paths import repo_path


# KLAVA_DEVICE는 torch 디바이스 선택에 사용하므로 백엔드 설정과 분리한다
BACKENDS = ("mlx", "torch")
DEFAULT_BACKEND = "mlx"


@dataclass(frozen=True) # 이미 생성된 값 변경 금지
class KLaVASettings:
    """환경 변수로 조절 가능한 서버 설정."""
    ckpt_dir: str
    device: str | None = None
    backend: str = DEFAULT_BACKEND

    @classmethod
    def from_env(cls) -> "KLaVASettings":
        backend = (os.environ.get("KLAVA_BACKEND") or DEFAULT_BACKEND).strip().lower()
        if backend not in BACKENDS:
            raise ValueError(
                f"KLAVA_BACKEND={backend!r} 는 모르는 값이다. "
                f"허용: {list(BACKENDS)} (기본 {DEFAULT_BACKEND})."
            )
        return cls(
            ckpt_dir=os.environ.get("KLAVA_CKPT", repo_path("runs", "klava_instruct_784_r64", "ckpts", "fin")),
            device=os.environ.get("KLAVA_DEVICE") or None,
            backend=backend,
        )

def build_shared_runtime(settings: KLaVASettings) -> SharedRuntimeProtocol:
    """선택한 백엔드 모듈을 지연 import해 런타임을 생성."""
    backend = settings.backend

    if backend == "mlx":
        # 준비되지 않은 MLX 설정은 torch로 자동 전환하지 않는다
        from kava.serving.klava_mlx.runtime import MLXSharedRuntime

        if settings.device:
            print(
                f"[KLaVA] KLAVA_DEVICE={settings.device!r} 는 mlx 백엔드에서 무시된다"
                " (MLX 는 통합 메모리 위의 단일 Metal 디바이스라 고를 것이 없다)."
                " torch 백엔드를 쓰려면 KLAVA_BACKEND=torch 로 지정하라.",
                flush=True,
            )
        runtime = MLXSharedRuntime(settings.ckpt_dir, settings.device)
        print(f"[KLaVA] backend=mlx {runtime.describe()}", flush=True)
        return runtime

    if backend == "torch":
        from kava.serving.runtime import SharedRuntime

        runtime = SharedRuntime(settings.ckpt_dir, settings.device)
        print(
            f"[KLaVA] backend=torch device={runtime.device} ckpt_dir={settings.ckpt_dir}",
            flush=True,
        )
        return runtime

    # from_env 를 거치지 않고 KLaVASettings 를 직접 만든 경우까지 막는다.
    raise ValueError(
        f"KLAVA_BACKEND={backend!r} 는 모르는 값이다. 허용: {list(BACKENDS)}."
    )

def get_runtime(request: Request) -> SharedRuntimeProtocol:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="KLaVA runtime이 아직 준비되지 않았습니다.")
    return runtime


RuntimeDep = Annotated[SharedRuntimeProtocol, Depends(get_runtime)]
