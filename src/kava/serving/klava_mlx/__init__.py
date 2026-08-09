"""KLaVA MLX 백엔드의 경로와 기본 설정."""

from __future__ import annotations

import os
from pathlib import Path

from kava.paths import REPO_ROOT

__all__ = [
    "preprocess", "vision", "model", "convert",
    "MLX_SUBDIR", "default_ckpt_dir", "mlx_root",
]

# 체크포인트를 바꾸면 해당 MLX 변환 결과도 함께 선택된다
MLX_SUBDIR = "mlx"

# 서버 설정과 같은 기본 체크포인트 경로
_DEFAULT_CKPT_PARTS = ("runs", "klava_instruct_784_r64", "ckpts", "fin")


def default_ckpt_dir() -> str:
    """KLAVA_CKPT 가 있으면 그것, 없으면 저장소 기본 체크포인트 경로."""
    return os.environ.get("KLAVA_CKPT") or str(REPO_ROOT.joinpath(*_DEFAULT_CKPT_PARTS))


def mlx_root(ckpt_dir: str | os.PathLike | None = None) -> Path:
    """MLX 변환 산출물이 들어가는 디렉터리(<ckpt_dir>/mlx)."""
    ckpt = str(ckpt_dir) if ckpt_dir is not None else default_ckpt_dir()
    return Path(ckpt).expanduser() / MLX_SUBDIR
