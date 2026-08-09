from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> str:
    """저장소 루트를 기준으로 경로를 만들어 문자열로 반환한다."""
    return str(REPO_ROOT.joinpath(*parts))
