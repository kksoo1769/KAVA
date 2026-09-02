"""체크포인트 meta.json 로딩.

기본값으로 추측하지 않는다. 학습이 기록한 작동점(시각 토큰 시퀀스 길이, 비전 타워 ID 등)과
다른 값으로 조용히 로드되면 성능이 재현되지 않는데 예외도 나지 않아 원인 추적이 어렵다.
실제로 이 저장소에는 siglip_num_patches 의 기본값이 576, 768, 784 로 갈려 있어 어느 경로로
로드하느냐에 따라 다른 모델이 되는 상태였다.
"""

from __future__ import annotations

import json
import os

# train/vlm.py 의 save_ckpt 가 항상 기록하는 키. 하나라도 없으면 로드를 거부한다.
REQUIRED_META_KEYS = (
    "exaone_id",
    "vision_backbone",
    "siglip_model_id",
    "siglip_num_patches",
)


def load_meta(ckpt_dir: str | os.PathLike) -> dict:
    """<ckpt_dir>/meta.json 을 읽고 필수 키를 검증해 반환한다."""
    path = os.path.join(str(ckpt_dir), "meta.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"meta.json 이 없습니다: {path}")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)

    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise KeyError(
            f"meta.json 에 필수 키가 없습니다: {missing} ({path}). "
            "기본값으로 추측하면 학습과 다른 작동점으로 조용히 로드되므로 거부합니다."
        )

    backbone = meta["vision_backbone"]
    if backbone != "siglip2":
        raise ValueError(
            f"이 체크포인트의 vision_backbone 은 {backbone!r} 입니다. AxisMF 백본은 CUDA 전용 "
            "mamba-ssm 커널에 의존해 제거했으므로 siglip2 체크포인트만 로드할 수 있습니다."
        )
    return meta
