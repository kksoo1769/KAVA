import os

from datetime import datetime
from PIL import Image

from kava.paths import repo_path


CROP_DIR = os.environ.get("KLAVA_CROP_DIR", repo_path("analysis", "reread_crop_images"))


def save_bbox_crop(img_path: str, bbox: tuple[float, float, float, float]) -> str:
    """정규화된 bbox 영역을 잘라 CROP_DIR에 저장하고, 그 경로를 반환"""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = bbox # [0, 1]로 정규화된 좌표

    crop = img.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))
    stem, ext = os.path.splitext(os.path.basename(img_path))
    os.makedirs(CROP_DIR, exist_ok=True)
    crop_path = os.path.join(CROP_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{stem}_{x1:.2f}_{y1:.2f}_{x2:.2f}_{y2:.2f}{ext}")
    crop.save(crop_path, quality=95)
    return crop_path