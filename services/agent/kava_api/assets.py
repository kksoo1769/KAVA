"""업로드된 이미지의 id와 실제 경로 연결."""
import io
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from PIL import Image

from kava_api.domain import ImageRef
from kava_api.errors import AssetNotFound, AssetTooLarge, InvalidAsset

# 저장을 허용하는 확장자. 여기 없는 확장자는 PIL이 열 수 있어도 받지 않는다.
ALLOWED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})

# 경로로 해석되면 base_dir 밖을 가리킬 수 있는 문자들.
FORBIDDEN_IN_ID = ("/", "\\", "\0", "..")


class AssetStore(Protocol):
    """이미지 id를 실제 경로로 변환. 같은 id는 언제나 같은 경로로 돌아와야 한다."""

    def save(self, filename: str, data: bytes) -> ImageRef:
        """업로드 바이트를 저장하고 새 id가 붙은 ImageRef를 반환."""
        ...

    def resolve(self, img_id: str) -> ImageRef:
        """id에 대응하는 ImageRef를 반환하고, 없으면 AssetNotFound를 낸다."""
        ...


class LocalAssetStore:
    """업로드 이미지를 로컬 디렉터리에 저장."""
    def __init__(
        self,
        base_dir: Path | str,
        *,
        max_bytes: int = 20 * 1024 * 1024,
        max_pixels: int = 40_000_000, # 8000x5000 사진까지는 통과하는 크기
    ) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels

    def save(self, filename: str, data: bytes) -> ImageRef:
        """크기와 확장자, 실제 내용을 검사한 뒤 저장하고 새 id를 붙여 반환."""
        if len(data) > self.max_bytes:
            raise AssetTooLarge(f"이미지가 너무 큽니다. 최대 {self.max_bytes}바이트까지 올릴 수 있습니다.")
        suffix = Path(filename).suffix.lower() # 확장자는 항상 소문자로 정규화한다
        if suffix not in ALLOWED_SUFFIXES:
            raise InvalidAsset(f"지원하지 않는 이미지 확장자입니다. 허용: {' '.join(sorted(ALLOWED_SUFFIXES))}")
        try:
            # 확장자와 별개로 실제 이미지 형식과 크기를 확인한다
            image = Image.open(io.BytesIO(data))
            pixels = image.width * image.height
            image.verify() # 구조가 깨진 파일을 걸러낸다
        except Exception as exc: # PIL은 UnidentifiedImageError 외에도 여러 예외를 낸다
            raise InvalidAsset("이미지 파일로 읽을 수 없습니다.") from exc
        if pixels > self.max_pixels: # 작은 파일로 1억 픽셀을 표현하는 압축 폭탄 방어
            raise InvalidAsset(f"이미지가 너무 큽니다. 가로 x 세로가 {self.max_pixels}픽셀 이하여야 합니다.")

        path = self.base_dir / f"{uuid4().hex}{suffix}" # 파일 이름이 곧 id다
        path.write_bytes(data)
        return ImageRef(id=path.name, path=str(path))

    def resolve(self, img_id: str) -> ImageRef:
        """안전한 id로 저장된 이미지 경로를 조회."""
        if not img_id or any(bad in img_id for bad in FORBIDDEN_IN_ID) or img_id != img_id.lower():
            # 저장소에서 생성하지 않은 형식의 id는 거부한다
            raise AssetNotFound("올바른 이미지 id가 아닙니다.")
        try:
            # 실제 경로가 자산 디렉터리 안에 있는지 확인한다
            path = (self.base_dir / img_id).resolve()
            if path.parent != self.base_dir or not path.is_file():
                raise AssetNotFound("이미지가 존재하지 않습니다.")
        except OSError as exc: # 파일 이름 길이 상한(macOS 255바이트)을 넘으면 여기로 온다
            raise AssetNotFound("올바른 이미지 id가 아닙니다.") from exc
        return ImageRef(id=path.name, path=str(path)) # 정규화된 이름을 담는다
