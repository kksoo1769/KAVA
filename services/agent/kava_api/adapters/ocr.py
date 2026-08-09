"""Apple Vision OCR subprocess 클라이언트"""
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4] # kava
DEFAULT_BINARY_PATH = REPO_ROOT / "services" / "vision" / "vision-ocr"
DEFAULT_TIMEOUT = 60


class VisionOCRClient:
    """Vision OCR 호출 경로"""
    def __init__(
        self,
        binary_path: str = str(DEFAULT_BINARY_PATH),
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.binary_path = binary_path
        self.timeout = timeout

    def run(self, mode: str, img_path: str) -> dict:
        """OCR을 실행하고 실패하면 빈 dict를 반환해 다른 모드들의 결과로만 진행."""
        try:
            completed = subprocess.run(
                [self.binary_path, "--mode", mode, img_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return json.loads(completed.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
            print(f"OCR 실행 실패. {mode} mode: {exc}")
            return {}