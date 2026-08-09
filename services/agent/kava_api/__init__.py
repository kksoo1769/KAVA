"""환경 변수를 불러오는 kava_api 패키지."""
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3] # kava
load_dotenv(REPO_ROOT / ".env", override=False)
