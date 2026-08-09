"""그래프에 주입하는 외부 의존성 설정."""
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from kava_api.adapters import klava as klava_adapter
from kava_api.adapters import ocr as ocr_adapter

logger = logging.getLogger(__name__)

# 업로드 이미지 기본 보관 위치. runs/는 이미 생성물 디렉터리이고 .gitignore에 올려 두었다.
DEFAULT_ASSET_DIR = ocr_adapter.REPO_ROOT / "runs" / "uploads"

# 그래프 한 턴에서 허용할 최대 단계 수
DEFAULT_RECURSION_LIMIT = 25

# 개발용 프론트엔드(next dev 3000, e2e 3100)만 기본 허용한다.
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:3100",
    "http://localhost:3100",
)


class KLaVAPort(Protocol):
    """그래프가 KLaVA에 원하는 것을 담고, 실패 시 KLaVAError을 낸다."""
    def generate(
        self,
        img_path: str,
        prompt: str,
        history: list[dict] | None = None,
        temperature: float = 0.,
        max_new_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> str: ...

    def stream_generate(
        self,
        img_path: str,
        prompt: str,
        history: list[dict] | None = None,
        temperature: float = 0.,
        max_new_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> Iterator[str | klava_adapter.StreamResult]:
        """증분 텍스트를 전달한 뒤 generate와 같은 최종 결과를 반환."""
        ...

class OCRPort(Protocol):
    """그래프가 OCR에 원하는 것을 담고, 실패 시 빈 dict를 반환한다."""
    def run(self, mode: str, img_path: str) -> dict: ...

class TextPort(Protocol):
    """그래프에서 사용하는 텍스트 모델 인터페이스."""
    def invoke(self, messages: list): ...

    def stream(self, messages: list) -> Iterator: ...


def split_csv(raw: str) -> tuple[str, ...]:
    """쉼표로 구분된 환경 변수 값을 항목 튜플로 변환."""
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class AgentSettings:
    """에이전트 환경 변수 설정."""
    klava_url: str = klava_adapter.DEFAULT_BASE_URL
    klava_timeout: float = klava_adapter.DEFAULT_TIMEOUT
    ocr_binary_path: str = str(ocr_adapter.DEFAULT_BINARY_PATH)
    ocr_timeout: float = ocr_adapter.DEFAULT_TIMEOUT
    temperature: float = .1
    max_turns: int = 10
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    asset_dir: str = str(DEFAULT_ASSET_DIR)
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("최대 턴 수는 1 이상이어야 합니다.")
        if self.recursion_limit < 1:
            raise ValueError("최대 재귀 깊이는 1 이상이어야 합니다.")

    @classmethod
    def from_env(cls) -> "AgentSettings":
        return cls(
            klava_url=os.environ.get("KAVA_KLAVA_URL", klava_adapter.DEFAULT_BASE_URL),
            klava_timeout=float(os.environ.get("KAVA_HTTP_TIMEOUT", str(klava_adapter.DEFAULT_TIMEOUT))),
            ocr_binary_path=os.environ.get("KAVA_OCR_BIN", str(ocr_adapter.DEFAULT_BINARY_PATH)),
            ocr_timeout=float(os.environ.get("KAVA_OCR_TIMEOUT", str(ocr_adapter.DEFAULT_TIMEOUT))),
            temperature=float(os.environ.get("KAVA_TEMPERATURE", "0.1")),
            max_turns=int(os.environ.get("KAVA_MAX_TURNS", "10")),
            recursion_limit=int(os.environ.get("KAVA_RECURSION_LIMIT", str(DEFAULT_RECURSION_LIMIT))),
            asset_dir=os.environ.get("KAVA_ASSET_DIR", str(DEFAULT_ASSET_DIR)),
            cors_origins=split_csv(os.environ.get("KAVA_CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS))),
        )

@dataclass(frozen=True)
class AgentDependencies:
    """그래프가 사용하는 의존성 묶음."""
    klava: KLaVAPort
    ocr: OCRPort
    text_model: TextPort
    tools: list

def build_default_dependencies(settings: AgentSettings | None = None) -> AgentDependencies:
    """설정 값들을 받아 의존성을 반환."""
    from kava_api.adapters.text import ChatEXAONE
    from kava_api.tools import calc, web_search

    resolved = settings or AgentSettings.from_env()
    if not os.environ.get("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY가 없습니다. web_search 도구는 실패 메시지만 반환합니다.")
    tools = [calc, web_search]
    return AgentDependencies(
        klava=klava_adapter.KLaVAClient(base_url=resolved.klava_url, timeout=resolved.klava_timeout),
        ocr=ocr_adapter.VisionOCRClient(binary_path=resolved.ocr_binary_path, timeout=resolved.ocr_timeout),
        text_model=ChatEXAONE(temperature=resolved.temperature).bind_tools(tools),
        tools=tools,
    )
