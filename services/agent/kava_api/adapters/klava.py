"""KLaVA 추론 서버 HTTP 클라이언트(8001 port)"""
import json
import requests
from dataclasses import dataclass


DEFAULT_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_TIMEOUT = 120


class KLaVAError(RuntimeError):
    """KLaVA 서버 호출 실패"""

@dataclass
class StreamResult:
    """스트림의 마지막에 한 번 나오는 최종 결과."""
    value: str

def sse_field(line: str, field: str) -> str:
    value = line[len(field) + 1:]
    if value.startswith(" "):
        return value[1:]
    return value

def iter_sse_events(response):
    """SSE 응답을 (event 이름, data dict) 쌍으로 하나씩 돌려준다."""
    event = ""
    data = ""
    for raw in response.iter_lines():
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if line == "": # 빈 줄 = 프레임 하나가 끝났다
            if event:
                yield event, json.loads(data) if data else {}
            event = ""
            data = ""
        elif line.startswith("event:"):
            event = sse_field(line, "event")
        elif line.startswith("data:"):
            data = sse_field(line, "data")

    # 빈 줄 없이 연결이 닫혀도 마지막 프레임을 처리한다
    if event:
        yield event, json.loads(data) if data else {}


class KLaVAClient:
    """KLaVA의 추론 엔드포인트 호출 경로"""
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        img_path: str,
        prompt: str,
        history: list[dict] | None = None,
        temperature: float = .1,
        max_new_tokens: int = 2048,
        enable_thinking: bool = False,
    ):
        """이미지 전체를 보고 답."""
        return self._post("/generate",{
            "img_path": img_path,
            "prompt": prompt,
            "history": history,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "enable_thinking": enable_thinking,
        })
    
    def reread(
        self,
        img_path: str,
        prompt: str,
        bbox: list[float],
        history: list[dict] | None = None,
        temperature: float = .1,
        max_new_tokens: int = 2048,
        enable_thinking: bool = False,
    ):
        """bbox 영역만 보고 답."""
        return self._post("/reread",{
            "img_path": img_path,
            "prompt": prompt,
            "bbox": bbox,
            "history": history,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "enable_thinking": enable_thinking,
        })

    def stream_generate(
        self,
        img_path: str,
        prompt: str,
        history: list[dict] | None = None,
        temperature: float = .1,
        max_new_tokens: int = 2048,
        enable_thinking: bool = False,
    ):
        """증분 텍스트를 전달한 뒤 generate와 같은 최종 결과를 반환."""
        return self._post_stream("/generate", {
            "img_path": img_path,
            "prompt": prompt,
            "history": history,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "enable_thinking": enable_thinking,
            "stream": True,
        })

    def _post(self, path: str, payload: dict) -> str:
        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as exc:
            raise KLaVAError(f"KLaVA 서버 호출 실패: {exc}") from exc

        if response.status_code != 200:
            raise KLaVAError(f"KLaVA 서버 응답 오류({response.status_code}): {response.text[-256:]}")

        return response.json()["response"]

    def _post_stream(self, path: str, payload: dict):
        """SSE 응답을 증분 문자열과 최종 결과로 변환."""
        url = f"{self.base_url}{path}"
        done = False
        try:
            # 소비를 중단해도 연결이 남지 않도록 응답을 닫는다
            with requests.post(url, json=payload, timeout=self.timeout, stream=True) as response:
                if response.status_code != 200:
                    raise KLaVAError(
                        f"KLaVA 서버 응답 오류({response.status_code}): {response.text[-256:]}"
                    )
                for event, body in iter_sse_events(response):
                    if event == "delta":
                        yield body.get("text", "")
                    elif event == "done":
                        done = True
                        yield StreamResult(value=body.get("response", ""))
                    elif event == "error":
                        raise KLaVAError(f"KLaVA 서버 오류: {body.get('detail', '')}")
        except requests.exceptions.RequestException as exc:
            raise KLaVAError(f"KLaVA 서버 호출 실패: {exc}") from exc

        if not done:
            # 완료 이벤트가 없으면 최종 응답을 확정할 수 없다
            raise KLaVAError("KLaVA 스트림이 done 이벤트 없이 끊겼습니다.")
