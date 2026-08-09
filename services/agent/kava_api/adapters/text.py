import json
import logging
import re
import uuid

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from kava_api.adapters.klava import iter_sse_events


logger = logging.getLogger(__name__)

CHAT_URL = "http://127.0.0.1:8001/chat"

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# 화면에 표시하지 않을 태그
TOOL_CALL_OPEN = "<tool_call>"
THINK_CLOSE = "</think>"

# 일반 응답과 스트리밍 응답에서 함께 쓰는 문구
THINK_OVERFLOW_MESSAGE = "(응답에 필요한 사고 과정이 너무 길어 답변하지 못하였습니다. 질문을 단순화하세요.)"


def to_exaone_messages(messages) -> list[dict]:
    """LangChain 메시지를 EXAONE 템플릿 list로 파싱"""
    out = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            out.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            out.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            prompt = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                prompt["tool_calls"] = [
                    {"name": tool_call["name"], "arguments": tool_call["args"]}
                    for tool_call in msg.tool_calls
                ]
            out.append(prompt)
        elif isinstance(msg, ToolMessage):
            out.append({"role": "tool", "content": str(msg.content)})
        else:
            raise ValueError(f"유효하지 않는 메시지 타입: {type(msg)}")
    return out

def parse_tool_calls(answer: str) -> tuple[str, list[dict]]:
    """답변 본문(answer)에서 tool call를 파싱"""
    calls = []
    n_broken = 0
    for match in TOOL_CALL_RE.finditer(answer):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError: # 형식이 깨진 호출은 개수 세기
            n_broken += 1
            continue
        calls.append({
            "name": obj.get("name", ""),
            "args": obj.get("arguments", {}) or {},
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        })
    content = TOOL_CALL_RE.sub("", answer).strip()
    if n_broken and not calls:
        content += "\n(도구 호출 형식 오류로 실행되지 않았습니다. 형식을 확인해 다시 시도하세요.)"
    return content, calls

def tag_prefix_len(text: str, tag: str) -> int:
    """다음 청크에서 태그로 이어질 수 있는 꼬리의 길이를 반환."""
    longest = min(len(tag) - 1, len(text))
    for n in range(longest, 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0

class ScreenTextGuard:
    """화면에 보낼 텍스트만 관리. 태그 후보와 끝 공백은 다음 청크까지 보류한다."""

    def __init__(self, skip_think: bool = False):
        self.skip_think = skip_think
        self.pending = "" # 아직 내보내지 않은 꼬리
        self.emitted = "" # 화면에 내보낸 텍스트
        self.started = False # 앞 공백 처리 여부
        self.stopped = False # 도구 호출 뒤의 출력 중단 여부

    def feed(self, delta: str) -> str:
        """증분 텍스트를 받아, 지금 화면에 내보내도 되는 부분만 돌려준다."""
        if self.stopped: # 도구 호출 뒤에는 더 내보내지 않는다
            return ""
        self.pending += delta

        if self.skip_think: # 추론이 끝날 때까지 보류한다
            _, sep, tail = self.pending.partition(THINK_CLOSE)
            if not sep:
                return "" # 아직 추론 중이다
            self.pending = tail
            self.skip_think = False

        head, sep, _ = self.pending.partition(TOOL_CALL_OPEN)
        if sep:
            # 도구 호출 앞부분까지만 내보낸다
            self.stopped = True
            self.pending = ""
            out = head.rstrip()
        else:
            # 태그로 이어질 수 있는 꼬리는 보류한다
            hold = tag_prefix_len(self.pending, TOOL_CALL_OPEN)
            cut = len(self.pending) - hold # cut 뒤는 다음 청크까지 보류한다
            # 태그 앞에서 제거될 수 있는 공백도 함께 보류한다
            while cut > 0 and self.pending[cut - 1].isspace():
                cut -= 1
            out, self.pending = self.pending[:cut], self.pending[cut:]

        if not self.started:
            out = out.lstrip() # 최종 응답과 같이 앞 공백을 제거한다
            if out:
                self.started = True
        self.emitted += out
        return out

    def finish(self, final_content: str) -> str:
        """최종 응답에서 아직 내보내지 않은 부분을 반환."""
        if final_content.startswith(self.emitted):
            return final_content[len(self.emitted):]
        logger.warning(
            "스트리밍으로 내보낸 텍스트가 최종 답변의 접두사가 아니다. "
            "화면 텍스트와 최종 답변이 어긋난다(내보낸 길이=%d, 최종 길이=%d).",
            len(self.emitted), len(final_content),
        )
        return ""

class ChatEXAONE(BaseChatModel):
    """'/chat' 앤드포인트를 사용하는 LangChain 기반 chat model"""
    url: str = CHAT_URL
    temperature: float = 0.1
    max_new_tokens: int = 16384
    timeout: float = 600.
    enable_thinking: bool = False

    # over-ride
    def bind_tools(self, tools, **kwargs):
        schemas = [convert_to_openai_tool(tool) for tool in tools] # convert_to_openai_tool: tool 함수의 이름, docstring, 인자 스키마를 json으로 변환
        return self.bind(tools=schemas, **kwargs)

    def build_payload(self, messages, tools=None, stream: bool = False) -> dict:
        """일반 응답과 스트리밍 응답에 공통으로 사용할 요청 본문을 생성."""
        payload = {
            "messages": to_exaone_messages(messages),
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "enable_thinking": self.enable_thinking,
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        return payload

    def to_message(self, data: dict) -> AIMessage:
        """채팅 응답을 최종 AIMessage로 변환."""
        if not data.get("think_closed", True):
            # 사고만 한 경우에 질문 단순화를 요청
            return AIMessage(content=THINK_OVERFLOW_MESSAGE)

        content, tool_calls = parse_tool_calls(data["response"])
        return AIMessage(
            content=content, tool_calls=tool_calls,
            additional_kwargs={"reasoning": data.get("reasoning", "")}
        )

    # over-ride
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        payload = self.build_payload(messages, tools=kwargs.get("tools"))

        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        msg = self.to_message(response.json())
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # over-ride
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """추론 내용과 도구 호출을 제외한 응답 청크를 차례로 반환.

        마지막 청크에는 일반 응답과 동일한 내용과 도구 호출 정보를 담는다.
        """
        payload = self.build_payload(messages, tools=kwargs.get("tools"), stream=True)

        guard = ScreenTextGuard(skip_think=self.enable_thinking)
        data = None
        with requests.post(self.url, json=payload, timeout=self.timeout, stream=True) as response:
            response.raise_for_status()
            for event, body in iter_sse_events(response):
                if event == "delta":
                    visible = guard.feed(body.get("text", ""))
                    if visible:
                        chunk = ChatGenerationChunk(message=AIMessageChunk(content=visible))
                        if run_manager:
                            run_manager.on_llm_new_token(visible, chunk=chunk)
                        yield chunk
                elif event == "done":
                    data = body
                elif event == "error":
                    raise RuntimeError(f"/chat 스트리밍 오류: {body.get('detail', '')}")

        if data is None:
            # 완료 이벤트가 없으면 도구 호출을 포함한 최종 응답을 확정할 수 없다
            raise RuntimeError("/chat 스트림이 done 이벤트 없이 끊겼습니다.")

        msg = self.to_message(data)

        # LangChain이 병합할 수 있도록 도구 인자를 JSON 문자열로 전달한다
        tool_chunks = []
        for index, call in enumerate(msg.tool_calls):
            tool_chunks.append(tool_call_chunk(
                name=call["name"],
                args=json.dumps(call["args"], ensure_ascii=False),
                id=call["id"],
                index=index,
            ))

        yield ChatGenerationChunk(message=AIMessageChunk(
            content=guard.finish(msg.content),
            tool_call_chunks=tool_chunks,
            additional_kwargs=msg.additional_kwargs,
        ))

    @property
    def _llm_type(self) -> str:
        return "exaone-shared-runtime"
