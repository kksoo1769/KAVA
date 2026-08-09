from typing import Literal
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[dict] | None = None

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    tools: list[dict] | None = None
    temperature: float = .6
    max_new_tokens: int = 4096
    enable_thinking: bool = True
    # True이면 응답을 SSE로 전달한다
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    reasoning: str = ""
    think_closed: bool = True
