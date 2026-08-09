"""LangChain 메시지를 KLaVA가 받는 history 형태로 변환."""
from langchain_core.messages import AIMessage, HumanMessage


def role_of(message) -> str | None:
    """KLaVA history에 넣을 역할(user, assistant)을 반환하고, 넣지 않을 메시지는 None을 반환."""
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage) and not message.tool_calls:
        return "assistant"
    return None # SystemMessage, ToolMessage, 도구 호출만 한 AIMessage

def content_of(message) -> str:
    content = message.content
    return content.strip() if isinstance(content, str) else ""

def to_klava_history(messages: list) -> list[dict]:
    """user, assistant 완결된 쌍만 뽑아 KLaVA history 형태로 변환."""
    candidates = []
    for msg in messages:
        role = role_of(msg)
        content = content_of(msg)
        if role and content:
            candidates.append({"role": role, "content": content})

    history: list[dict] = []
    idx = 0
    while idx + 1 < len(candidates): # <user, assistant>의 완결된 쌍만 추출
        first, second = candidates[idx], candidates[idx + 1]
        if first["role"] == "user" and second["role"] == "assistant":
            history.append(first)
            history.append(second)
            idx += 2
        else: # 쌍을 이루지 못한 경우는 버리기
            idx += 1
    return history
