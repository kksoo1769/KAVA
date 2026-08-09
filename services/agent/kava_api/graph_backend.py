"""대화 상태와 에이전트 그래프를 연결."""
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage

from kava_api.dependencies import DEFAULT_RECURSION_LIMIT, AgentDependencies
from kava_api.domain import ConversationState, ImageRef, Message, OcrCache
from kava_api.errors import KAVAError, UpstreamFailure
from kava_api.graph import build_agent_graph
from kava_api.graph_state import ImgKind, carry_over, initial_state
from kava_api.routing import PROGRESS_LABELS


@dataclass(frozen=True, slots=True)
class TurnResult:
    """한 턴의 결과 중 다음 턴에 필요한 값들."""
    answer: str
    img_kind: ImgKind
    ocr_cache: OcrCache | None

# Progress와 Delta는 스트림에서 전달할 값 하나만 담는다
@dataclass(frozen=True, slots=True)
class Progress:
    """지금 어떤 단계가 진행 중인지 알리는 값. 문구는 routing.PROGRESS_LABELS에서 온다."""
    label: str

@dataclass(frozen=True, slots=True)
class Delta:
    """화면에 이어 붙일 답변 조각."""
    text: str

def to_graph_messages(messages: list[Message], question: str) -> list:
    """저장된 대화 뒤에 이번 질문을 붙이고 LangChain 메시지로 변환."""
    converted = []
    for msg in messages:
        if msg.role == "user":
            converted.append(HumanMessage(msg.content))
        else:
            converted.append(AIMessage(msg.content))
    converted.append(HumanMessage(question))
    return converted

def extract_answer(final_state: dict, *, sent: int = 0) -> str:
    """이번 턴에 생성된 메시지에서 최종 답변을 추출."""
    messages = final_state.get("messages") or []
    for msg in reversed(messages[sent:]):
        if not isinstance(msg, AIMessage):
            continue # 사람의 질문과 도구 결과는 제외한다
        if msg.tool_calls:
            continue # 도구 호출을 요청한 중간 메시지는 제외한다
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content.strip()
    raise UpstreamFailure("답변을 생성하지 못하였습니다.")

class GraphBackend:
    """컴파일한 그래프를 턴 단위로 실행."""
    def __init__(self, deps: AgentDependencies, *, recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> None:
        self._graph = build_agent_graph(deps)
        self._recursion_limit = recursion_limit

    async def stream_turn(
        self,
        state: ConversationState,
        question: str,
        *,
        img: ImageRef | None
    ) -> AsyncIterator[Progress | Delta | TurnResult]:
        """진행 단계와 답변 조각을 전달한 뒤 최종 결과를 반환.

        values는 최종 상태에, debug는 진행 단계에, custom은 화면용 답변에 사용한다.
        """
        img_kind, cached = state.derived_for(img)

        img_path = ""
        if img is not None:
            img_path = img.path

        ocr_text = ""
        if cached is not None:
            ocr_text = cached.text

        graph_input = initial_state(
            to_graph_messages(state.messages, question),
            img_path=img_path,
            img_kind=img_kind,
            ocr_cache=ocr_text,
            ocr_done=cached is not None,
        )
        sent = len(graph_input["messages"])

        final_state: dict = {}
        last_label = ""
        try:
            async for mode, payload in self._graph.astream(
                graph_input,
                stream_mode=["values", "debug", "custom"],
                config={"recursion_limit": self._recursion_limit},
            ):
                if mode == "values":
                    final_state = payload
                elif mode == "custom":
                    yield Delta(payload["text"])
                elif mode == "debug" and payload["type"] == "task":
                    node_name = payload["payload"]["name"]
                    label = PROGRESS_LABELS.get(node_name, "")
                    # 등록되지 않았거나 직전과 같은 진행 문구는 보내지 않는다
                    if label != "" and label != last_label:
                        last_label = label
                        yield Progress(label)
        except KAVAError:
            raise
        except Exception as exc:
            # 내부 원인은 로그에만 남고 HTTP 응답에는 고정 문구가 사용된다
            raise UpstreamFailure(f"그래프 실행이 실패하였습니다: {type(exc).__name__}: {exc}") from exc

        carried = carry_over(final_state)

        ocr_cache = None
        if carried["ocr_done"]:
            ocr_cache = OcrCache(carried["ocr_cache"])

        # 최종 답변은 델타가 아닌 그래프 상태에서 가져온다
        yield TurnResult(
            answer=extract_answer(final_state, sent=sent),
            img_kind=carried["img_kind"],
            ocr_cache=ocr_cache,
        )

    async def run_turn(
        self,
        state: ConversationState,
        question: str,
        *,
        img: ImageRef | None
    ) -> TurnResult:
        """스트림에서 진행 정보와 답변 조각을 제외한 최종 결과만 반환."""
        async for item in self.stream_turn(state, question, img=img):
            if isinstance(item, TurnResult):
                return item
        raise UpstreamFailure("답변을 만들지 못했습니다.")
