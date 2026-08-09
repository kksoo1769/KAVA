"""에이전트 그래프를 구성."""
from functools import partial

from langchain_core.messages import AIMessage, SystemMessage, message_chunk_to_message
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from kava_api.adapters.klava import KLaVAError, StreamResult
from kava_api.dependencies import AgentDependencies
from kava_api.graph_state import TurnGraphState
from kava_api.history import to_klava_history
from kava_api.ocr_merge import merge_ocr_results, render_ocr
from kava_api.prompts import ROUTER_PROMPT, SYSTEM_TEXT, build_ocr_prompt
from kava_api.routing import (
    NODE_OCR_DOC,
    NODE_OCR_MERGE,
    NODE_OCR_NUM_ACC,
    NODE_OCR_NUM_FAST,
    NODE_ROUTER,
    NODE_TEXT,
    NODE_TEXT_TOOLS,
    NODE_VISION,
    NODE_VISION_WITH_OCR,
    get_task,
    nodes_for_task,
    to_img_kind,
)


def build_agent_graph(deps: AgentDependencies):
    """그래프 구조를 만들고 컴파일."""
    # LangGraph가 state만 넘기므로 deps를 미리 연결한다
    bind = lambda node: partial(node, deps=deps)

    builder = StateGraph(TurnGraphState)

    builder.add_node(NODE_ROUTER, bind(router))
    builder.add_node(NODE_TEXT, bind(text))
    builder.add_node(NODE_TEXT_TOOLS, ToolNode(tools=deps.tools))
    builder.add_node(NODE_OCR_DOC, bind(ocr_doc))
    builder.add_node(NODE_OCR_NUM_ACC, bind(ocr_num_acc))
    builder.add_node(NODE_OCR_NUM_FAST, bind(ocr_num_fast))
    builder.add_node(NODE_OCR_MERGE, ocr_merge)
    builder.add_node(NODE_VISION, bind(vision))
    builder.add_node(NODE_VISION_WITH_OCR, bind(vision_with_ocr))

    builder.add_edge(START, NODE_ROUTER)
    builder.add_conditional_edges(
        NODE_ROUTER,
        routing_task,
        [NODE_TEXT, NODE_VISION, NODE_OCR_DOC, NODE_OCR_NUM_ACC, NODE_OCR_NUM_FAST, NODE_VISION_WITH_OCR],
    )
    # text-only 경로는 원 LM(EXAONE-4.0-1.2B) ReAct
    builder.add_conditional_edges(
        NODE_TEXT,
        tools_condition,
        {"tools": NODE_TEXT_TOOLS, END: END},
    )
    builder.add_edge(NODE_TEXT_TOOLS, NODE_TEXT)
    # 자연 이미지, 차트 경로는 KLaVA 즉답
    builder.add_edge(NODE_VISION, END)
    # 문서, 표 경로는 OCR 결과를 KLaVA에 주입하여 응답
    builder.add_edge([NODE_OCR_DOC, NODE_OCR_NUM_ACC, NODE_OCR_NUM_FAST], NODE_OCR_MERGE) # Apple Vision OCR tool(document, accurate, fast) fan-in
    builder.add_edge(NODE_OCR_MERGE, NODE_VISION_WITH_OCR)
    builder.add_edge(NODE_VISION_WITH_OCR, END)

    return builder.compile()

def emit(text: str) -> None:
    """화면에 표시할 답변 조각만 스트림으로 전달."""
    if not text:
        return
    writer = get_stream_writer()
    writer({"text": text})

def stream_klava(chunks) -> str:
    """증분은 화면으로 보내고 StreamResult의 값을 최종 답변으로 반환."""
    for chunk in chunks:
        if isinstance(chunk, StreamResult):
            return chunk.value
        emit(chunk)
    raise KLaVAError("KLaVA 스트림이 최종 결과 없이 끝났습니다.")

def router(state: TurnGraphState, deps: AgentDependencies) -> dict:
    """이미지 종류를 분류해 실행 경로를 선택.

    이전 턴에서 분류한 이미지는 다시 확인하지 않으며 내부 분류 결과는 화면에 보내지 않는다.
    """
    img_path = state["img_path"]
    if not img_path:
        return {"task_type": "text-only", "img_kind": ""}

    img_kind = state["img_kind"]
    if img_kind:
        return {"task_type": get_task(img_kind), "img_kind": img_kind}

    try:
        answer = deps.klava.generate(img_path, ROUTER_PROMPT, temperature=0, max_new_tokens=8)
    except KLaVAError:
        return {"task_type": "natural", "img_kind": ""}

    img_kind = to_img_kind(answer)
    return {"task_type": get_task(img_kind), "img_kind": img_kind}

def routing_task(state: TurnGraphState) -> list[str]:
    task_type = state["task_type"]
    has_ocr_cache = bool(state["ocr_cache"])
    ocr_done = state["ocr_done"]
    return nodes_for_task(task_type, has_ocr_cache, ocr_done)

def ocr_doc(state: TurnGraphState, deps: AgentDependencies) -> dict:
    return {"ocr_document": deps.ocr.run("document", state["img_path"])}

def ocr_num_acc(state: TurnGraphState, deps: AgentDependencies) -> dict:
    return {"ocr_numeric_acc": deps.ocr.run("numeric-accurate", state["img_path"])}

def ocr_num_fast(state: TurnGraphState, deps: AgentDependencies) -> dict:
    return {"ocr_numeric_fast": deps.ocr.run("numeric-fast", state["img_path"])}

def ocr_merge(state: TurnGraphState) -> dict:
    """세 ocr 결과를 합쳐 근거 텍스트와 bbox 목록들을 조립."""
    document = state["ocr_document"]
    numeric_acc = state["ocr_numeric_acc"]
    numeric_fast = state["ocr_numeric_fast"]
    result = merge_ocr_results(document, numeric_acc, numeric_fast)
    return {
        "ocr_merged": result,
        "ocr_cache": render_ocr(result),
        "ocr_done": bool(document),
        "bboxes": result.get("document", {}).get("lines", [])
    }

def vision(state: TurnGraphState, deps: AgentDependencies) -> dict:
    """이미지 답변을 생성하고 증분을 화면으로 전달."""
    question = state["messages"][-1].content
    history = to_klava_history(state["messages"])
    chunks = deps.klava.stream_generate(
        state["img_path"],
        question,
        history=history,
        temperature=.1,
        enable_thinking=False,
    )
    answer = stream_klava(chunks)
    return {"messages": [AIMessage(content=answer)]}

def vision_with_ocr(state: TurnGraphState, deps: AgentDependencies) -> dict:
    """OCR 결과를 근거로 답변하되 생성된 답변만 화면으로 전달."""
    question = state["messages"][-1].content
    evidence = state["ocr_cache"]
    prompt = build_ocr_prompt(evidence, question)
    history = to_klava_history(state["messages"])
    chunks = deps.klava.stream_generate(
        state["img_path"],
        prompt,
        history=history,
        temperature=0,
        enable_thinking=False,
    )
    answer = stream_klava(chunks)
    return {"messages": [AIMessage(content=answer)]}

def text(state: TurnGraphState, deps: AgentDependencies) -> dict:
    """텍스트 청크를 전달하면서 도구 호출 정보가 포함된 최종 메시지를 생성."""
    messages = [SystemMessage(content=SYSTEM_TEXT)] + state["messages"]

    # 도구 호출 정보가 유지되도록 문자열이 아닌 청크를 병합한다
    merged = None

    for chunk in deps.text_model.stream(messages):
        if isinstance(chunk.content, str):
            emit(chunk.content)

        if merged is None:
            merged = chunk
        else:
            merged = merged + chunk

    if merged is None:
        raise RuntimeError("텍스트 모델 스트림이 청크 없이 끝났습니다.")

    return {"messages": [message_chunk_to_message(merged)]}
