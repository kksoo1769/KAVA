"""한 턴 동안 graph가 들고 다니는 상태."""
from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

type TaskType = Literal["", "text-only", "natural", "reading"]
type ImgKind = Literal["", "photo", "table", "document", "chart"]

class TurnGraphState(TypedDict):
    messages: Annotated[list, add_messages]
    img_path: str
    task_type: TaskType
    img_kind: ImgKind
    bboxes: list[dict]
    ocr_cache: str
    ocr_done: bool
    ocr_document: dict
    ocr_numeric_acc: dict
    ocr_numeric_fast: dict
    ocr_merged: dict


def initial_state(
    messages: list,
    img_path: str = "",
    img_kind: ImgKind = "",
    ocr_cache: str = "",
    ocr_done: bool = False,
) -> TurnGraphState:
    return {
        "messages": messages,
        "img_path": img_path,
        "task_type": "",
        "img_kind": img_kind,
        "bboxes": [],
        "ocr_cache": ocr_cache,
        "ocr_done": ocr_done,
        "ocr_document": {},
        "ocr_numeric_acc": {},
        "ocr_numeric_fast": {},
        "ocr_merged": {},
    }

def carry_over(state: TurnGraphState) -> dict:
    """다음 턴에 이어 받을 상태를 정한다."""
    return {
        "img_kind": state["img_kind"],
        "ocr_cache": state["ocr_cache"],
        "ocr_done": state["ocr_done"],
    }
