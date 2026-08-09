KIND_MAP = {"사진": "photo", "차트": "chart", "문서": "document", "표": "table"}
NODE_ROUTER = "router"
NODE_TEXT = "text"
NODE_TEXT_TOOLS = "text_tools"
NODE_VISION = "vision"
NODE_OCR_DOC = "ocr_doc"
NODE_OCR_NUM_ACC = "ocr_num_acc"
NODE_OCR_NUM_FAST = "ocr_num_fast"
NODE_OCR_MERGE = "ocr_merge"
NODE_VISION_WITH_OCR = "vision_with_ocr"

# 같은 문구를 연속으로 사용하면 스트리밍 단계에서 한 번만 전달된다
PROGRESS_LABELS = {
    NODE_ROUTER: "이미지 종류를 확인하는 중",
    NODE_OCR_DOC: "이미지에서 글자를 읽는 중",
    NODE_OCR_NUM_ACC: "이미지에서 글자를 읽는 중",
    NODE_OCR_NUM_FAST: "이미지에서 글자를 읽는 중",
    NODE_OCR_MERGE: "이미지에서 글자를 읽는 중",
    NODE_VISION: "이미지를 보고 답변하는 중",
    NODE_VISION_WITH_OCR: "글자와 이미지를 보고 답변하는 중",
    NODE_TEXT: "답변을 작성하는 중",
    NODE_TEXT_TOOLS: "도구를 사용하는 중",
}

# 최종 답변을 만드는 노드만 화면에 텍스트를 보낸다
STREAMING_NODES = frozenset({NODE_VISION, NODE_VISION_WITH_OCR, NODE_TEXT})


def parse(ans: str) -> str:
    """라우터 응답을 네 종류 중 하나로 정규화한다."""
    a = (ans or "").replace(" ", "")
    if "차트" in a or "그래프" in a or "도표" in a:
        return "차트"
    if "표" in a or "테이블" in a:
        return "표"
    if "문서" in a or "서류" in a or "글" in a:
        return "문서"
    return "사진"

def to_img_kind(ans: str) -> str:
    return KIND_MAP[parse(ans)]

def get_task(img_kind: str) -> str:
    return "natural" if img_kind in ("photo", "chart") else "reading"

def nodes_for_task(
        task_type: str,
        has_ocr_cache: bool = False,
        ocr_done: bool = False,
    ) -> list[str]:
    """분류 결과와 OCR 상태에 맞는 다음 노드를 반환."""
    if task_type == "text-only":
        return [NODE_TEXT]
    if task_type == "natural":
        return [NODE_VISION]
    if has_ocr_cache or ocr_done:
        return [NODE_VISION_WITH_OCR]
    return [NODE_OCR_DOC, NODE_OCR_NUM_ACC, NODE_OCR_NUM_FAST]
