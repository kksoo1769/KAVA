"""그래프에서 사용하는 프롬프트 정리."""
ROUTER_PROMPT = "이 이미지의 종류로 가장 알맞은 것은 무엇입니까? 다음 넷 중 하나의 단어로만 답하세요: 문서, 표, 차트, 사진" # 학습 시 사용한 프롬프트
SYSTEM_TEXT = """당신은 한국어 질의응답을 하는 어시스턴트입니다.
모든 출력은 한국어로만 작성하고, 아래의 행동 지침들을 반드시 따릅니다.

[행동 지침]
1. 산술 계산은 아무리 간단해 보여도 직접 하지 않고 반드시 calc 도구를 사용한다.
    여러 값의 평균을 구할 때는 합계를 괄호로 묶는다. 예시) (1+2+4)/5
    거듭제곱의 연산 기호는 **를 사용한다. 예시) 2**10
2. 학습된 지식만으로 확신할 수 없는 최신, 외부 정보는 web_search 도구를 사용한다.
3. 사용한 도구들의 결과를 기반으로 근거에 입각해 한국어로 최종 답변한다.
"""
OCR_HEAD = "[이미지 OCR 구조]"
OCR_TAIL = """표의 각 열은 탭 문자로 구분되어 있습니다. [미인식]은 OCR을 통해 내용을 확인할 수 없어 임시로 둔 셀이며, 실제 이미지와 위의 구조를 함께 확인하여 답하세요."""
PROMPT_TAIL = "위 지시문을 반복하지 말고 다음 질문에 대한 답을 작성하세요."

def build_ocr_prompt(evidence: str, question: str) -> str:
    """OCR의 근거와 질문을 klava_with_ocr용 프롬프트로 조립해 반환."""
    if not evidence.strip():
        return (
            f"OCR로 글자를 추출하지 못하였습니다. 이미지만 보고 답하세요.\n\n"
            f"{PROMPT_TAIL}\n\n"
            f"{question}"
        )
    return (
        f"{OCR_HEAD}\n\n"
        f"{evidence}\n\n"
        f"{OCR_TAIL}\n\n"
        f"{PROMPT_TAIL}\n\n"
        f"{question}"
    )
