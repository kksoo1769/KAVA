import os
import re

from langchain_core.tools import tool
from simpleeval import simple_eval
from tavily import TavilyClient

# KLAVA_REREAD_URL = "http://127.0.0.1:8001/reread"

# HTTP_TIMEOUT = float(os.environ.get("KAVA_HTTP_TIMEOUT", "180"))

POW_RE = re.compile(r"(?<![\^*])\^(?!\^)")

@tool
def calc(expression: str) -> str:
    """산술 수식을 계산. 평균의 합계는 괄호로 묶고 단위 변환식은 직접 작성한다."""
    expr = POW_RE.sub("**", expression) # 캐럿을 거듭제곱 연산자로 바꾼다
    try:
        return simple_eval(expr)
    except Exception as e: # noqa: BLE001 - 도구 실패는 모델이 읽고 다시 시도할 문자열이어야 한다
        return f'산술 계산 실패: {e}'

@tool
def web_search(query: str) -> str:
    """최신 정보나 외부 사실을 검색해 요약과 출처 URL을 반환.

    query에는 한국어 또는 영어 검색어를 입력한다. 이미지 판독과 계산에는 사용하지 않는다.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "웹 검색 실패: TAVILY_API_KEY 환경변수가 설정되지 않았습니다."
    try:
        res = TavilyClient(api_key=api_key).search(query, max_results=3, include_answer=True)
    except Exception as e: # noqa: BLE001 - 검색 실패로 그래프를 죽이지 않고 모델에게 알린다
        return f"웹 검색 실패: {e}"

    parts = []
    if res.get("answer"):
        parts.append(f"[요약] {res['answer']}")
    hits = res.get("results", [])
    if not hits and not parts:
        return f"'{query}'에 대한 웹 검색 결과가 없습니다."
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] {h.get('title', '')}\n{h.get('content', '')}\n출처: {h.get('url', '')}")
    return "\n\n".join(parts)

# @tool
# def run_reread(state: Annotated[dict, InjectedState], bbox_ids: list[int], prompt: str) -> str:
#     """근거 자료에서 번호가 붙은 부분을 골라, 그 부분만 잘라 확대해 시각 모델에게 다시 읽힌다.

#     값이 깨져 보이거나 명백히 이상할 때, 자료들이 서로 어긋날 때,
#     질문에 답하기에 읽어낸 내용이 모자랄 때 사용한다.
#     직접 좌표를 만들지 말고, 반드시 근거 자료에 붙은 번호로만 지정한다.

#     Args:
#         bbox_ids: 다시 읽을 번호 목록. 예: [3, 5].
#             관련된 번호를 함께 넘기면 각각을 개별로 확대해 읽고,
#             그 전체를 포함하는 영역도 함께 넘겨 배치 관계까지 확인한다.
#         prompt: 그 부분에서 무엇을 확인할지 묻는 구체적인 한국어 질문.
#             시각 모델은 잘라낸 이미지만 보므로 "이 영역"이 아니라 "이미지"라고 지칭한다.
#             확인하려는 값 자체는 질문에 넣지 않는다.

#     Returns:
#         시각 모델이 지정한 각 부분과 전체 영역을 보고 답한 결과.
#     """
#     if not state.get("bboxes"):
#         return "이 이미지에는 번호가 붙은 부분이 없어 번호 기반 재판독을 쓸 수 없습니다. 좌표로 직접 보려면 run_reread_region을 사용하세요."
#     idx_to_bbox = {bbox["idx"]: bbox["bbox"] for bbox in state["bboxes"]}
#     selected = [(i, idx_to_bbox[i]) for i in bbox_ids if i in idx_to_bbox]
#     missing = [i for i in bbox_ids if i not in idx_to_bbox]
#     if not selected:
#         return f"재판독 실패: 유효한 번호가 없습니다. 받은 번호: {bbox_ids}. 근거 자료에 실제로 붙어 있는 번호로 다시 지정하세요."

#     def reread_one(bbox: list[float]) -> str:
#         try:
#             response = requests.post(
#                 KLAVA_REREAD_URL,
#                 json={
#                     "img_path": state["img_path"],
#                     "prompt": prompt,
#                     "bbox": bbox,
#                     "temperature": .1,
#                     "max_new_tokens": 1024
#                 },
#                 timeout=HTTP_TIMEOUT
#             )
#             if response.status_code != 200:
#                 return f"KLaVA 서버 응답 오류: {response.text[-256:]}"
#             return response.json()["response"]
#         except requests.exceptions.RequestException as e:
#             return f"KLaVA 서버 호출 실패: {e}"

#     results = []
#     for i, bbox in selected:
#         results.append(f"[{i}번 재판독] {reread_one(bbox)}")

#     if len(selected) >= 2:
#         coords = [bbox for _, bbox in selected]
#         union = [
#             min(c[0] for c in coords),
#             min(c[1] for c in coords),
#             max(c[2] for c in coords),
#             max(c[3] for c in coords),
#         ]
#         results.append(f"[전체 영역 재판독] {reread_one(union)}")

#     if missing:
#         results.append(f"[없는 번호] {missing}은 근거 자료에 없어 건너뛰었습니다. 번호를 다시 확인하세요.")

#     return "\n".join(results)

# @tool
# def run_reread_region(state: Annotated[dict, InjectedState], bbox: list[float], prompt: str) -> str:
#     """글자로 표현되지 않는 시각 정보(체크 표시, 색, 막대 길이, 개수 등)를 확인하려고
#     지정한 영역을 직접 잘라 확대해 시각 모델에게 다시 보인다.

#     글자를 다시 읽는 일은 번호로 고르는 run_reread를 먼저 쓴다.
#     이 도구는 글자가 아닌 시각 정보를 확인해야 하거나, 번호가 붙은 부분이 없을 때만 쓴다.
#     초안이나 인식된 글자로 존재가 확인된 대상만 지정한다.
#     있는지 모르는 대상을 지정하면 시각 모델이 없는 것을 있다고 답할 수 있다.

#     Args:
#         bbox: 확대할 영역 [x1, y1, x2, y2]. 이미지 크기 대비 0에서 1로 정규화한 소수이며
#             (x1, y1)은 왼쪽 위, (x2, y2)는 오른쪽 아래다. 픽셀 값이 아니다.
#             예: 이미지 왼쪽 아래 사분면은 [0.0, 0.5, 0.5, 1.0], 전체는 [0.0, 0.0, 1.0, 1.0].
#             확인할 대상이 잘리지 않도록 주변까지 넉넉히 포함해 잡는다.
#         prompt: 그 부분에서 무엇을 확인할지 묻는 구체적인 한국어 질문.
#             시각 모델은 잘라낸 이미지만 보므로 "이 영역"이 아니라 "이미지"라고 지칭한다.

#     Returns:
#         시각 모델이 그 부분만 보고 답한 결과.
#     """
#     try:
#         response = requests.post(
#             KLAVA_REREAD_URL,
#             json={
#                 "img_path": state["img_path"],
#                 "prompt": prompt,
#                 "bbox": bbox,
#                 "temperature": .1,
#                 "max_new_tokens": 1024
#             },
#             timeout=HTTP_TIMEOUT,
#         )
#         if response.status_code != 200:
#             return f"KLaVA 서버 응답 오류: {response.text[-256:]}"
#         return response.json()["response"]
#     except requests.exceptions.RequestException as e:
#         return f"KLaVA 서버 호출 실패: {e}"
