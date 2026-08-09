"""KAVA 그래프를 수동으로 점검."""
import json
import sys
import time
from pathlib import Path
from pprint import pprint

REPO_ROOT = Path(__file__).resolve().parents[3]

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from kava_api.dependencies import build_default_dependencies
from kava_api.graph import build_agent_graph
from kava_api.graph_state import initial_state

load_dotenv(REPO_ROOT / ".env")

SAMPLES = REPO_ROOT / "docs"
CASES = [
    ("3 + 2 * 4 - 2 // 5 ** 2 의 값을 구해줘.", None),
    ("임진왜란이 일어난 연도와 그 당시 조선의 왕을 알려주세요.", None),
    ("이미지에서 무엇이 보이나요?", "vlm_comparison/samples/image3.jpg"),
    ("이미지를 한국어로 설명해주세요.", "vlm_comparison/samples/image9.jpg"),
    ("이미지를 자세히 설명해주세요.", "dataset_samples/stage2/korean_gqa_71711.jpg"),
    ("2023년 3분기 회사의 매출액, 영업이익, 순이익 실적을 알려줘.", "vlm_comparison/heldout_kdtc/document_41.jpg"),
    ("대학생들의 진로 선택 이유들의 빈도와 비율을 내림차순으로 정렬해줘.", "vlm_comparison/heldout_kdtc/table_22.jpg"),
    ("이 차트의 핵심 내용을 요약해주세요.", "vlm_comparison/heldout_kdtc/chart_31.jpg"),
    ("이미지에서 모든 글자들을 읽어줘.", "dataset_samples/stage2/ocr_ke_71730.jpg"),
]

BULKY_FIELDS = {"ocr_document", "ocr_numeric_acc", "ocr_numeric_fast", "ocr_merged", "bboxes"}


def compact(data):
    """긴 OCR 필드를 크기와 앞부분만으로 줄인다."""
    if not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        if key in BULKY_FIELDS and value:
            encoded = json.dumps(value, ensure_ascii=False)
            result[key] = {
                "size": f"{len(encoded):,}자",
                "preview": encoded[:100] + ("..." if len(encoded) > 100 else ""),
            }
        else:
            result[key] = value
    return result


def run_case(graph, number, question, relative_path):
    img_path = str(SAMPLES / relative_path) if relative_path else ""
    if img_path and not Path(img_path).exists():
        print(f"[사례 {number}] 이미지 없음, 건너뜀: {img_path}")
        return

    print(f"\n{'=' * 80}")
    print(f"[사례 {number}] {question}")
    print(f"이미지: {img_path or '(없음)'}")

    started = time.perf_counter()
    for event in graph.stream(
        initial_state([HumanMessage(question)], img_path), stream_mode="debug"
    ):
        payload = event["payload"]
        if event["type"] == "task":
            print(f"\n[Step {event['step']} 시작] {payload['name']}")
            pprint(compact(payload["input"]), sort_dicts=False, width=140)
        elif event["type"] == "task_result":
            print(f"\n[Step {event['step']} 종료] {payload['name']}")
            if payload.get("error"):
                print("오류:")
                pprint(payload["error"], sort_dicts=False)
            else:
                pprint(compact(payload.get("result")), sort_dicts=False, width=140)

    print(f"\n[사례 {number} 완료: {time.perf_counter() - started:.2f}초]")


def main(argv):
    wanted = {int(a) for a in argv} if argv else None
    graph = build_agent_graph(build_default_dependencies())

    for number, (question, relative_path) in enumerate(CASES, start=1):
        if wanted is None or number in wanted:
            run_case(graph, number, question, relative_path)


if __name__ == "__main__":
    main(sys.argv[1:])
