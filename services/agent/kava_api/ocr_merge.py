import copy
import re

NUMBER_PATTERN = (
    r"[+−-]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?"
)

UNIT_PATTERN = (
    r"(?:"
    r"%|‰|퍼센트|"
    r"㎍|µg|mg|㎎|g|kg|㎏|그램|킬로그램|톤|"
    r"mL|ml|㎖|L|ℓ|밀리리터|리터|"
    r"mm|cm|km|m|㎡|㎥|"
    r"℃|℉|°C|°F|"
    r"원|만원|억원|"
    r"개|명|건|회|"
    r"kWh|kW|MW|"
    r"km/h|m/s|km/h²|m/s²"
    r")"
)

QUANTITY_RE = re.compile(
    rf"^\s*"
    rf"(?P<prefix>[₩$€£¥]?\s*)"
    rf"(?P<number>{NUMBER_PATTERN})"
    rf"\s*(?P<unit>{UNIT_PATTERN})?"
    rf"\s*$",
    re.IGNORECASE,
)


def is_numeric(text: str) -> bool:
    return bool(text and any(char.isdigit() for char in text) and QUANTITY_RE.fullmatch(text))

def digit_count(text: str) -> int:
    return sum(char.isdigit() for char in text)

def round_coords(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "bbox" and isinstance(v, list):
                node[k] = [round(float(elem), 2) for elem in v]
            elif k == "polygon" and isinstance(v, list):
                node[k] = [[round(float(elem), 2) for elem in point] for point in v]
            else:
                round_coords(v)
    elif isinstance(node, list):
        for item in node:
            round_coords(item)

def lines_inside(lines: list[dict], bbox: list[float]) -> list[dict]:
    x1, y1, x2, y2 = bbox
    matched = []
    for line in lines:
        lx1, ly1, lx2, ly2 = line["bbox"]
        cx, cy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            matched.append(line)
    return sorted(matched, key=lambda x: (x["bbox"][1], x["bbox"][0])) # 위: 아래, 왼: 오 정렬

def restore_decimal_zero(row: list[dict]) -> None:
    def number_part(cell: dict) -> str | None:
        match = QUANTITY_RE.fullmatch(cell["text"] or "")
        return match.group("number") if match else None

    peer_count = sum(
        bool(number and re.fullmatch(r"\d+\.0", number))
        for cell in row
        if (number := number_part(cell)) is not None
    )
    if peer_count < 3:
        return

    for cell in row:
        text = cell["text"]
        match = QUANTITY_RE.fullmatch(text or "")
        if not match:
            continue

        number = match.group("number")
        if re.fullmatch(r"\d{2,}", number) and number.endswith("0"):
            corrected = number[:-1] + ".0"
            cell["text"] = (
                text[:match.start("number")]
                + corrected
                + text[match.end("number"):]
            )
            cell["source"] = "row-consensus"

def replace_number_preserving_unit(
    original: str,
    candidate: str,
) -> str | None:
    old = QUANTITY_RE.fullmatch(original or "")
    new = QUANTITY_RE.fullmatch(candidate or "")
    if not old or not new:
        return None

    old_number = old.group("number")
    new_number = new.group("number")
    if digit_count(old_number) != digit_count(new_number):
        return None

    return original[:old.start("number")] + new_number + original[old.end("number"):]

def merge_ocr_results(
    document_result: dict,
    accurate_result: dict,
    fast_result: dict
) -> dict:
    result = copy.deepcopy(document_result)
    document = result.get("document", {})
    document_lines = document.get("lines", [])
    accurate_lines = accurate_result.get("lines", [])
    fast_lines = fast_result.get("lines", [])

    for block in document.get("blocks", []):
        if block.get("type") != "table" or not block.get("table"):
            continue

        for row in block["table"]["rows"]:
            for cell in row:
                if not cell["text"]: # 표 구조는 잡았지만 셀 내의 문자열은 인식하지 못한 경우
                    matches = lines_inside(document_lines, cell["bbox"])
                    if matches:
                        cell["text"] = " ".join(x["text"] for x in matches)
                        cell["source"] = "document-line"
                        cell["confidence"] = max(x["confidence"] for x in matches)

                accurate = [line for line in lines_inside(accurate_lines, cell["bbox"]) if is_numeric(line["text"])]
                if accurate:
                    candidate = max(accurate, key=lambda line: line.get("confidence", 0))
                    corrected = replace_number_preserving_unit(cell["text"], candidate["text"])
                    if corrected is not None:
                        cell["text"] = corrected
                        cell["source"] = "numeric-accurate"
                        cell["confidence"] = candidate["confidence"]

                # 여전히 빈 셀만 fast 결과로 채우기
                if not cell["text"]:
                    fast = [line for line in lines_inside(fast_lines, cell["bbox"]) if is_numeric(line["text"])]
                    if fast:
                        candidate = max(fast, key=lambda line: line.get("confidence", 0))
                        cell["text"] = candidate["text"]
                        cell["source"] = "numeric-fast"
                        cell["confidence"] = candidate["confidence"]
            restore_decimal_zero(row)

    round_coords(result) # KLaVA는 소숫점 둘째 자리까지 표시되는 정규화 bbox로 학습되었으니 이에 맞춤.
    result["engine"] = "apple-vision-hybrid"
    result["mode"] = "hybrid"
    return result

def render_ocr(result: dict, max_chars: int = 4096) -> str:
    rendered: list[str]  = []
    used = 0

    def _append(text: str) -> bool:
        nonlocal used
        extra = len(text) + (2 if rendered else 0)
        if used + extra > max_chars:
            return False

        rendered.append(text)
        used += extra
        return True

    document = result.get("document", {})
    for block in document.get("blocks", []):
        block_type = block.get("type")
        if block_type == "title" and block.get("text"):
            _append("[제목]\n" + block["text"])
        elif block_type == "paragraph" and block.get("text"):
            _append("[문단]\n" + block["text"])
        elif block_type == "list" and block.get("list"):
            lines = [f'{x["marker"]} {x["text"]}' for x in block["list"]["items"]]
            _append("[목록]\n" + "\n".join(lines))
        elif block_type == "table" and block.get("table"):
            rows = ["\t".join(cell["text"] or "[미인식]" for cell in row) for row in block["table"]["rows"]]
            _append("[표]\n" + "\n".join(rows))

    if not rendered:
        _append(document.get("transcript", "")[:max_chars])
    return "\n\n".join(rendered).strip()
