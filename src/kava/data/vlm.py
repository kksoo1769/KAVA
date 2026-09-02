"""LLaVA를 따른 VLM 데이터: (이미지 + 대화): EXAONE 챗 템플릿 인코딩 + 시각 자리표시."""

from __future__ import annotations

import json
import os

import torch
from torch.utils.data import Dataset

# 특수 토큰
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
IMAGE_PLACEHOLDER = "<image>"

# VLM 전용 시스템 프롬프트
SYS_VLM = (
    "당신은 이미지와 텍스트를 이해하는 AI 어시스턴트입니다. "
    "이미지가 함께 주어지면 이미지에 실제로 보이는 내용에 근거해 사실대로 답하고, "
    "이미지가 없으면 주어진 텍스트에 근거해 답합니다. "
    "질문이 한국어이면 한국어로, 영어이면 영어로 답합니다."
)

# stage 1 정렬용 캡션 지시들. 아래의 지시들 중 하나를 랜덤으로 선택
CAPTION_INSTRUCTIONS = [
    "이 이미지를 한국어로 설명해 주세요.",
    "사진에 무엇이 보이는지 자세히 설명해 주세요.",
    "이미지의 내용을 묘사해 주세요.",
    "이 장면에서 일어나는 일을 설명해 주세요.",
    "사진 속 모습을 구체적으로 알려 주세요.",
    "이미지를 보고 보이는 것을 서술해 주세요.",
]


def tokenizer_image_token(text: str, tokenizer) -> list[int]:
    """<image>로 분할해 각 조각을 인코딩하고 사이에 IMAGE_TOKEN_INDEX를 끼워 넣는다.

    LLaVA의 tokenizer_image_token과 동일하게 마커가 특정 토큰 ID로 분해되는 것에 의존하지 않아 토크나이저와 무관하게 정확히 마커 개수만큼의 시각 자리표시를 만든다.
    """
    parts = text.split(IMAGE_PLACEHOLDER)
    out: list[int] = []
    for i, part in enumerate(parts):
        if i > 0:
            out.append(IMAGE_TOKEN_INDEX)
        if part:
            out.extend(tokenizer(part, add_special_tokens=False)["input_ids"])
    return out

def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def _render(tokenizer, messages, add_generation_prompt: bool,
            enable_thinking: bool = False) -> list[int]:
    """EXAONE 원본 챗 템플릿으로 렌더하고 <image>를 IMAGE_TOKEN_INDEX로 치환한다.

    skip_think=False로 고정한다. 템플릿 기본값(True)이면 reasoning_content가 있어도
    think가 비워져 CoT를 학습할 수 없다. 원본 템플릿은 loop.last 조건 때문에 대화의
    마지막 메시지에 한해 think를 채우므로, 그 턴을 타깃으로 삼을 때만 프리픽스를
    enable_thinking=True로 끊어야 supervise 구간이 추론부터 시작한다.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        skip_think=False, enable_thinking=enable_thinking,
    )
    return tokenizer_image_token(text, tokenizer)

def encode_vlm_conversation(
    tokenizer, messages: list[dict], max_len: int = 2048,
    add_default_system: bool = True, default_system: str | None = None,
    expect_image: bool = True,
) -> dict | None:
    """대화를 이미지 자리표시가 포함된 토큰과 학습 label로 변환."""
    if add_default_system and (not messages or messages[0]["role"] != "system"):
        messages = [{"role": "system", "content": default_system or SYS_VLM}, *messages]

    seq = _render(tokenizer, messages, add_generation_prompt=False)
    sup = [False] * len(seq)
    last_i = len(messages) - 1
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        # 원본 템플릿은 대화의 마지막 메시지일 때만 think를 채운다(loop.last). 그 턴에서만
        # 프리픽스를 <think>\n 까지 끊어야 seq와 접두가 일치해 추론부터 supervise된다.
        has_think = bool(msg.get("reasoning_content")) and i == last_i
        prefix = _render(tokenizer, messages[:i], add_generation_prompt=True,
                         enable_thinking=has_think)
        start = _common_prefix_len(prefix, seq)
        end = len(_render(tokenizer, messages[: i + 1], add_generation_prompt=False))
        for k in range(start, min(end, len(seq))):
            sup[k] = True

    labels = [seq[t] if sup[t] else IGNORE_INDEX for t in range(len(seq))]

    if max_len is not None and len(seq) > max_len:
        seq, labels = seq[:max_len], labels[:max_len]

    if expect_image and IMAGE_TOKEN_INDEX not in seq:
        return None # 절단으로 이미지 자리표시가 사라진 데이터는 제외한다
    if all(l == IGNORE_INDEX for l in labels):
        return None # 학습할 토큰이 없으면 제외한다
    return {"input_ids": seq, "labels": labels}

def _prepend_image(content: str) -> str:
    """user 콘텐츠 맨 앞에 <image>\\n 마커를 (중복 없이) 붙인다."""
    content = content or ""
    if IMAGE_PLACEHOLDER in content:
        return content
    return f"{IMAGE_PLACEHOLDER}\n{content}".strip()


def caption_messages(row: dict, instruction: str) -> list[dict]:
    """stage 1: {caption}: 1턴 (user=<image>+캡션지시, assistant=caption)."""
    return [
        {"role": "user", "content": _prepend_image(instruction)},
        {"role": "assistant", "content": row["caption"].strip()},
    ]


def conversation_messages(row: dict) -> list[dict]:
    """stage 2의 LLaVA 대화를 message 목록으로 변환.

    모든 턴에서 리터럴 <image>를 먼저 제거한 뒤 멀티모달 행의 첫 user 턴에만 1개를 주입한다.
    2번째 human 턴이나 gpt 답변 안에 남은 마커가 여분의 시각 토큰을 만드는 것을 막는다.

    role_map에 tool 계열을 포함한다. 없으면 도구 결과 턴이 조용히 사라져 "도구를 부른 뒤
    결과를 보지 않고 답을 지어내는" 대화가 만들어진다. 매핑에 없는 role을 만나면 그 턴만
    건너뛰지 않고 행 전체를 드롭한다(조용한 구조 오염보다 조용한 드롭이 안전하다).

    turn의 think는 해당 assistant 메시지의 reasoning_content로 옮긴다. 템플릿이 이 값을
    <think> 블록으로 렌더하고 encode_vlm_conversation이 그 구간까지 supervise한다.
    """
    role_map = {
        "human": "user", "user": "user",
        "gpt": "assistant", "assistant": "assistant",
        "system": "system",
        "tool": "tool", "function": "tool", "observation": "tool",
    }
    has_image = bool(row.get("image"))
    msgs, injected = [], False
    for t in row.get("conversations", []):
        role = role_map.get(str(t.get("from", "")).lower())
        if role is None:
            return [] # 알 수 없는 role: 행 전체 드롭
        content = (t.get("value") or "").replace(IMAGE_PLACEHOLDER, "").strip()
        if has_image and role == "user" and not injected:
            content = f"{IMAGE_PLACEHOLDER}\n{content}".strip()
            injected = True
        msg = {"role": role, "content": content}
        # think 안의 리터럴 <image>도 제거한다. 남기면 tokenizer_image_token이 진짜 마커로
        # 오인해 IMAGE_TOKEN_INDEX를 하나 더 만들고, _merge는 첫 자리만 치환하므로 두 번째가
        # 어휘 범위 밖 타깃으로 남아 손실 계산이 죽는다.
        think = (t.get("think") or "").replace(IMAGE_PLACEHOLDER, "").strip()
        if role == "assistant" and think:
            msg["reasoning_content"] = think
        msgs.append(msg)
    return msgs


def row_to_messages(row: dict, stage: str, rng_idx: int = 0) -> list[dict]:
    if stage == "align":
        instr = CAPTION_INSTRUCTIONS[rng_idx % len(CAPTION_INSTRUCTIONS)]
        return caption_messages(row, instr)
    return conversation_messages(row)


# Dataset과 collate
class VLMDataset(Dataset):
    """manifest(list[dict]) + 이미지 디스크 로드: 인코딩.

    각 row: {"image": 경로, ("caption" | "conversations"), ...}.
    """

    def __init__(
        self, records: list[dict], tokenizer_name: str, stage: str = "align",
        img_size: int = 384, max_len: int = 8192, train_aug: bool = False,
        add_default_system: bool = True, vision_backbone: str = "siglip2",
        siglip_model_id: str = "google/siglip2-so400m-patch16-naflex",
        max_num_patches: int = 784 # 운영 체크포인트 기준. 호출부(train/vlm.py)가 cfg 값을 넘긴다.
    ):
        self.records = records
        self.tokenizer_name = tokenizer_name
        self.stage = stage
        self.img_size = img_size
        self.max_len = max_len
        self.train_aug = train_aug
        self.add_default_system = add_default_system
        self.vision_backbone = vision_backbone
        self.siglip_model_id = siglip_model_id
        self.max_num_patches = max_num_patches
        self.tokenizer = None
        self._siglip_proc = None # SigLIP2 NaFlex 프로세서
        if vision_backbone != "siglip2":
            raise ValueError(
                f"지원하지 않는 vision_backbone: {vision_backbone!r}. siglip2만 지원합니다."
            )

    def _lazy(self):
        if self.tokenizer is None:
            from kava.data.tokenizer import load_tokenizer
            self.tokenizer = load_tokenizer(self.tokenizer_name)
            from kava.vision.siglip2 import build_siglip2_processor
            self._siglip_proc = build_siglip2_processor(self.siglip_model_id, max_num_patches=self.max_num_patches)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        self._lazy()
        row = self.records[i]
        # CoT(think) 행도 일반 대화 경로로 처리한다. conversation_messages가 think를
        # reasoning_content로 옮기고 원본 템플릿이 마지막 턴의 <think>를 채우므로 싱글턴은
        # 옛 전용 경로와 토큰 단위로 같고, 멀티턴은 중간 턴이 보존된다(옛 경로는 첫 user와
        # 마지막 assistant만 남기고 중간 턴을 버렸다).
        messages = row_to_messages(row, self.stage, rng_idx=i)
        enc = None if not messages else encode_vlm_conversation(
            self.tokenizer, messages, max_len=self.max_len,
            add_default_system=self.add_default_system,
            expect_image=bool(row.get("image")),
        )
        if enc is None: # Fallback: 인코딩 드롭: 다음 인덱스
            return self.__getitem__((i + 1) % len(self.records))
        # 텍스트 전용 데이터는 pixel을 사용하지 않는다
        img_path = row.get("image")
        if img_path:
            try:
                from kava.vision.siglip2 import siglip2_preprocess
                pixel = siglip2_preprocess(img_path, self._siglip_proc)
            except Exception:
                return self.__getitem__((i + 1) % len(self.records))
        else:
            pixel = None
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "labels": torch.tensor(enc["labels"], dtype=torch.long),
            "pixel_values": pixel,
        }


def collate_vlm(batch, pad_id: int):
    """텍스트와 이미지 입력을 오른쪽 패딩해 배치로 구성."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    labels = torch.full((B, maxlen), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((B, maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attention_mask[i, :n] = 1

    out = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    pix = [b.get("pixel_values") for b in batch]
    ref = next((p for p in pix if p is not None), None)
    if ref is None:
        out["pixel_values"] = None # 순수 텍스트 배치
    elif isinstance(ref, dict):
        # SigLIP2의 전처리 출력: dict(pixel_values, pixel_attention_mask, spatial_shapes)
        bundles = [_fill(p, ref) for p in pix] # 입력값을 0으로 채워 반환
        out["pixel_values"] = torch.stack([bd["pixel_values"] for bd in bundles], dim=0)
        out["pixel_attention_mask"] = torch.stack([bd["pixel_attention_mask"] for bd in bundles], dim=0)
        out["spatial_shapes"] = torch.stack([bd["spatial_shapes"] for bd in bundles], dim=0)
    else:
        raise TypeError(
            f"SigLIP-2 NaFlex 번들(dict)을 기대했으나 {type(ref).__name__}을 받았습니다."
        )
    return out

def _fill(p, ref):
    if p is not None:
        return p
    return {
        "pixel_values": torch.zeros_like(ref["pixel_values"]),
        "pixel_attention_mask": ref["pixel_attention_mask"].clone(),
        "spatial_shapes": ref["spatial_shapes"].clone()
    }
        

def load_manifest(path: str) -> list[dict]:
    """jsonl manifest 로드(각 행 1개 레코드)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 없음. vlm_prepare.py로 manifest를 먼저 만드세요.")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
