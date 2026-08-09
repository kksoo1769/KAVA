"""LLaVA를 따른 VLM 데이터: (이미지 + 대화): EXAONE 챗 템플릿 인코딩 + 시각 자리표시."""

from __future__ import annotations

import json
import os

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# 특수 토큰
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
IMAGE_PLACEHOLDER = "<image>"
ENDOFTURN = "[|endofturn|]"

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

def _render(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
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
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = _render(tokenizer, messages[:i], add_generation_prompt=True)
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

def encode_vlm_reasoning(
    tokenizer, user_content: str, think: str, answer: str, max_len: int = 2048,
    add_default_system: bool = True, default_system: str | None = None,
    expect_image: bool = False,
) -> dict | None:
    """추론(CoT) 1턴: {"input_ids","labels"}. CoT + 답변 + 종료 토큰만 supervise."""
    messages = [{"role": "user", "content": user_content}]
    if add_default_system:
        messages = [{"role": "system", "content": default_system or SYS_VLM}, *messages]

    # EXAONE-4.0 템플릿으로 변환
    prefix_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    ) # [|user|]...[|endofturn|]\n[|assistant|]\n<think>\n
    if not prefix_text.endswith("<think>\n"):
        return None # Fallback: 예상 prefix 아님(템플릿 변경)
    body = f"{think.strip()}\n</think>\n\n{answer.strip()}{ENDOFTURN}\n" # [|user|]...[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\n[|answer|]...[|endofturn|]\n
    prefix_ids = tokenizer_image_token(prefix_text, tokenizer)
    full_ids = tokenizer_image_token(prefix_text + body, tokenizer)

    start = _common_prefix_len(prefix_ids, full_ids) # CoT 시작 위치
    labels = [full_ids[t] if t >= start else IGNORE_INDEX for t in range(len(full_ids))] # 추론, 답변, 종료 토큰을 제외한 나머지 masking
    if max_len is not None and len(full_ids) > max_len:
        full_ids, labels = full_ids[:max_len], labels[:max_len]
    if expect_image and IMAGE_TOKEN_INDEX not in full_ids:
        return None # 시각 추론인데 절단으로 이미지 자리 표시 소실: 드롭
    if all(l == IGNORE_INDEX for l in labels):
        return None # 전부 학습하지 않는 토큰: 드롭
    return {"input_ids": full_ids, "labels": labels}


def reasoning_parts(row: dict, has_image: bool = False):
    """reasoning 행에서 (user, think, answer) 추출. has_image면 user에 <image> 1개 주입. 실패 시 None."""
    user = think = answer = None
    for t in row.get("conversations", []):
        r = str(t.get("from", "")).lower()
        if r in ("human", "user") and user is None:
            user = (t.get("value") or "").replace(IMAGE_PLACEHOLDER, "").strip()
        elif r in ("gpt", "assistant"):
            answer = (t.get("value") or "").strip()
            think = (t.get("think") or "").strip()
    if user and think and answer:
        if has_image: # 시각 추론: user 앞에 <image> 마커 1개
            user = f"{IMAGE_PLACEHOLDER}\n{user}".strip()
        return user, think, answer
    return None


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

    첫 human 턴에 <image> 마커가 없으면 자동 주입. value 안의 LLaVA <image> 마커는 그대로 사용.
    """
    role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                "system": "system"}
    has_image = bool(row.get("image"))
    msgs, injected = [], False
    for t in row.get("conversations", []):
        role = role_map.get(str(t.get("from", "")).lower())
        if role is None:
            continue
        content = (t.get("value") or "").replace(IMAGE_PLACEHOLDER, "").strip()
        if has_image and role == "user" and not injected:
            content = f"{IMAGE_PLACEHOLDER}\n{content}".strip()
            injected = True
        msgs.append({"role": role, "content": content})
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
        max_num_patches: int = 576
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
        if row.get("reasoning"): # CoT 추론 행 확인: think supervise 경로
            has_img = bool(row.get("image"))
            parts = reasoning_parts(row, has_image=has_img)
            enc = None if parts is None else encode_vlm_reasoning(
                self.tokenizer, parts[0], parts[1], parts[2], max_len=self.max_len,
                add_default_system=self.add_default_system, expect_image=has_img
            )
        else:
            messages = row_to_messages(row, self.stage, rng_idx=i)
            enc = encode_vlm_conversation(
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


def build_vlm_dataloader(records, tokenizer_name, pad_id, stage="align", batch_size=8,
                         img_size=224, max_len=2048, rank=0, world_size=1, num_workers=8,
                         seed=42, shuffle=True, train_aug=False, add_default_system=True,
                         vision_backbone="siglip2",
                         siglip_model_id="google/siglip2-so400m-patch16-naflex",
                         max_num_patches=576):
    ds = VLMDataset(records, tokenizer_name, stage=stage, img_size=img_size, max_len=max_len,
                    train_aug=train_aug, add_default_system=add_default_system,
                    vision_backbone=vision_backbone, siglip_model_id=siglip_model_id,
                    max_num_patches=max_num_patches)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                 shuffle=shuffle, seed=seed, drop_last=True)
    loader = DataLoader(
        ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0),
        collate_fn=lambda b: collate_vlm(b, pad_id),
    )
    return loader, sampler
