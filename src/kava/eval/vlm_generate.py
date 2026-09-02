"""학습된 KLaVA로 이미지 설명과 답변을 생성."""
from __future__ import annotations

import argparse
import glob as _glob
import os

import torch
from safetensors.torch import load_file
from kava.ckpt import load_meta
from kava.device import resolve_device
from kava.model_vlm import build_vlm, IMAGE_TOKEN_INDEX
from kava.vision.siglip2 import Siglip2Config, build_siglip2_processor, siglip2_preprocess
from kava.data.tokenizer import load_tokenizer
from kava.data.vlm import tokenizer_image_token, IMAGE_PLACEHOLDER, SYS_VLM


def load_vlm_for_inference(ckpt_dir: str, device=None, dtype=torch.bfloat16, verbose=True):
    """meta.json 기반으로 KLaVA를 추론용으로 복원. (vlm, meta) 반환.

    device=None이면 cuda: mps: cpu 순으로 자동 선택한다(KLAVA_DEVICE로 override).
    """
    device = resolve_device(device)
    meta = load_meta(ckpt_dir)
    has_adapter = bool(meta.get("lora")) and os.path.isdir(os.path.join(ckpt_dir, "adapter"))
    ft_vision = os.path.exists(os.path.join(ckpt_dir, "vision_encoder.safetensors"))

    base_id = meta["exaone_id"]

    # SigLIP-2 NaFlex: HF 사전학습 로드(미세조정본 있으면 아래서 덮어씀).
    vlm = build_vlm(
        exaone_id=base_id,
        vision_backbone="siglip2",
        siglip_cfg=Siglip2Config(
            model_id=meta["siglip_model_id"],
            max_num_patches=int(meta["siglip_num_patches"])),
        lm_master_dtype=dtype, vis_master_dtype=dtype, con_master_dtype=dtype,
        device=device, attn_implementation="sdpa", verbose=verbose,
    )
    # projector
    proj = load_file(os.path.join(ckpt_dir, "projector.safetensors"))
    vlm.connector.load_state_dict({k: v for k, v in proj.items()}, strict=True)
    # fine-tuned vision 백본
    if ft_vision:
        vis = load_file(os.path.join(ckpt_dir, "vision_encoder.safetensors"))
        vlm.vision_encoder.backbone.load_state_dict({k: v for k, v in vis.items()}, strict=True)
    # LoRA adapter
    if has_adapter:
        from peft import PeftModel
        vlm.language_model = PeftModel.from_pretrained(
            vlm.language_model, os.path.join(ckpt_dir, "adapter")).to(device)
    elif meta.get("lm_mode") == "full":
        lm = load_file(os.path.join(ckpt_dir, "language_model.safetensors"))
        vlm.language_model.load_state_dict({k: v for k, v in lm.items()}, strict=True)
    vlm.eval()
    if verbose:
        print(f"[infer] ckpt={ckpt_dir} backbone={meta['vision_backbone']} device={device} "
              f"seq_len={meta['siglip_num_patches']} "
              f"adapter={has_adapter} ft_vision={ft_vision}", flush=True)
    return vlm, meta


def _prepare_image_inputs(image_path, siglip_proc, device):
    """이미지 1장: encode_images에 넘길 kwargs.

    SigLIP-2 NaFlex: {"pixel_values": (1,N,768), "pixel_attention_mask": (1,N), "spatial_shapes": (1,2)}.
    """
    bundle = siglip2_preprocess(image_path, siglip_proc)
    return {
        "pixel_values": bundle["pixel_values"].unsqueeze(0).to(device),
        "pixel_attention_mask": bundle["pixel_attention_mask"].unsqueeze(0).to(device),
        "spatial_shapes": bundle["spatial_shapes"].unsqueeze(0).to(device),
    }


def build_vlm_messages(prompt: str, history: list[dict] | None = None) -> list[dict]:
    """이미지 마커를 유지하며 KLaVA 추론용 대화 메시지를 생성."""
    if not isinstance(prompt, str):
        raise ValueError("prompt는 문자열이어야 합니다.")
    clean_prompt = prompt.replace(IMAGE_PLACEHOLDER, "").strip()
    if not clean_prompt:
        raise ValueError("prompt는 비어 있을 수 없습니다.")

    clean_history: list[dict] = []
    expected_role = "user"
    for index, message in enumerate(history or []):
        if not isinstance(message, dict):
            raise ValueError(f"history[{index}]는 메시지 객체여야 합니다.")

        role = message.get("role")
        if role not in ("user", "assistant"):
            raise ValueError(
                f"history[{index}].role은 user 또는 assistant여야 합니다: {role!r}"
            )
        if role != expected_role:
            raise ValueError(
                f"history[{index}].role은 {expected_role}여야 합니다: {role!r}"
            )

        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"history[{index}].content는 문자열이어야 합니다.")
        content = content.replace(IMAGE_PLACEHOLDER, "").strip()
        if not content:
            raise ValueError(f"history[{index}].content는 비어 있을 수 없습니다.")

        clean_history.append({"role": role, "content": content})
        expected_role = "assistant" if role == "user" else "user"

    if expected_role == "assistant":
        raise ValueError("history는 assistant 응답까지 포함한 완결된 턴이어야 합니다.")

    messages = [
        {"role": "system", "content": SYS_VLM},
        *clean_history,
        {"role": "user", "content": clean_prompt},
    ]
    first_user = next(message for message in messages if message["role"] == "user")
    first_user["content"] = f"{IMAGE_PLACEHOLDER}\n{first_user['content']}"
    return messages


@torch.inference_mode()
def generate_text(
    vlm, tokenizer, image_path, prompt,
    siglip_proc=None, max_new_tokens=128, temperature=0.1, enable_thinking=False,
    device=None, history: list[dict] | None = None,
):
    """이미지 1장과 완결된 대화 history로 다음 assistant 응답을 생성한다."""
    device = resolve_device(device)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens는 1 이상이어야 합니다.")
        
    # 텍스트 인코딩(EXAONE chat 템플릿 + <image>: IMAGE_TOKEN_INDEX)
    messages = build_vlm_messages(prompt, history=history)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    ids = tokenizer_image_token(text, tokenizer)
    image_token_count = ids.count(IMAGE_TOKEN_INDEX)
    if image_token_count != 1:
        raise RuntimeError(
            f"KLaVA 프롬프트에는 이미지 토큰이 1개여야 합니다: {image_token_count}개"
        )
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)

    # 이미지: encode_images kwargs
    image_inputs = _prepare_image_inputs(image_path, siglip_proc, device)

    # 학습과 같은 encode_images와 _merge를 사용한다
    image_embeds, img_token_mask = vlm.encode_images(**image_inputs)
    inputs_embeds, _, keep = vlm._merge(input_ids, attn, None, image_embeds, img_token_mask)
    position_ids = (keep.cumsum(-1) - 1).clamp(min=0)

    lm = vlm.language_model
    prompt_tokens = int(keep[0].sum().item())
    context_limit = getattr(lm.config, "max_position_embeddings", None)
    if context_limit is not None:
        remaining = int(context_limit) - prompt_tokens
        if remaining < 1:
            raise ValueError(
                f"멀티 턴 입력이 모델 컨텍스트 한도({context_limit})를 초과했습니다. "
                "오래된 대화 턴을 줄여 주세요."
            )
        max_new_tokens = min(max_new_tokens, remaining)

    gen = lm.generate(
        inputs_embeds=inputs_embeds, attention_mask=keep, position_ids=position_ids,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0), temperature=(temperature or None),
        top_p=0.95 if temperature > 0 else None,
        eos_token_id=tokenizer.convert_tokens_to_ids("[|endofturn|]"),
        pad_token_id=tokenizer.pad_token_id,
    )
    # inputs_embeds 경로의 generate는 새 토큰만 반환한다.
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    return out.strip()


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def collect_images(image, image_dir):
    """파일이나 글롭으로 받은 이미지를 디렉터리 입력과 합쳐 정렬."""
    paths: list[str] = []
    for it in (image or []):
        if any(ch in it for ch in "*?["):
            paths.extend(_glob.glob(it))
        else:
            paths.append(it)
    if image_dir:
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith(_IMG_EXTS):
                paths.append(os.path.join(image_dir, f))
    # 중복 제거 + 파일명 자연 정렬(img2 < img10)
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); uniq.append(p)

    import re
    def natkey(p):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r"(\d+)", os.path.basename(p))]
    return sorted(uniq, key=natkey)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", action="append")
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--prompt", default="이 이미지를 한국어로 설명해 주세요.")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--enable_thinking", action="store_true", default=False)
    args = ap.parse_args()

    images = collect_images(args.image, args.image_dir)
    if not images:
        raise SystemExit("이미지가 없습니다: --image 또는 --image_dir 를 확인하세요.")

    device = resolve_device()
    tokenizer = load_tokenizer()
    vlm, meta = load_vlm_for_inference(args.ckpt, device=device) # 모델은 1회만 로드
    siglip_proc = build_siglip2_processor(
        meta["siglip_model_id"],
        max_num_patches=int(meta["siglip_num_patches"]))
    print(f"\n[batch] {len(images)}개 이미지 순차 처리 (device={device}, prompt='{args.prompt}')\n" + "=" * 60)
    for i, img in enumerate(images, 1):
        try:
            cap = generate_text(
                vlm, tokenizer, img, args.prompt,
                siglip_proc=siglip_proc, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, enable_thinking=args.enable_thinking,
                device=device,
            )
        except Exception as e:
            cap = f"<생성 실패: {e}>"
        print(f"[{i:2d}/{len(images)}] {os.path.basename(img)}\n       : {cap}", flush=True)
    print("=" * 60)


if __name__ == "__main__":
    main()
