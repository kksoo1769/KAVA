"""KLaVA builder"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from kava.device import resolve_attn_implementation, resolve_device
from kava.vision.siglip2 import Siglip2Config, build_siglip2_tower, Siglip2VisionTower
from kava.data.vlm import IGNORE_INDEX, IMAGE_TOKEN_INDEX


EXAONE_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"


def build_connector(vision_hidden: int, lm_hidden: int) -> nn.Sequential:
    # MLP만 사용
    return nn.Sequential(
        nn.Linear(vision_hidden, lm_hidden),
        nn.GELU(),
        nn.Linear(lm_hidden, lm_hidden)
    )


class KLaVA(nn.Module):
    """비전 인코더 + MLP 커넥터 + EXAONE-4.0-1.2B LM."""

    def __init__(
        self,
        language_model: nn.Module, # Exaone4ForCausalLM 또는 PEFT 모델
        vision_encoder: Siglip2VisionTower,
        lm_hidden: int,
    ):
        super().__init__()
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        self.connector = build_connector(vision_encoder.hidden_size, lm_hidden)
        self.lm_hidden = lm_hidden
        self.train_vision = False # Stage 1: freeze, Stage 2: unfreeze
        
        # Freeze
        for p in self.vision_encoder.parameters():
            p.requires_grad_(False)
        self.vision_encoder.eval()

    def set_trainable(
        self,
        train_lm: bool | None = False,
        train_projector: bool = True,
        train_vision: bool = False
    ):
        """학습 단계에 맞춰 모델 구성 요소를 활성화."""
        self.train_vision = train_vision
        # 학습 단계에 맞춰 사전학습 백본의 고정 여부를 바꾼다
        for p in self.vision_encoder.backbone.parameters():
            p.requires_grad_(train_vision)
        if train_vision:
            self.vision_encoder.train()
        else:
            self.vision_encoder.eval()
        for p in self.connector.parameters():
            p.requires_grad_(train_projector)
        if train_lm is not None:
            for p in self.language_model.parameters():
                p.requires_grad_(train_lm)

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """이미지를 시각 토큰 임베딩과 토큰 마스크로 변환."""
        device = pixel_values.device.type
        with torch.autocast(
            device_type=device,
            dtype=torch.bfloat16,
        ):
            feats, token_mask = self.vision_encoder.encode(
                pixel_values, pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes, frozen=not self.train_vision,
            ) # feats: (B, P, Cv)
        P = feats.shape[1]
        assert P == self.vision_encoder.num_patches, (
            f"vision tokens {P} != expected {self.vision_encoder.num_patches}"
        )
        feats = feats.to(self.connector[0].weight.dtype)
        return self.connector(feats), token_mask # (B, P, D), (B, P) | None

    def _merge(self, input_ids, attention_mask, labels, image_embeds, image_token_mask=None):
        """오른쪽 패딩을 제외한 텍스트에 유효한 시각 토큰을 삽입."""
        B = input_ids.shape[0]
        D = self.lm_hidden
        device = input_ids.device
        embed_tokens = self.language_model.get_input_embeddings()
        emb_dtype = image_embeds.dtype if image_embeds is not None else embed_tokens.weight.dtype

        merged_embeds: list[torch.Tensor] = []
        merged_labels: list[torch.Tensor | None] = []
        lens: list[int] = []
        for b in range(B):
            real = int(attention_mask[b].sum().item()) # 우패딩이라 앞 real개가 실제 토큰
            ids = input_ids[b, :real]
            lab = labels[b, :real] if labels is not None else None
            img_pos = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if img_pos.numel() == 0 or image_embeds is None:
                # 이미지 없는(순수 텍스트) 행: 그대로 임베딩.
                cur_emb = embed_tokens(ids.clamp_min(0)).to(emb_dtype)
                cur_lab = lab
            else:
                p = int(img_pos[0])
                # 백본에 맞춰 유효한 시각 토큰만 선택한다
                img_toks = (image_embeds[b][image_token_mask[b].bool()]
                            if image_token_mask is not None else image_embeds[b])
                Pn = img_toks.shape[0]
                pre_emb = embed_tokens(ids[:p].clamp_min(0)).to(emb_dtype)
                post_emb = embed_tokens(ids[p + 1:].clamp_min(0)).to(emb_dtype)
                cur_emb = torch.cat([pre_emb, img_toks, post_emb], dim=0)
                if lab is not None:
                    img_lab = torch.full((Pn,), IGNORE_INDEX, dtype=lab.dtype, device=device)
                    cur_lab = torch.cat([lab[:p], img_lab, lab[p + 1:]], dim=0)
                else:
                    cur_lab = None
            merged_embeds.append(cur_emb)
            merged_labels.append(cur_lab)
            lens.append(cur_emb.shape[0])

        L = max(lens)
        out_emb = torch.zeros(B, L, D, device=device, dtype=emb_dtype)
        out_keep = torch.zeros(B, L, dtype=torch.long, device=device)
        out_lab = (torch.full((B, L), IGNORE_INDEX, device=device, dtype=torch.long)
                   if labels is not None else None)
        for b in range(B):
            n = lens[b]
            out_emb[b, :n] = merged_embeds[b]
            out_keep[b, :n] = 1
            if out_lab is not None and merged_labels[b] is not None:
                out_lab[b, :n] = merged_labels[b]
        return out_emb, out_lab, out_keep

    def forward(self, input_ids, attention_mask, labels=None, pixel_values=None,
                pixel_attention_mask=None, spatial_shapes=None):
        """텍스트와 이미지 입력으로 언어 모델의 출력을 계산."""
        if pixel_values is not None:
            image_embeds, img_token_mask = self.encode_images(
                pixel_values, pixel_attention_mask, spatial_shapes)
        else:
            image_embeds, img_token_mask = None, None
        inputs_embeds, merged_labels, keep = self._merge(
            input_ids, attention_mask, labels, image_embeds, img_token_mask)

        # EXAONE는 padding을 무시하도록 attention_mask가, RoPE 정합을 위해 position_ids가 필수.
        attn = keep # (B, L) long {0,1}
        position_ids = (attn.cumsum(dim=-1) - 1).clamp(min=0)

        out = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            position_ids=position_ids,
            labels=merged_labels,
            use_cache=False,
        )
        loss = out.loss
        # 텍스트 전용 배치에도 비전 파라미터를 연결해 DDP의 파라미터 집합을 맞춘다
        if loss is not None:
            if image_embeds is None:
                # NaFlex 더미 입력으로 비전 파라미터를 그래프에 연결한다
                dummy = self.vision_encoder.make_dummy_inputs(input_ids.device)
                image_embeds, _ = self.encode_images(**dummy)
            loss = loss + 0.0 * image_embeds.float().sum()
        return {"loss": loss, "logits": out.logits}


def build_vlm(
    exaone_id: str = EXAONE_ID,
    vision_backbone: str = "siglip2",
    siglip_cfg: Siglip2Config | None = None,
    lora_config=None,
    lm_master_dtype=torch.float32,
    vis_master_dtype=torch.float32,
    con_master_dtype=torch.float32,
    device: str | None = None,
    local_files_only: bool = True,
    attn_implementation: str = "flash_attention_2",
    verbose: bool = True,
) -> KLaVA:
    """EXAONE-4.0-1.2B LM + SigLIP-2 Vision tower + MLP Connector로 KLaVA 생성."""
    if vision_backbone != "siglip2":
        raise ValueError(
            f"지원하지 않는 vision_backbone: {vision_backbone!r}. siglip2만 지원합니다."
        )
    device = resolve_device(device)
    attn_implementation = resolve_attn_implementation(attn_implementation, device)
    lm = AutoModelForCausalLM.from_pretrained(
        exaone_id, torch_dtype=lm_master_dtype, local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    lm_hidden = lm.config.hidden_size # EXAONE-4.0-1.2B: 2048
    if lora_config is not None:
        from peft import get_peft_model
        lm = get_peft_model(lm, lora_config)
        if verbose:
            lm.print_trainable_parameters()
    lm = lm.to(device=device)

    ve = build_siglip2_tower(
        cfg=siglip_cfg, device=device, dtype=vis_master_dtype,
        local_files_only=local_files_only, verbose=verbose,
    )
    vlm = KLaVA(lm, ve, lm_hidden=lm_hidden)
    vlm.connector = vlm.connector.to(device=device, dtype=con_master_dtype)
    return vlm
