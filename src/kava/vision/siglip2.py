"""SigLIP-2 NaFlex 비전 타워"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from PIL import Image

import torch
import torch.nn as nn

from transformers import Siglip2VisionModel, Siglip2ImageProcessor


DEFAULT_SIGLIP2_ID = "google/siglip2-so400m-patch16-naflex"


@dataclass
class Siglip2Config:
    """SigLIP-2 NaFlex 타워 설정."""
    model_id: str = DEFAULT_SIGLIP2_ID
    max_num_patches: int = 784 # 시퀀스 길이 기본값
    patch_size: int = 16
    attn_implementation: str = "sdpa" # Siglip2는 Flash-attn 사용 불가로 sdpa만 사용

    @property
    def pixel_dim(self) -> int:
        return 3 * self.patch_size * self.patch_size


class Siglip2VisionTower(nn.Module):
    """HF Siglip2VisionModel 래퍼(패치 토큰 시퀀스만 추출)."""

    def __init__(self, cfg: Siglip2Config, backbone: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone

    @property
    def hidden_size(self) -> int:
        return self.backbone.config.hidden_size

    @property
    def num_patches(self) -> int:
        return self.cfg.max_num_patches

    def encode(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        frozen: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """패치 입력을 시각 토큰과 유효 토큰 마스크로 변환."""
        assert pixel_attention_mask is not None and spatial_shapes is not None, (
            "SigLIP2 NaFlex는 pixel_attention_mask와 spatial_shapes가 필요합니다."
        )
        ctx = torch.no_grad() if frozen else contextlib.nullcontext()
        with ctx:
            out = self.backbone(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
        tokens = out.last_hidden_state # (B, P, C): post_layernorm 적용된 패치 토큰
        if frozen:
            tokens = tokens.detach()
        return tokens, pixel_attention_mask.bool()

    def make_dummy_inputs(self, device, batch_size: int = 1) -> dict:
        """이미지가 사용 없이 텍스트만 사용하는 경우를 위한 더미 입력"""
        P = self.cfg.max_num_patches
        side = max(1, int(P ** 0.5))
        h = w = side
        pv = torch.zeros(batch_size, P, self.cfg.pixel_dim, device=device, dtype=torch.float32)
        pam = torch.zeros(batch_size, P, dtype=torch.long, device=device)
        pam[:, : h * w] = 1
        ss = torch.tensor([[h, w]], device=device, dtype=torch.long).expand(batch_size, 2).contiguous()
        return {"pixel_values": pv, "pixel_attention_mask": pam, "spatial_shapes": ss}


def build_siglip2_tower(
    cfg: Siglip2Config | None = None,
    device="cuda",
    dtype=torch.float32,
    local_files_only: bool = True,
    verbose: bool = True,
) -> Siglip2VisionTower:
    """Siglip2VisionModel(head 제외)을 로드해 Siglip2VisionTower로 감싼다."""

    cfg = cfg or Siglip2Config()
    backbone = Siglip2VisionModel.from_pretrained(
        cfg.model_id,
        local_files_only=local_files_only,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=dtype,
    )
    # head 제거
    backbone.use_head = False
    if hasattr(backbone, "head"):
        del backbone.head
    backbone = backbone.to(device=device, dtype=dtype)
    tower = Siglip2VisionTower(cfg, backbone)
    if verbose:
        print(
            f"[vision] SigLIP-2 tower loaded: {cfg.model_id} | hidden={tower.hidden_size} "
            f"seq_len(max_num_patches)={cfg.max_num_patches} attn={cfg.attn_implementation}",
            flush=True,
        )
    return tower

def build_siglip2_processor(
    model_id: str = DEFAULT_SIGLIP2_ID,
    max_num_patches: int = 576,
    local_files_only: bool = True,
):
    """Siglip2 전용 전처리 로드. max_num_patches로 최대 패치 수를 고정한다."""
    return Siglip2ImageProcessor.from_pretrained(
        model_id, max_num_patches=max_num_patches, local_files_only=local_files_only
    )


def siglip2_preprocess(path_or_image, processor) -> dict:
    """이미지(경로 또는 PIL): NaFlex dict."""
    if isinstance(path_or_image, str):
        with Image.open(path_or_image) as im:
            bf = processor(images=im.convert("RGB"), return_tensors="pt")
    else:
        bf = processor(images=path_or_image.convert("RGB"), return_tensors="pt")
    return {
        "pixel_values": bf["pixel_values"][0], # (P, 768(=3 x 16 x 16)) float
        "pixel_attention_mask": bf["pixel_attention_mask"][0].long(), # (P,)
        "spatial_shapes": torch.as_tensor(bf["spatial_shapes"][0]).long(), # (2,)
    }
