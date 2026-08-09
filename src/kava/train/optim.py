"""Set AdamW with no decay param groups + warmup + cosine lr schedule"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, lr: float, wd: float, betas: tuple[float, float]):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            # bias, norm 파라미터 weight decay 제외
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.}
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=True)

def make_lr_scheduler(optimizer, warmup_steps: int, max_steps: int, min_ratio: float):
    def lr_lambda(step: int):
        if step < warmup_steps: # warmup
            return (step + 1) / max(1, warmup_steps) # 선형 증가
        if step >= max_steps:
            return min_ratio
        prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_ratio + .5 * (1 - min_ratio) * (1 + math.cos(math.pi * prog)) # cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
