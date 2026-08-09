"""실행 환경에 맞는 디바이스와 attention 구현을 선택."""

from __future__ import annotations

import os

import torch


def resolve_device(preferred: str | None = None) -> str:
    """설정값과 KLAVA_DEVICE 환경 변수를 기준으로 디바이스를 선택."""
    req = preferred or os.environ.get("KLAVA_DEVICE") or "auto"
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_cuda(device: str | torch.device | None = None) -> bool:
    return str(resolve_device(str(device) if device is not None else None)).startswith("cuda")


def resolve_attn_implementation(
    requested: str = "flash_attention_2", device: str | None = None
) -> str:
    """CUDA에서 flash attention을 사용할 수 없으면 sdpa를 선택."""
    if requested != "flash_attention_2":
        return requested
    if not is_cuda(device):
        return "sdpa"
    try:
        import flash_attn
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


def synchronize(device: str | None = None) -> None:
    """디바이스 비동기 큐를 비운다. CPU에서는 할 일이 없다."""
    dev = resolve_device(device)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def max_memory_allocated_gib(device: str | None = None) -> float:
    """로그용 피크 메모리(GiB). MPS는 driver allocated, CPU는 0.0."""
    dev = resolve_device(device)
    if dev.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    if dev == "mps":
        return torch.mps.driver_allocated_memory() / (1024 ** 3)
    return 0.0
