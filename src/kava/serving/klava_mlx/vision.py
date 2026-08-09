"""KLaVA 체크포인트를 사용하는 MLX SigLIP-2 비전 타워."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .preprocess import aa_bilinear_weights

__all__ = [
    "Siglip2VisionConfig",
    "Siglip2VisionModel",
    "sanitize",
    "read_safetensors_header",
    "reconcile_keys",
    "load_vision_tower",
]


# 설정
@dataclass
class Siglip2VisionConfig:
    """MLX 비전 타워에서 사용하는 SigLIP2 설정."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    num_patches: int = 256 # 위치 임베딩 그리드 크기(16x16). 시퀀스 길이가 아니다.

    # 시퀀스 길이(= 전처리의 max_num_patches). meta.json 의 siglip_num_patches.
    max_num_patches: int = 784

    @property
    def pixel_dim(self) -> int:
        return self.num_channels * self.patch_size * self.patch_size

    @property
    def position_embedding_size(self) -> int:
        return int(self.num_patches**0.5)

    @classmethod
    def from_meta(cls, meta: dict, hf_config: Optional[dict] = None) -> "Siglip2VisionConfig":
        """체크포인트 메타데이터와 HF 설정에서 값을 읽어 생성."""
        vc = dict((hf_config or {}).get("vision_config") or {})
        kwargs: dict[str, Any] = {}
        for name in ("hidden_size", "intermediate_size", "num_hidden_layers",
                     "num_attention_heads", "num_channels", "patch_size",
                     "hidden_act", "layer_norm_eps", "num_patches"):
            if name in vc and vc[name] is not None:
                kwargs[name] = vc[name]
        if meta.get("vision_hidden"):
            kwargs.setdefault("hidden_size", int(meta["vision_hidden"]))
        kwargs["max_num_patches"] = int(meta.get("siglip_num_patches", 784))
        return cls(**kwargs)


def _activation(name: str):
    """설정 이름에 맞는 활성화 함수를 반환."""
    if name in ("gelu_pytorch_tanh", "gelu_new", "gelu_tanh"):
        return nn.gelu_approx
    if name == "gelu":
        return nn.gelu
    if name in ("silu", "swish"):
        return nn.silu
    if name == "relu":
        return nn.relu
    raise ValueError(f"지원하지 않는 hidden_act: {name!r}: 추측하지 않는다.")


# 위치 임베딩 크기를 조정한다
_RESIZE_CACHE: dict[tuple[int, int], mx.array] = {}


def _resize_matrix(in_size: int, out_size: int) -> mx.array:
    """float32 antialias bilinear 리샘플 행렬을 생성."""
    key = (in_size, out_size)
    cached = _RESIZE_CACHE.get(key)
    if cached is not None:
        return cached
    starts, weights = aa_bilinear_weights(in_size, out_size)
    dense = np.zeros((out_size, in_size), dtype=np.float32)
    k = weights.shape[1]
    for i in range(out_size):
        for j in range(k):
            w = weights[i, j]
            if w == 0.0:
                continue
            col = starts[i] + j
            if 0 <= col < in_size:
                dense[i, col] += w
    arr = mx.array(dense)
    _RESIZE_CACHE[key] = arr
    return arr


def resize_positional_embeddings(
    positional_embeddings: mx.array, # (grid, grid, D) float32
    spatial_shapes: list[tuple[int, int]],
    max_length: int,
    dtype: mx.Dtype,
) -> mx.array:
    """공간 크기에 맞게 위치 임베딩을 리사이즈."""
    grid_h, grid_w, dim = positional_embeddings.shape
    out = []
    for (h, w) in spatial_shapes:
        if h <= 0 or w <= 0:
            raise ValueError(f"spatial_shapes 가 양수가 아니다: {(h, w)}")
        if h * w > max_length:
            raise ValueError(f"h*w({h * w}) > max_length({max_length})")
        x = positional_embeddings # (gh, gw, D) float32
        if h != grid_h:
            wh = _resize_matrix(grid_h, h) # (h, gh)
            x = mx.einsum("hg,gwd->hwd", wh, x)
        if w != grid_w:
            ww = _resize_matrix(grid_w, w) # (w, gw)
            x = mx.einsum("wg,hgd->hwd", ww, x)
        flat = x.reshape(h * w, dim).astype(dtype) # :182,:185
        if h * w < max_length:
            pad = mx.broadcast_to(flat[0:1], (max_length - h * w, dim)) # :188
            flat = mx.concatenate([flat, pad], axis=0)
        out.append(flat)
    return mx.stack(out, axis=0)


# 모듈
class Siglip2VisionEmbeddings(nn.Module):
    """SigLIP2 임베딩 계층."""

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Linear(config.pixel_dim, config.hidden_size, bias=True)
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)

    def __call__(self, pixel_values: mx.array, spatial_shapes: list[tuple[int, int]]) -> mx.array:
        # :202-203 patch_embedding 의 dtype 으로 캐스팅 후 Linear.
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.astype(target_dtype))

        grid = self.config.position_embedding_size
        # :206-208 (num_patches, D): (grid, grid, D). float32 로 올려서 리사이즈(:164-165).
        pos = self.position_embedding.weight.reshape(grid, grid, -1).astype(mx.float32)
        resized = resize_positional_embeddings(
            pos, spatial_shapes, max_length=pixel_values.shape[1], dtype=target_dtype
        )
        return patch_embeds + resized # :214


class Siglip2Attention(nn.Module):
    """분리된 선형 계층으로 구성한 attention."""

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim({self.embed_dim}) 이 num_heads({self.num_heads}) 로 "
                "나누어떨어지지 않는다."
            )
        self.scale = self.head_dim**-0.5 # :295
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        # :315-317 (B,L,D): (B,L,H,hd): (B,H,L,hd)
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1) # :334
        return self.out_proj(o)


class Siglip2MLP(nn.Module):
    """SigLIP2 MLP 계층."""

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.activation_fn = _activation(config.hidden_act)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.activation_fn(self.fc1(x)))


class Siglip2EncoderLayer(nn.Module):
    """pre-LN 구조의 SigLIP2 encoder layer."""

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask) # :371-379
        x = x + self.mlp(self.layer_norm2(x)) # :381-384
        return x


class Siglip2Encoder(nn.Module):
    """SigLIP2 인코더 계층."""

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.layers = [Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Siglip2VisionModel(nn.Module):
    """분류 head를 제외한 SigLIP2 비전 모델."""

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @staticmethod
    def build_attention_mask(
        pixel_attention_mask: mx.array, dtype: mx.Dtype
    ) -> mx.array:
        """패치 마스크를 attention에 사용할 가산 마스크로 변환."""
        neg = mx.finfo(dtype).min
        keep = pixel_attention_mask.astype(mx.bool_)
        m = mx.where(keep, mx.array(0.0, dtype=dtype), mx.array(neg, dtype=dtype))
        return m[:, None, None, :]

    def __call__(
        self,
        pixel_values: mx.array, # (B, P, pixel_dim)
        pixel_attention_mask: mx.array, # (B, P) {0,1}
        spatial_shapes: Any, # (B, 2) array 또는 [(h,w), ...]
    ) -> mx.array:
        shapes = _as_shape_list(spatial_shapes)
        if len(shapes) != pixel_values.shape[0]:
            raise ValueError(
                f"spatial_shapes 개수({len(shapes)})가 배치({pixel_values.shape[0]})와 다르다"
            )
        h = self.embeddings(pixel_values, shapes) # :561
        mask = self.build_attention_mask(pixel_attention_mask, h.dtype) # :563-567
        h = self.encoder(h, mask) # :569-573
        return self.post_layernorm(h) # :576


def _as_shape_list(spatial_shapes: Any) -> list[tuple[int, int]]:
    if isinstance(spatial_shapes, mx.array):
        spatial_shapes = np.array(spatial_shapes.tolist())
    arr = np.asarray(spatial_shapes)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"spatial_shapes 모양이 (B,2) 가 아니다: {arr.shape}")
    return [(int(a), int(b)) for a, b in arr]


# 저장 방식에 따라 붙을 수 있는 체크포인트 접두사
_STRIP_PREFIXES = ("vision_model.", "backbone.", "vision_encoder.backbone.",
                   "vision_encoder.", "model.vision_model.")

# 이 타워에서 사용하지 않는 키
_KNOWN_DROP_PREFIXES = ("head.",)


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """텐서 데이터를 불러오지 않고 safetensors 헤더만 읽는다."""
    p = Path(path)
    with p.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise ValueError(f"safetensors 헤더를 읽을 수 없다: {p}")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 256 * 1024 * 1024:
            raise ValueError(f"safetensors 헤더 길이가 이상하다({n}): {p}")
        return json.loads(fh.read(n).decode("utf-8"))


def sanitize(keys_or_weights: Any) -> tuple[dict[str, str], list[str]]:
    """체크포인트 키를 MLX 모듈의 파라미터 경로로 변환."""
    if isinstance(keys_or_weights, dict):
        keys = [k for k in keys_or_weights if k != "__metadata__"]
    else:
        keys = [k for k in keys_or_weights if k != "__metadata__"]

    mapping: dict[str, str] = {}
    dropped: list[str] = []
    for key in keys:
        name = key
        for pref in _STRIP_PREFIXES:
            if name.startswith(pref):
                name = name[len(pref):]
                break
        if any(name.startswith(p) for p in _KNOWN_DROP_PREFIXES):
            dropped.append(key)
            continue
        mapping[key] = name
    return mapping, dropped


@dataclass
class Reconciliation:
    """키 대조 결과. 사람이 읽는 보고서 + 기계가 읽는 판정."""

    matched: dict[str, str] = field(default_factory=dict) # ckpt key: module path
    dropped: list[str] = field(default_factory=list) # 의도적으로 버림
    unmapped: list[str] = field(default_factory=list) # 체크포인트에만 있음
    missing: list[str] = field(default_factory=list) # 모듈에만 있음
    shape_mismatch: list[tuple[str, tuple, tuple]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unmapped or self.missing or self.shape_mismatch)

    def report(self, limit: int = 20) -> str:
        lines = [
            f"매칭  {len(self.matched)}개",
            f"버림  {len(self.dropped)}개 (의도적)",
            f"미매핑{len(self.unmapped)}개 (체크포인트에만 있음)",
            f"미충족{len(self.missing)}개 (모듈 파라미터가 안 채워짐)",
            f"shape 불일치 {len(self.shape_mismatch)}개",
        ]
        for name, items in (("미매핑", self.unmapped), ("미충족", self.missing)):
            for k in items[:limit]:
                lines.append(f"  [{name}] {k}")
            if len(items) > limit:
                lines.append(f"  [{name}] ... 외 {len(items) - limit}개")
        for path, want, got in self.shape_mismatch[:limit]:
            lines.append(f"  [shape] {path}: 모듈 {want} != 체크포인트 {got}")
        return "\n".join(lines)


def reconcile_keys(
    header_or_weights: dict[str, Any], model: Siglip2VisionModel
) -> Reconciliation:
    """체크포인트와 모듈의 누락된 키를 양쪽에서 확인."""
    mapping, dropped = sanitize(header_or_weights)

    # 모듈 쪽 파라미터 경로: shape
    module_shapes: dict[str, tuple[int, ...]] = {
        k: tuple(v.shape) for k, v in tree_flatten(model.parameters())
    }

    # 체크포인트 형식에 맞춰 shape을 읽는다
    def _ckpt_shape(key: str) -> Optional[tuple[int, ...]]:
        v = header_or_weights[key]
        if isinstance(v, dict) and "shape" in v:
            return tuple(int(x) for x in v["shape"])
        shape = getattr(v, "shape", None)
        return tuple(int(x) for x in shape) if shape is not None else None

    rec = Reconciliation(dropped=dropped)
    for ckpt_key, path in mapping.items():
        if path not in module_shapes:
            rec.unmapped.append(f"{ckpt_key}: {path}")
            continue
        want = module_shapes[path]
        got = _ckpt_shape(ckpt_key)
        if got is not None and want != got:
            rec.shape_mismatch.append((path, want, got))
            continue
        rec.matched[ckpt_key] = path

    filled = set(rec.matched.values())
    rec.missing = sorted(set(module_shapes) - filled)
    return rec


def load_vision_tower(
    weights_path: str | Path,
    config: Optional[Siglip2VisionConfig] = None,
    dtype: mx.Dtype = mx.bfloat16,
    verbose: bool = False,
) -> tuple[Siglip2VisionModel, Reconciliation]:
    """체크포인트를 검사한 뒤 MLX 비전 타워에 로드."""
    cfg = config or Siglip2VisionConfig()
    model = Siglip2VisionModel(cfg)

    header = read_safetensors_header(weights_path)
    rec = reconcile_keys(header, model)
    if verbose:
        print(rec.report())
    if not rec.ok:
        raise RuntimeError(
            "비전 체크포인트 키 대조 실패: 부분 로드는 허용하지 않는다.\n" + rec.report()
        )

    weights = mx.load(str(weights_path))
    to_load = [(rec.matched[k], weights[k].astype(dtype)) for k in rec.matched]
    model.load_weights(to_load, strict=True)
    model.eval()
    mx.eval(model.parameters())
    del weights
    return model, rec


# CLI: 키 대조만 (가중치 데이터는 읽지 않는다)
def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m klava_mlx.vision",
        description="비전 체크포인트 키 와 MLX 모듈 파라미터 대조 (헤더만 읽는다).",
    )
    ap.add_argument("--weights", required=True, help="vision_encoder.safetensors 경로")
    ap.add_argument("--meta", default=None, help="체크포인트 meta.json (설정 출처)")
    ap.add_argument("--hf-config", default=None,
                    help="google/siglip2-... 의 config.json (vision_config 출처)")
    ap.add_argument("--show", type=int, default=8, help="샘플로 출력할 매핑 개수")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8")) if args.meta else {}
    hf_cfg = (
        json.loads(Path(args.hf_config).read_text(encoding="utf-8"))
        if args.hf_config else None
    )
    cfg = Siglip2VisionConfig.from_meta(meta, hf_cfg)
    print("설정 (출처: meta.json + hf config.json + dataclass 기본값)")
    for k, v in cfg.__dict__.items():
        print(f"  {k:24s} = {v}")

    model = Siglip2VisionModel(cfg)
    n_params = sum(int(np.prod(v.shape)) for _, v in tree_flatten(model.parameters()))
    print(f"\n모듈 파라미터 텐서 {len(tree_flatten(model.parameters()))}개, "
          f"총 {n_params:,} 개 원소")

    header = read_safetensors_header(args.weights)
    n_keys = len([k for k in header if k != "__metadata__"])
    print(f"체크포인트 텐서 {n_keys}개: {args.weights}")

    rec = reconcile_keys(header, model)
    print("\n대조 결과")
    print(rec.report())

    print(f"\n샘플 매핑 (앞 {args.show}개, 정렬)")
    for k in sorted(rec.matched)[: args.show]:
        shape = header[k]["shape"]
        print(f"  {k:56s}: {rec.matched[k]:56s} {shape}")

    print("\n" + ("대조 통과: 모든 키가 소비되고 모든 파라미터가 채워진다."
                  if rec.ok else "대조 실패."))
    return 0 if rec.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
