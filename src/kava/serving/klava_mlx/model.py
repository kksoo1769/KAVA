"""PyTorch KLaVA를 MLX로 변환 작성한 코드."""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from kava.ckpt import load_meta
from .preprocess import preprocess_image
from .vision import (
    Siglip2VisionConfig,
    Siglip2VisionModel,
    load_vision_tower,
    read_safetensors_header,
)

__all__ = [
    "IMAGE_TOKEN_INDEX",
    "IMAGE_PLACEHOLDER",
    "SYS_VLM",
    "ENDOFTURN",
    "tokenizer_image_token",
    "build_vlm_messages",
    "Connector",
    "KlavaExaone4",
    "LoraController",
    "KlavaMLXVLM",
    "load_klava_mlx",
    "GenerationOutput",
    "emit_token",
]

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
IMAGE_PLACEHOLDER = "<image>"
ENDOFTURN = "[|endofturn|]"
SYS_VLM = (
    "당신은 이미지와 텍스트를 이해하는 AI 어시스턴트입니다. "
    "이미지가 함께 주어지면 이미지에 실제로 보이는 내용에 근거해 사실대로 답하고, "
    "이미지가 없으면 주어진 텍스트에 근거해 답합니다. "
    "질문이 한국어이면 한국어로, 영어이면 영어로 답합니다."
)


def emit_token(on_token: Callable[[str], None], text: str) -> bool:
    """토큰 콜백을 호출하고 성공 여부를 반환."""
    try:
        on_token(text)
        return True
    except Exception as e: # noqa: BLE001 - 화면 전달 실패가 생성을 죽이면 안 된다
        print(f"[on_token warning] {type(e).__name__}: {e}", flush=True)
        return False


def _hf_tokenizer(tokenizer):
    """mlx-lm 의 TokenizerWrapper 안쪽 HF 토크나이저를 꺼낸다(없으면 그대로)."""
    return getattr(tokenizer, "_tokenizer", tokenizer)


def tokenizer_image_token(text: str, tokenizer) -> list[int]:
    """이미지 마커를 유지하면서 텍스트를 토큰화."""
    hf = _hf_tokenizer(tokenizer)
    parts = text.split(IMAGE_PLACEHOLDER)
    out: list[int] = []
    for i, part in enumerate(parts):
        if i > 0:
            out.append(IMAGE_TOKEN_INDEX)
        if part:
            out.extend(hf(part, add_special_tokens=False)["input_ids"])
    return out


def build_vlm_messages(prompt: str, history: Optional[list[dict]] = None) -> list[dict]:
    """KLaVA 추론용 대화 메시지를 생성."""
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
            raise ValueError(f"history[{index}].role은 user 또는 assistant여야 합니다: {role!r}")
        if role != expected_role:
            raise ValueError(f"history[{index}].role은 {expected_role}여야 합니다: {role!r}")
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
    first_user = next(m for m in messages if m["role"] == "user")
    first_user["content"] = f"{IMAGE_PLACEHOLDER}\n{first_user['content']}"
    return messages


class Connector(nn.Module):
    def __init__(self, vision_hidden: int, lm_hidden: int):
        super().__init__()
        self.fc0 = nn.Linear(vision_hidden, lm_hidden, bias=True)
        self.fc2 = nn.Linear(lm_hidden, lm_hidden, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc0(x)))

    @staticmethod
    def key_mapping() -> dict[str, str]:
        """nn.Sequential 인덱스 0과 2를 모듈 이름으로 변환."""
        return {
            "0.weight": "fc0.weight",
            "0.bias": "fc0.bias",
            "2.weight": "fc2.weight",
            "2.bias": "fc2.bias",
        }


def load_connector(
    weights_path: str | Path,
    vision_hidden: int,
    lm_hidden: int,
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[Connector, dict[str, str]]:
    """projector.safetensors를 Connector에 빠짐없이 로드."""
    conn = Connector(vision_hidden, lm_hidden)
    mapping = Connector.key_mapping()
    header = read_safetensors_header(weights_path)
    ckpt_keys = {k for k in header if k != "__metadata__"}
    module_shapes = {k: tuple(v.shape) for k, v in tree_flatten(conn.parameters())}

    unmapped = sorted(ckpt_keys - set(mapping))
    missing = sorted(set(module_shapes) - {mapping[k] for k in ckpt_keys & set(mapping)})
    bad_shape = []
    for ck in sorted(ckpt_keys & set(mapping)):
        want = module_shapes[mapping[ck]]
        got = tuple(int(x) for x in header[ck]["shape"])
        if want != got:
            bad_shape.append((mapping[ck], want, got))
    if unmapped or missing or bad_shape:
        raise RuntimeError(
            "커넥터 키 대조 실패: 부분 로드는 허용하지 않는다.\n"
            f"  미매핑(체크포인트에만): {unmapped}\n"
            f"  미충족(모듈 파라미터): {missing}\n"
            f"  shape 불일치: {bad_shape}"
        )

    weights = mx.load(str(weights_path))
    conn.load_weights([(mapping[k], weights[k].astype(dtype)) for k in ckpt_keys], strict=True)
    conn.eval()
    mx.eval(conn.parameters())
    return conn, mapping


# LM: exaone4 + input_embeddings
def _make_exaone4_classes():
    """exaone4 서브클래스 두 개를 만든다. import 시점 부작용을 피하려고 함수 안에 둔다."""
    from mlx_lm.models import exaone4 # noqa: PLC0415

    class _KlavaExaone4Text(exaone4.ExaoneModel):
        """input_embeddings를 받을 수 있도록 확장한 EXAONE 모델."""

        def __call__(self, inputs, cache=None, input_embeddings=None):
        # EXAONE 구현과 다른 키만 바꾼다
            h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings

            if cache is None: # :163-164
                cache = [None] * len(self.layers)
            global_mask = exaone4.create_attention_mask(h, cache[self.full_idx]) # :165
            if self.swa_idx is not None: # :166-171
                swa_mask = exaone4.create_attention_mask(
                    h, cache[self.swa_idx], window_size=self.window_size
                )
            else:
                swa_mask = None

            for layer, c in zip(self.layers, cache): # :173-175
                mask = swa_mask if layer.self_attn.is_local else global_mask
                h = layer(h, mask, c)

            return self.norm(h) # :177

    class KlavaExaone4(exaone4.Model):
        """input_embeddings를 지원하는 EXAONE 래퍼."""

        def __init__(self, args):
            super().__init__(args)
            # 모델을 다시 만들지 않고 호환되는 서브클래스로 바꿔 메모리 중복을 피한다
            if not isinstance(self.model, _KlavaExaone4Text):
                self.model.__class__ = _KlavaExaone4Text

        def __call__(self, inputs, cache=None, input_embeddings=None):
            out = self.model(inputs, cache, input_embeddings)
            if self.args.tie_word_embeddings: # :191-194
                out = self.model.embed_tokens.as_linear(out)
            else:
                out = self.lm_head(out)
            return out

    return KlavaExaone4, exaone4


def _assert_exaone4_shape() -> None:
    """mlx-lm 구현이 포팅한 구조와 맞는지 확인."""
    from mlx_lm.models import exaone4 # noqa: PLC0415

    sig = list(inspect.signature(exaone4.ExaoneModel.__call__).parameters)
    if sig[:3] != ["self", "inputs", "cache"]:
        raise RuntimeError(
            f"mlx_lm.models.exaone4.ExaoneModel.__call__ 시그니처가 바뀌었다: {sig}. "
            "klava_mlx/model.py 의 _KlavaExaone4Text 를 다시 맞춰야 한다."
        )
    src = inspect.getsource(exaone4.ExaoneModel.__call__)
    for token in ("self.embed_tokens(inputs)", "create_attention_mask",
                  "self.full_idx", "self.swa_idx", "self.norm(h)"):
        if token not in src:
            raise RuntimeError(
                f"mlx_lm exaone4 본문에서 {token!r} 을 찾지 못했다: 구현이 바뀌었다. "
                "klava_mlx/model.py 의 _KlavaExaone4Text 를 다시 맞춰야 한다."
            )


def _klava_get_classes(config: dict):
    """KLaVA용 MLX 모델 클래스를 반환."""
    from mlx_lm.utils import _get_classes as mlx_get_classes # noqa: PLC0415

    if config.get("model_type") == "exaone4":
        _assert_exaone4_shape()
        KlavaExaone4, exaone4 = _make_exaone4_classes()
        return KlavaExaone4, exaone4.ModelArgs
    return mlx_get_classes(config)


# LoRA 적용 전환
class LoraController:
    """한 모델에서 LoRA 어댑터의 적용 여부를 전환."""

    def __init__(self, model: nn.Module, adapter_path: str | Path, cache_weights: bool = True):
        self.model = model
        self.adapter_path = Path(adapter_path)
        self.cache_weights = cache_weights
        self._weights: Optional[dict[str, mx.array]] = None
        self.enabled = False

        cfg_path = self.adapter_path / "adapter_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"adapter_config.json 이 없다: {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.fine_tune_type = cfg.get("fine_tune_type", "lora")
        if self.fine_tune_type != "lora":
            raise ValueError(
                f"fine_tune_type={self.fine_tune_type!r} 는 지원하지 않는다 "
                "(mlx_convert.py convert-lora 는 'lora' 만 만든다)."
            )
        self.num_layers = int(cfg["num_layers"])
        self.lora_parameters = dict(cfg["lora_parameters"])
        self.config = cfg

    @property
    def weights_file(self) -> Path:
        return self.adapter_path / "adapters.safetensors"

    def enable(self) -> None:
        if self.enabled:
            return
        from mlx_lm.tuner.utils import linear_to_lora_layers # noqa: PLC0415

        linear_to_lora_layers(self.model, self.num_layers, self.lora_parameters)
        if self._weights is None:
            w = mx.load(str(self.weights_file))
            if self.cache_weights:
                self._weights = w
        else:
            w = self._weights
        # 어댑터에 없는 베이스 가중치는 유지한다
        self.model.load_weights(list(w.items()), strict=False)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.enabled = True

    def disable(self) -> None:
        if not self.enabled:
            return
        from mlx_lm.tuner.lora import LoRALinear # noqa: PLC0415

        replacements = [
            (path, module.linear)
            for path, module in self.model.named_modules()
            if isinstance(module, LoRALinear)
        ]
        if replacements:
            self.model.update_modules(tree_unflatten(replacements))
        self.model.eval()
        self.enabled = False

    def set(self, on: bool) -> None:
        self.enable() if on else self.disable()

    def n_lora_layers(self) -> int:
        from mlx_lm.tuner.lora import LoRALinear # noqa: PLC0415

        return sum(1 for _, m in self.model.named_modules() if isinstance(m, LoRALinear))


# VLM 조립
@dataclass
class GenerationOutput:
    """한 번의 생성 결과 + 구간별 시간. 러너가 그대로 JSONL 에 넣는다."""

    text: str
    text_streamed: str
    token_ids: list[int] = field(default_factory=list)
    n_visual_tokens: int = 0
    n_text_prompt_tokens: int = 0
    n_prompt_tokens: int = 0
    n_generated_tokens: int = 0
    t_tokenize_s: float = 0.0
    t_image_preprocess_s: float = 0.0
    t_vision_encode_s: float = 0.0
    t_merge_s: float = 0.0
    t_generate_s: float = 0.0
    ttft_s: Optional[float] = None
    finish_reason: Optional[str] = None
    reported_generation_tokens: int = -1
    spatial_shapes: tuple[int, int] = (0, 0)
    resized_hw: tuple[int, int] = (0, 0)


class KlavaMLXVLM:
    """비전 타워 + 커넥터 + LM(+선택적 LoRA) 하나로 묶은 추론 객체."""

    def __init__(
        self,
        vision: Siglip2VisionModel,
        connector: Connector,
        lm: nn.Module,
        tokenizer,
        meta: dict,
        vision_config: Siglip2VisionConfig,
        lora: Optional[LoraController] = None,
        embed_dtype: mx.Dtype = mx.bfloat16,
    ):
        self.vision = vision
        self.connector = connector
        self.lm = lm
        self.tokenizer = tokenizer
        self.meta = meta
        self.vision_config = vision_config
        self.lora = lora
        self.embed_dtype = embed_dtype
        self.max_num_patches = int(meta["siglip_num_patches"])

    # -- 이미지 -------------------------------------------------------------------
    def preprocess(self, image_path: str, resize_backend: str = "numpy_aa") -> dict:
        """VLM 입력에 맞게 이미지를 전처리."""
        return preprocess_image(
            image_path,
            patch_size=self.vision_config.patch_size,
            max_num_patches=self.max_num_patches,
            resize_backend=resize_backend,
        )

    def encode_image(self, bundle: dict) -> tuple[mx.array, mx.array]:
        """이미지를 비전 토큰으로 변환."""
        pv = mx.array(bundle["pixel_values"])[None] # (1,P,768)
        pam = mx.array(bundle["pixel_attention_mask"])[None] # (1,P)
        shapes = bundle["spatial_shapes"][None] # (1,2)
        feats = self.vision(pv, pam, shapes) # (1,P,1152)
        p = feats.shape[1]
        if p != self.max_num_patches: # :137-139
            raise RuntimeError(f"vision tokens {p} != expected {self.max_num_patches}")
        return self.connector(feats.astype(self.embed_dtype)), pam.astype(mx.bool_)

    # -- splice --------------------------------------------------------------------
    def merge(
        self, input_ids: list[int], image_embeds: mx.array, image_token_mask: mx.array
    ) -> tuple[mx.array, int, int]:
        """배치 크기 1에서 텍스트 임베딩과 유효한 이미지 토큰을 결합."""
        if image_embeds.shape[0] != 1:
            raise ValueError("배치 1 만 지원한다 (position_ids 등가성 근거는 docstring 참고).")
        n_marker = sum(1 for t in input_ids if t == IMAGE_TOKEN_INDEX)
        if n_marker != 1:
        # 학습 경로와 같은 입력 길이 검사를 적용한다
            raise RuntimeError(f"KLaVA 프롬프트에는 이미지 토큰이 1개여야 합니다: {n_marker}개")
        p = input_ids.index(IMAGE_TOKEN_INDEX)

        embed_tokens = self.lm.model.embed_tokens
        pre_ids = input_ids[:p]
        post_ids = input_ids[p + 1:]

        # MLX 0.32는 불리언 인덱싱을 지원하지 않아 정수 인덱스를 사용한다
        mask_np = np.asarray(image_token_mask[0].astype(mx.int32).tolist(), dtype=np.int64)
        idx = np.nonzero(mask_np)[0]
        if idx.size == 0:
            raise RuntimeError("유효한 시각 패치가 하나도 없다: pixel_attention_mask 가 전부 0이다.")
        img_toks = mx.take(image_embeds[0], mx.array(idx), axis=0)
        n_visual = int(img_toks.shape[0])

        pieces = []
        if pre_ids:
            pieces.append(embed_tokens(mx.array(pre_ids)).astype(self.embed_dtype))
        pieces.append(img_toks.astype(self.embed_dtype))
        if post_ids:
            pieces.append(embed_tokens(mx.array(post_ids)).astype(self.embed_dtype))
        merged = mx.concatenate(pieces, axis=0) # :179
        return merged, n_visual, len(pre_ids) + len(post_ids)

    # -- 프롬프트 -------------------------------------------------------------------
    def build_prompt_ids(
        self, prompt: str, history: Optional[list[dict]] = None, enable_thinking: bool = False
    ) -> list[int]:
        """이미지 임베딩과 텍스트 임베딩을 입력 순서대로 결합."""
        messages = build_vlm_messages(prompt, history=history)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return tokenizer_image_token(text, self.tokenizer)

    # -- 생성 -----------------------------------------------------------------------
    def eos_token_ids(self) -> list[int]:
        """토크나이저와 설정에 등록된 EOS 토큰 id를 반환."""
        return sorted(int(x) for x in self.tokenizer.eos_token_ids)

    def generate(
        self,
        image_path: str,
        prompt: str,
        history: Optional[list[dict]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        prefill_step_size: int = 2048,
        resize_backend: str = "numpy_aa",
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        on_generation_start: Optional[Callable[[int], None]] = None,
        on_first_token: Optional[Callable[[], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_generation_end: Optional[Callable[[int], None]] = None,
        extra_generate_kwargs: Optional[dict] = None,
    ) -> GenerationOutput:
        """이미지와 프롬프트로 답변을 생성하고 구간별 시간을 반환."""
        from mlx_lm import stream_generate # noqa: PLC0415

        if temperature != 0.0 and sampler is None:
            raise ValueError(
                f"temperature={temperature}: sampler 없이 이 경로는 greedy 전용이다. "
                "샘플러 동등성을 먼저 설계하지 않고 켜면 두 백엔드 비교가 무의미해진다."
            )

        # 텍스트 토큰화
        t0 = time.perf_counter()
        input_ids = self.build_prompt_ids(prompt, history=history,
                                          enable_thinking=enable_thinking)
        t_tokenize = time.perf_counter() - t0

        # 이미지 전처리
        t0 = time.perf_counter()
        bundle = self.preprocess(image_path, resize_backend=resize_backend)
        t_preprocess = time.perf_counter() - t0

        # 비전 인코딩과 커넥터 적용
        t0 = time.perf_counter()
        image_embeds, token_mask = self.encode_image(bundle)
        mx.eval(image_embeds, token_mask)
        t_encode = time.perf_counter() - t0

        # 텍스트와 시각 토큰 결합
        t0 = time.perf_counter()
        embeds, n_visual, n_text = self.merge(input_ids, image_embeds, token_mask)
        mx.eval(embeds)
        t_merge = time.perf_counter() - t0
        del image_embeds

        n_prompt = int(embeds.shape[0])

        # 학습 경로와 같은 방식으로 컨텍스트 길이를 제한한다
        max_new = int(max_new_tokens)
        limit = self.context_limit()
        if limit is not None:
            remaining = int(limit) - n_prompt
            if remaining < 1:
                # 사용자에게 전달되는 오류 문구는 torch 경로와 같게 유지한다
                raise ValueError(
                    f"멀티 턴 입력이 모델 컨텍스트 한도({limit})를 초과했습니다. "
                    "오래된 대화 턴을 줄여 주세요."
                )
            max_new = min(max_new, remaining)

        gen_kwargs: dict[str, Any] = {
            "max_tokens": max_new,
            "prefill_step_size": prefill_step_size,
            "input_embeddings": embeds,
        }
        if sampler is not None:
            gen_kwargs["sampler"] = sampler
        gen_kwargs.update(extra_generate_kwargs or {})

        # 임베딩만 사용하는 경로이므로 prompt에는 빈 배열을 넘긴다
        empty_prompt = mx.array([], dtype=mx.int32)
        token_ids: list[int] = []
        pieces: list[str] = []
        ttft: Optional[float] = None
        last_resp = None

        if on_generation_start is not None:
            on_generation_start(n_prompt)
        t_gen0 = time.perf_counter()
        for resp in stream_generate(self.lm, self.tokenizer, empty_prompt, **gen_kwargs):
            if ttft is None:
                ttft = time.perf_counter() - t_gen0
                if on_first_token is not None:
                    on_first_token()
            token_ids.append(int(resp.token))
            pieces.append(resp.text)
            # 빈 조각(resp.text == "")은 흘리지 않는다. 소비자 쪽에 의미 없는 이벤트만 쌓인다.
            if on_token is not None and resp.text:
                if not emit_token(on_token, resp.text):
                    on_token = None # 한 번 실패하면 그 뒤로는 부르지 않는다
            last_resp = resp
        _sync()
        t_generate = time.perf_counter() - t_gen0
        if on_generation_end is not None:
            on_generation_end(len(token_ids))

        text = _decode(self.tokenizer, token_ids)
        nh, nw = (int(v) for v in bundle["spatial_shapes"])
        return GenerationOutput(
            text=text,
            text_streamed="".join(pieces),
            token_ids=token_ids,
            n_visual_tokens=n_visual,
            n_text_prompt_tokens=n_text,
            n_prompt_tokens=n_prompt,
            n_generated_tokens=len(token_ids),
            t_tokenize_s=t_tokenize,
            t_image_preprocess_s=t_preprocess,
            t_vision_encode_s=t_encode,
            t_merge_s=t_merge,
            t_generate_s=t_generate,
            ttft_s=ttft,
            finish_reason=getattr(last_resp, "finish_reason", None),
            reported_generation_tokens=int(getattr(last_resp, "generation_tokens", -1))
            if last_resp is not None else -1,
            spatial_shapes=(nh, nw),
            resized_hw=tuple(bundle["meta"]["resized_hw"]),
        )

    def context_limit(self) -> Optional[int]:
        """언어 모델의 최대 입력 길이를 반환."""
        args = getattr(self.lm, "args", None)
        return getattr(args, "max_position_embeddings", None) if args is not None else None


def _decode(tokenizer, ids: list[int]) -> str:
    """특수 토큰을 제외하고 앞뒤 공백을 정리해 디코딩."""
    try:
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
    except TypeError:
        return tokenizer.decode(ids).strip()


def _sync() -> None:
    fn = getattr(mx, "synchronize", None)
    if fn is not None:
        try:
            fn()
        except Exception: # noqa: BLE001 - 계측이 벤치를 죽이면 안 된다
            pass


# 로더
def load_klava_mlx(
    lm_dir: str | Path,
    ckpt_dir: str | Path,
    adapter_path: Optional[str | Path] = None,
    enable_lora: bool = True,
    dtype: mx.Dtype = mx.bfloat16,
    hf_siglip_config: Optional[str | Path] = None,
    verbose: bool = False,
) -> KlavaMLXVLM:
    """MLX 언어 모델과 비전 타워 및 커넥터를 로드."""
    from mlx_lm.utils import load_model, load_tokenizer # noqa: PLC0415

    lm_dir = Path(lm_dir)
    ckpt_dir = Path(ckpt_dir)
    meta = load_meta(ckpt_dir)

    hf_cfg = None
    if hf_siglip_config:
        hf_cfg = json.loads(Path(hf_siglip_config).read_text(encoding="utf-8"))
    vcfg = Siglip2VisionConfig.from_meta(meta, hf_cfg)

    # 언어 모델 로드
    lm, lm_config = load_model(lm_dir, get_model_classes=_klava_get_classes)
    tokenizer = load_tokenizer(lm_dir, eos_token_ids=lm_config.get("eos_token_id"))
    from mlx_lm.utils import does_model_support_input_embeddings # noqa: PLC0415

    if not does_model_support_input_embeddings(lm):
        raise RuntimeError(
            "LM 이 input_embeddings 를 지원하지 않는다: _klava_get_classes 가 적용되지 "
            "않았다는 뜻이다. 이대로 두면 VLM 경로가 아예 동작하지 않는다."
        )
    lm_hidden = int(lm_config["hidden_size"])

    lora = None
    if adapter_path is not None:
        lora = LoraController(lm, adapter_path)
        if enable_lora:
            lora.enable()
    elif enable_lora:
        raise ValueError("enable_lora=True 인데 adapter_path 가 없다.")

    # 비전 모델과 커넥터 로드
    vision, rec = load_vision_tower(
        ckpt_dir / "vision_encoder.safetensors", config=vcfg, dtype=dtype, verbose=verbose
    )
    connector, _ = load_connector(
        ckpt_dir / "projector.safetensors",
        vision_hidden=vcfg.hidden_size, lm_hidden=lm_hidden, dtype=dtype,
    )
    if verbose:
        print(f"[klava_mlx] vision keys matched {len(rec.matched)}, "
              f"lm_hidden={lm_hidden}, lora={'on' if (lora and lora.enabled) else 'off'}")

    return KlavaMLXVLM(
        vision=vision, connector=connector, lm=lm, tokenizer=tokenizer, meta=meta,
        vision_config=vcfg, lora=lora, embed_dtype=dtype,
    )
