"""KLaVA MLX 런타임을 제공.

torch 런타임과 같은 공개 인터페이스를 유지하며 MLX 모델을 한 번만 로드해 공유한다.
"""

from __future__ import annotations

import gc
import os
import queue
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import mlx.core as mx

from kava.paths import REPO_ROOT
from kava.serving.protocol import StreamResult

from . import MLX_SUBDIR, default_ckpt_dir, mlx_root
from .model import ENDOFTURN, emit_token, load_klava_mlx

__all__ = ["MLXSharedRuntime", "MLXRuntimeConfig", "Job", "RESIZE_BACKEND", "MLX_SUBDIR"]


# 전처리 백엔드는 상수다. 환경변수로 바꿀 수 없다(모듈 docstring 참고).
RESIZE_BACKEND = "numpy_aa"

# KLAVA_MLX_QUANT 값: <ckpt_dir>/mlx/ 아래 기본 디렉터리 이름
_QUANT_DIRS = {
    "bf16": "exaone4-1.2b-bf16",
    "8bit": "exaone4-1.2b-8bit",
    "4bit": "exaone4-1.2b-4bit",
}

# torch 판이 lm.generate(top_p=0.95) 로 쓰는 값. do_sample 일 때만 적용된다.
TOP_P = 0.95


# 설정
@dataclass(frozen=True)
class MLXRuntimeConfig:
    """MLX 런타임 환경 변수 설정."""

    ckpt_dir: str
    model_dir: str
    adapter_dir: str
    quant: str = "bf16"
    prefill_step_size: int = 2048

    @staticmethod
    def default_ckpt_dir() -> str:
        # KLAVA_CKPT 를 보지 않는 "저장소 기본값"이어야 한다(from_env 가 순서를 정한다).
        return str(REPO_ROOT / "runs" / "klava_instruct_784_r64" / "ckpts" / "fin")

    @classmethod
    def from_env(cls, ckpt_dir: str | os.PathLike | None = None) -> "MLXRuntimeConfig":
        ckpt = str(ckpt_dir) if ckpt_dir is not None else default_ckpt_dir()

        quant = (os.environ.get("KLAVA_MLX_QUANT") or "bf16").strip().lower()
        if quant not in _QUANT_DIRS:
            raise ValueError(
                f"KLAVA_MLX_QUANT={quant!r} 는 모르는 값이다. "
                f"허용: {sorted(_QUANT_DIRS)} (기본 bf16)."
            )
        if quant == "4bit" and os.environ.get("KLAVA_MLX_ALLOW_4BIT") != "1":
            raise ValueError(
                "KLAVA_MLX_QUANT=4bit 는 지원 옵션이 아니다. 4bit 변환본은 한국어를 "
                "사실상 포기한다(실측 한글 문자 비율 0.91: 0.07). 그래도 실험하려면 "
                "KLAVA_MLX_ALLOW_4BIT=1 을 함께 지정해야 한다."
            )
        if quant == "8bit":
            print(
                "[MLXSharedRuntime] 경고: KLAVA_MLX_QUANT=8bit 는 디코드는 가장 빠르지만 "
                "VLM prefill(TTFT)이 torch 대비 60~75% 느리다. 라우터 노드는 "
                "max_new_tokens=8 로 VLM 을 호출하므로 사실상 TTFT 만 본다. "
                "기본값(bf16)에서 벗어난 설정임을 인지하고 쓰는 것인지 확인하라.",
                flush=True,
            )

        # 명시하지 않은 경로는 체크포인트 디렉터리에서 만든다
        root = mlx_root(ckpt)
        model_dir = os.environ.get("KLAVA_MLX_MODEL_DIR") or str(root / _QUANT_DIRS[quant])
        adapter_dir = os.environ.get("KLAVA_MLX_ADAPTER_DIR") or str(root / "lora-mlx")

        raw_prefill = os.environ.get("KLAVA_MLX_PREFILL_STEP")
        prefill = int(raw_prefill) if raw_prefill else 2048
        if prefill < 1:
            raise ValueError(f"KLAVA_MLX_PREFILL_STEP={prefill} 은 1 이상이어야 한다.")

        return cls(
            ckpt_dir=ckpt,
            model_dir=model_dir,
            adapter_dir=adapter_dir,
            quant=quant,
            prefill_step_size=prefill,
        )

    def validate(self) -> None:
        """없는 경로를 늦게(첫 요청에서) 터뜨리지 않고 생성자에서 바로 잡는다."""
        missing: list[str] = []
        for label, path, needed in (
            ("KLAVA_CKPT", Path(self.ckpt_dir), ["meta.json", "projector.safetensors",
                                                 "vision_encoder.safetensors"]),
            ("KLAVA_MLX_MODEL_DIR", Path(self.model_dir), ["config.json"]),
            ("KLAVA_MLX_ADAPTER_DIR", Path(self.adapter_dir), ["adapter_config.json",
                                                               "adapters.safetensors"]),
        ):
            if not path.is_dir():
                missing.append(f"{label}: 디렉터리가 없다: {path}")
                continue
            for f in needed:
                if not (path / f).is_file():
                    missing.append(f"{label}: {f} 가 없다: {path / f}")
        if missing:
            py = REPO_ROOT / ".venv" / "bin" / "python"
            raise FileNotFoundError(
                "MLX 런타임 아티팩트가 없어서 서버를 띄울 수 없다.\n\n"
                "없는 것:\n  " + "\n  ".join(missing) + "\n\n"
                "고치는 법: 아래 변환을 1회 실행한다(원본 체크포인트는 건드리지 않는다):\n"
                f"  {py} -m kava.serving.klava_mlx.convert convert-base --bits none --dry-run\n"
                f"  {py} -m kava.serving.klava_mlx.convert convert-base --bits none\n"
                f"  {py} -m kava.serving.klava_mlx.convert convert-lora\n"
                f"  {py} -m kava.serving.klava_mlx.convert verify\n\n"
                "당장 서버가 떠야 한다면 torch 백엔드로 우회한다:\n"
                "  KLAVA_BACKEND=torch <서버 실행 명령>\n\n"
                "※ 자동으로 torch 로 폴백하지 않는다. 이 기계에서 torch 경로는 실측 약 60 GB "
                "스왑아웃과 최대 5배 지연을 유발한다. 조용한 폴백은 '원인 모를 느려짐'이 되고, "
                "명시적 실패는 '설정이 틀렸다'가 된다. 후자가 낫다."
            )


# torch 런타임과 같은 작업 형식을 사용한다
@dataclass
class Job:
    """Queue의 job 하나"""

    kind: str
    payload: dict
    future: Future
    # 스트리밍 요청에서 증분 텍스트와 종료 센티널을 전달한다
    tokens: queue.Queue | None = None


# 런타임
class MLXSharedRuntime:
    """EXAONE을 한 번만 로드하여 KLaVA(adapter on)와 Supervisor(adapter off)에 공유.

    kava.serving.runtime.SharedRuntime 의 MLX 판. 시그니처와 반환 타입이 같다.
    """

    # 공개 시그니처는 torch 런타임과 같게 유지한다
    def __init__(self, ckpt_dir, device=None):
        # MLX는 디바이스를 선택하지 않지만 요청값은 진단용으로 보관한다
        self.requested_device = device
        self.device = "mlx"

        self.config = MLXRuntimeConfig.from_env(ckpt_dir)
        self.config.validate()

        if RESIZE_BACKEND != "numpy_aa": # pragma: no cover - 상수 변조 방지용 가드
            raise RuntimeError(
                f"RESIZE_BACKEND={RESIZE_BACKEND!r}: 'numpy_aa' 이외의 백엔드는 "
                "torchvision 참조와 픽셀 원소의 최대 14.5%에서 어긋납니다."
            )

        # KLaVA가 기본으로 사용하므로 어댑터를 켠 상태로 로드한다
        self.vlm = load_klava_mlx(
            lm_dir=self.config.model_dir,
            ckpt_dir=self.config.ckpt_dir,
            adapter_path=self.config.adapter_dir,
            enable_lora=True,
            dtype=mx.bfloat16,
            verbose=False,
        )
        self.tokenizer = self.vlm.tokenizer
        self.meta = self.vlm.meta

        # MLX 전처리는 체크포인트 메타데이터의 비전 패치 수를 사용한다
        self.processor = None
        self.max_num_patches = self.vlm.max_num_patches
        self.lm_id = self.meta.get("exaone_id", "LGAI-EXAONE/EXAONE-4.0-1.2B")

        # 두 백엔드가 같은 지점에서 생성을 끝내는지 확인한다
        self.eos_check = self._check_eos()
        if not self.eos_check["ok"]:
            raise RuntimeError(f"EOS 불일치: {self.eos_check}")

        # 모델을 로드한 스레드에서 워커 시작 전에 생성 경로를 초기화한다
        self._warmup()

        self.queue = queue.Queue(maxsize=8)
        self.worker = threading.Thread(
            target=self.worker_loop, name="model-worker", daemon=True
        )
        self.worker.start()

    # 모델 초기화
    def _warmup(self) -> None:
        """워커를 시작하기 전에 현재 스레드에서 MLX 생성 상태를 초기화."""
        try:
            self.run_chat(
                [{"role": "user", "content": "안녕"}],
                temperature=0.0,
                max_new_tokens=1,
                enable_thinking=False,
            )
        except Exception as exc: # noqa: BLE001
            raise RuntimeError(
                "MLX 생성 워밍업이 실패했다. 모델은 로드됐지만 생성이 안 되는 "
                f"상태다: {type(exc).__name__}: {exc}\n"
                "KLAVA_BACKEND=torch 로 우회할 수 있다."
            ) from exc

# EOS와 adapter
    def _check_eos(self) -> dict:
        try:
            eot = self.tokenizer.convert_tokens_to_ids(ENDOFTURN)
        except Exception as exc: # noqa: BLE001
            return {"ok": False, "reason": f"convert_tokens_to_ids 실패: {exc}"}
        try:
            mlx_eos = sorted(int(x) for x in self.tokenizer.eos_token_ids)
        except Exception as exc: # noqa: BLE001
            return {"ok": False, "reason": f"eos_token_ids 읽기 실패: {exc}",
                    "endofturn_id": eot}
        ok = eot is not None and eot in mlx_eos
        return {
            "ok": ok,
            "endofturn_id": eot,
            "mlx_eos_token_ids": mlx_eos,
            "reason": None if ok else f"{ENDOFTURN} 가 mlx eos_token_ids 에 없다",
        }

    @contextmanager
    def _adapter(self, on: bool):
        """요청에 따라 LoRA 어댑터를 켜거나 끄는 컨텍스트."""
        lora = self.vlm.lora
        if lora is None:
            yield
            return
        prev = lora.enabled
        lora.set(on)
        try:
            yield
        finally:
            lora.set(prev)

    @property
    def lora_enabled(self) -> bool:
        """진단용. 테스트가 어댑터 상태 전이를 확인할 때 쓴다."""
        return bool(self.vlm.lora is not None and self.vlm.lora.enabled)

    # 캐시 정리
    def cleanup_device_cache(self):
        """MLX 연산을 동기화하고 사용하지 않는 메모리를 정리."""
        try:
            mx.synchronize()
            gc.collect()
            mx.clear_cache()
        except Exception as e: # noqa: BLE001
            print(f"[MLX cleanup warning] {e}", flush=True)

    # 작업 워커
    def worker_loop(self):
        while True:
            job: Job = self.queue.get() # queue에서 job 꺼내기
            if job is None: # 종료 신호
                self.queue.task_done()
                break

            # 스트리밍 요청에만 토큰 큐를 콜백으로 넘긴다
            on_token = None
            if job.tokens is not None:
                on_token = job.tokens.put

            try:
                if job.kind == "vlm": # KLaVA 요청인 경우
                    result = self.run_vlm(**job.payload, on_token=on_token)
                elif job.kind == "chat": # Supervisor 요청인 경우
                    result = self.run_chat(**job.payload, on_token=on_token)
                else:
                    raise ValueError(f"알 수 없는 job: {job.kind}")
                job.future.set_result(result) # 완료시 future에 결과 기록
            except Exception as e: # noqa: BLE001
                job.future.set_exception(e) # error 발생 처리
            finally:
                if job.tokens is not None:
                    # 예외가 발생해도 대기 중인 소비자가 끝나도록 센티널을 보낸다
                    job.tokens.put(None)
                self.cleanup_device_cache()
                self.queue.task_done()

    def submit(self, kind: str, payload: dict) -> Future:
        """Queue에 job put"""
        future = Future()
        self.queue.put(Job(kind=kind, payload=payload, future=future))
        return future

    def run(self, kind: str, payload: dict, timeout: float | None = 600.0) -> str | dict:
        """job을 넣고 완료까지 기다린 후 결과를 반환"""
        return self.submit(kind, payload).result(timeout=timeout)

    def stream(
        self, kind: str, payload: dict, timeout: float | None = 600.0
    ) -> Iterator[str | StreamResult]:
        """증분 텍스트를 전달한 뒤 StreamResult로 마무리."""
        tokens: queue.Queue = queue.Queue() # maxsize 없음 = 무제한 (위 주석 참고)
        future: Future = Future()
        self.queue.put(Job(kind=kind, payload=payload, future=future, tokens=tokens))

        # timeout은 토큰마다가 아니라 스트림 전체에 적용한다
        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + timeout

        while True:
            wait = None # None = 제한 없이 기다린다
            if deadline is not None:
                wait = max(0.0, deadline - time.monotonic())
            try:
                text = tokens.get(timeout=wait)
            except queue.Empty as exc:
                # 일반 요청과 같은 TimeoutError로 변환한다
                raise TimeoutError("모델 추론이 timeout 되었습니다.") from exc
            if text is None: # 센티널 = 생성 끝(성공이든 실패든)
                break
            yield text

        # 실패한 작업이면 future가 원래 예외를 다시 발생시킨다
        yield StreamResult(future.result())

    def shutdown(self, timeout: float = 10.0):
        """종료 신호(None)을 queue에 넣고 worker가 종료될 때까지 대기"""
        self.queue.put(None)
        self.worker.join(timeout=timeout)

    # sampler 설정
    def _sampler(self, temperature: float) -> Optional[Callable[[Any], Any]]:
        """temperature가 양수일 때 사용할 sampler를 생성."""
        if temperature is None or temperature <= 0:
            return None
        from mlx_lm.sample_utils import make_sampler # noqa: PLC0415

        return make_sampler(temp=float(temperature), top_p=TOP_P)

    def _decode(self, ids: list[int]) -> str:
        """torch 의 tokenizer.decode(..., skip_special_tokens=True).strip() 과 동일."""
        try:
            return self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        except TypeError:
            return self.tokenizer.decode(ids).strip()

    # VLM 요청 처리
    def run_vlm(
        self,
        img_path: str,
        prompt: str,
        temperature: float = 0.1,
        max_new_tokens: int = 2048,
        history: list[dict] | None = None,
        enable_thinking: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """LoRA를 적용해 KLaVA의 단일 턴과 여러 턴 요청을 처리."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens는 1 이상이어야 합니다.")

        with self._adapter(True):
            out = self.vlm.generate(
                image_path=img_path,
                prompt=prompt,
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=float(temperature),
                enable_thinking=enable_thinking,
                prefill_step_size=self.config.prefill_step_size,
                resize_backend=RESIZE_BACKEND,
                sampler=self._sampler(temperature),
                on_token=on_token,
            )
        return out.text

    # 채팅 요청 처리
    def run_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_new_tokens: int = 4096,
        enable_thinking: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> dict:
        """LoRA를 끄고 Supervisor 요청을 처리."""
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_ids = [int(t) for t in prompt_ids]

        with self._adapter(False):
            token_ids = self._generate_chat_ids(
                prompt_ids, max_new_tokens, temperature, on_token=on_token
            )

        # 프롬프트를 제외한 새 토큰들만 slicing: MLX 는 애초에 새 토큰만 준다.
        text = self._decode(token_ids)

        # 답변과 추론 과정을 분리해 반환한다
        if not enable_thinking:
            return {"response": text, "reasoning": "", "think_closed": True}

        reasoning, sep, answer = text.partition("</think>")
        if not sep: # 사고 과정이 토큰을 모두 사용하는 경우
            return {"response": "", "reasoning": reasoning.strip(), "think_closed": False}
        return {"response": answer.strip(), "reasoning": reasoning.strip(), "think_closed": True}

    def _generate_chat_ids(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        on_token: Callable[[str], None] | None = None,
    ) -> list[int]:
        """생성된 토큰 id를 모으면서 증분 텍스트를 전달."""
        from mlx_lm import stream_generate # noqa: PLC0415

        if max_new_tokens <= 0:
            # torch 경로와 같이 새 토큰이 없으면 빈 결과를 반환한다
            return []

        gen_kwargs: dict[str, Any] = {
            "max_tokens": int(max_new_tokens),
            "prefill_step_size": self.config.prefill_step_size,
        }
        sampler = self._sampler(temperature)
        if sampler is not None:
            gen_kwargs["sampler"] = sampler

        token_ids: list[int] = []
        for resp in stream_generate(
            self.vlm.lm, self.tokenizer, mx.array(prompt_ids), **gen_kwargs
        ):
            token_ids.append(int(resp.token))
            if on_token is not None and resp.text:
                # 콜백이 죽어도 생성은 계속한다(model.emit_token docstring 참고).
                if not emit_token(on_token, resp.text):
                    on_token = None
        return token_ids

    # 진단 정보
    @property
    def lm(self):
        """torch 판 self.vlm.language_model 에 대응."""
        return self.vlm.lm

    def describe(self) -> dict:
        """로그와 상태 확인 및 테스트에 사용할 로드 정보를 반환."""
        return {
            "backend": "mlx",
            "runtime_class": type(self).__name__,
            "requested_device": self.requested_device,
            "device": self.device,
            "ckpt_dir": self.config.ckpt_dir,
            "model_dir": self.config.model_dir,
            "adapter_dir": self.config.adapter_dir,
            "quant": self.config.quant,
            "prefill_step_size": self.config.prefill_step_size,
            "resize_backend": RESIZE_BACKEND,
            "lm_id": self.lm_id,
            "max_num_patches": self.max_num_patches,
            "lora_enabled": self.lora_enabled,
            "n_lora_layers": (self.vlm.lora.n_lora_layers() if self.vlm.lora else 0),
            "eos_check": self.eos_check,
            "top_p": TOP_P,
        }


# 드롭인 별칭. from klava_mlx.runtime import SharedRuntime 로도 쓸 수 있게 둔다.
SharedRuntime = MLXSharedRuntime
