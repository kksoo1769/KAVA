from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable


import gc
import torch
from peft import PeftModel


from kava.data.tokenizer import load_tokenizer
from kava.device import resolve_device
from kava.eval.vlm_generate import load_vlm_for_inference, generate_text
from kava.serving.protocol import StreamResult
from kava.vision.siglip2 import build_siglip2_processor


@dataclass
class Job:
    """Queue의 job 하나"""
    kind: str
    payload: dict
    future: Future

class SharedRuntime:
    """EXAONE을 한 번만 로드하여 KLaVA(adapter on)와 Supervisor(adapter off)에 공유"""
    def __init__(self, ckpt_dir, device=None):
        device = resolve_device(device)
        self.tokenizer = load_tokenizer()
        self.vlm, self.meta = load_vlm_for_inference(ckpt_dir, device=device, dtype=torch.bfloat16, verbose=False)
        self.processor = build_siglip2_processor(self.meta.get("siglip_model_id", "google/siglip2-so400m-patch16-naflex"), int(self.meta.get("siglip_num_patches", 784)))
        self.device = device

        self.queue = queue.Queue(maxsize=8)
        self.worker = threading.Thread(target=self.worker_loop, name="model-worker", daemon=True)
        self.worker.start()

    def cleanup_device_cache(self):
        if not str(self.device).startswith("mps"):
            return

        try:
            torch.mps.synchronize()
            gc.collect()
            torch.mps.empty_cache()
        except Exception as e:
            print(f"[MPS cleanup warning] {e}", flush=True)

    def worker_loop(self):
        while True:
            job: Job = self.queue.get() # queue에서 job 꺼내기
            if job is None: # 종료 신호
                self.queue.task_done()
                break
            
            try:
                if job.kind == "vlm": # KLaVA 요청인 경우
                    result = self.run_vlm(**job.payload)
                elif job.kind == "chat": # Supervisor 요청인 경우
                    result = self.run_chat(**job.payload)
                else:
                    raise ValueError(f"알 수 없는 job: {job.kind}")
                job.future.set_result(result) # 완료시 future에 결과 기록
            except Exception as e:
                job.future.set_exception(e) # error 발생 처리
            finally:
                self.cleanup_device_cache()
                self.queue.task_done()
    
    def submit(self, kind: str, payload: dict) -> Future:
        """Queue에 job put"""
        future = Future()
        self.queue.put(Job(kind=kind, payload=payload, future=future))
        return future

    def run(self, kind: str, payload: dict, timeout: float | None = 600.) -> str | dict:
        """작업을 큐에 넣고 완료된 결과를 반환."""
        return self.submit(kind, payload).result(timeout=timeout)

    def stream(
        self, kind: str, payload: dict, timeout: float | None = 600.0
    ) -> Iterator[str | StreamResult]:
        """완성된 torch 결과를 한 번에 전달하고 StreamResult로 마무리."""
        result = self.run(kind, payload, timeout=timeout)
        # 화면에 표시할 텍스트만 전달한다
        text = result if isinstance(result, str) else result.get("response", "")
        if text:
            yield text
        yield StreamResult(result)

    def shutdown(self, timeout: float = 10.):
        """종료 신호(None)을 queue에 넣고 worker가 종료될 때까지 대기"""
        self.queue.put(None)
        self.worker.join(timeout=timeout) # timeout 또는 종료까지 대기

    def run_vlm(
        self,
        img_path: str, prompt: str,
        temperature: float = .1, max_new_tokens: int = 2048,
        history: list[dict] | None = None,
        enable_thinking: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """KLaVA의 단일 턴과 여러 턴 요청을 처리.

        torch 백엔드는 토큰 스트리밍을 지원하지 않아 on_token을 사용하지 않는다.
        """
        return generate_text(
            self.vlm, self.tokenizer, img_path, prompt,
            siglip_proc=self.processor,
            max_new_tokens=max_new_tokens, temperature=temperature,
            device=self.device, history=history, enable_thinking=enable_thinking,
        )

    def run_chat(
        self,
        messages: list[dict], tools: list[dict] | None = None,
        temperature: float = .1, max_new_tokens: int = 4096,
        enable_thinking: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> dict:
        """배치 크기 1의 Supervisor 요청을 처리.

        torch 백엔드는 토큰 스트리밍을 지원하지 않아 on_token을 사용하지 않는다.
        """
        encoded = self.tokenizer.apply_chat_template(
            messages, tools=tools, enable_thinking=enable_thinking,
            add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        input_ids, attn_mask = encoded["input_ids"], encoded["attention_mask"]
        do_sample = temperature > 0

        adapter_off = (self.lm.disable_adapter() if isinstance(self.lm, PeftModel) else nullcontext())
        with adapter_off:
            out = self.lm.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature if do_sample else None,
                top_p=.95 if do_sample else None,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("[|endofturn|]"),
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 프롬프트를 제외한 새 토큰들만 slicing
        new_tokens = out[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 답변, 추론 과정을 명시적으로 분리해 반환
        if not enable_thinking:
            return {"response": text, "reasoning": "", "think_closed": True}
        
        reasoning, sep, answer = text.partition("</think>")
        if not sep: # 사고 과정이 토큰을 모두 사용하는 경우
            return {"response": "", "reasoning": reasoning.strip(), "think_closed": False}
        return {"response": answer.strip(), "reasoning": reasoning.strip(), "think_closed": True}
        
    @property
    def lm(self):
        return self.vlm.language_model
