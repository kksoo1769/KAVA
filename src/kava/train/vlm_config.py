"""accelerate + PEFT + FSDP 기반 VLM 학습 하이퍼파라미터"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class VLMConfig:
    # 실험과 모델
    run_root: str = "runs"
    run_name: str = "klava_align"
    stage: str = "align" # 'align' | 'instruct'

    exaone_id: str = "LGAI-EXAONE/EXAONE-4.0-1.2B" # Huggingface ID

    # 'siglip2'만 지원한다. AxisMF 백본은 CUDA 전용 mamba-ssm 커널 의존으로 제거했다.
    vision_backbone: str = "siglip2"

    # SigLIP2 NaFlex 전용
    siglip_model_id: str = "google/siglip2-so400m-patch16-naflex"
    siglip_num_patches: int = 784 # 시각 토큰 시퀀스 최대 길이(운영 체크포인트 기준)

    attn_implementation: str = "flash_attention_2" # 'sdpa' | 'flash_attention_2' (LM용)
    projector_init: str = "" # stage 1 projector.safetensors 경로(미설정: 랜덤)

    # LM
    lora_enable: bool = False # True | False
    lora_r: int | None = 128 # rank
    lora_alpha: int | None = 256 # alpha
    lora_dropout: float | None = 0.05
    lora_targets: str | None = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # vision encoder
    train_vision: bool = False # instruct에서 unfreeze 여부
    vision_lr: float = 2e-6 # vision encoder lr

    # data
    data_manifest: str = os.path.join("data", "vlm", "align_stage1.train.jsonl")
    tokenizer_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    img_size: int = 384
    max_len: int = 4096
    micro_batch_size: int = 16
    grad_accum_steps: int = 4
    num_workers: int = 8
    prefetch_factor: int = 6
    add_default_system: bool = True
    train_aug: bool = False

    # optimizer와 schedule
    epochs: int = 1
    base_lr: float = 1e-3 # align: projector lr
    projector_lr: float = 2e-5 # instruct: projector LR
    lm_lr: float = 2e-5 # instruct: LM LR (full fine-tuning)
    lm_lora_lr: float = 1e-4 # instruct: LoRA LR
    min_lr_ratio: float = 0.0 # cosine decay to 0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    max_steps: int = 0 # >0이면 그 스텝에서 조기 종료(스모크, 디버그 용도)
    fused_optimizer: bool = True  

    # resume
    resume: str = "" # 예시) runs/klava_instruct/ckpts/last

    # precision
    param_dtype: str = "bfloat16"
    activation_checkpointing: bool = False

    # seed와 logging
    seed: int = 42
    log_every: int = 10
    save_every_epoch: bool = True
    save_every_steps: int = 0 #
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_run_name: str = ""

    @property
    def lm_mode(self) -> str:
        return 'frozen' if self.stage == 'align' else "lora" if self.lora_enabled else "full"
    @property
    def lora_enabled(self) -> bool:
        return self.lora_enable and self.stage == "instruct"

    @property
    def run_dir(self) -> str:
        return os.path.join(self.run_root, self.run_name) if self.run_name else self.run_root

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.run_dir, "ckpts")

    @property
    def lora_target_list(self) -> list[str]:
        return [t.strip() for t in self.lora_targets.split(",") if t.strip()]
