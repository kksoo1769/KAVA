"""정렬과 지시 학습 단계에서 사용하는 VLM 학습 코드."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, StateDictType
from torch.utils.data import DataLoader
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import set_seed
from safetensors.torch import save_file, load_file

from kava.device import max_memory_allocated_gib, synchronize
from kava.model_vlm import build_vlm
from kava.vision.siglip2 import Siglip2Config
from kava.train.vlm_config import VLMConfig
from kava.train.optim import make_lr_scheduler
from kava.data.tokenizer import load_tokenizer
from kava.data.vlm import load_manifest, VLMDataset, collate_vlm


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
TRAINER_STATE_FILE = "trainer_state.pt"

def _str2bool(s):
    return str(s).lower() in ("1", "true", "yes")

def parse_option() -> VLMConfig:
    cfg = VLMConfig()
    parser = argparse.ArgumentParser()
    for field, val in vars(cfg).items():
        if isinstance(val, bool):
            parser.add_argument(f"--{field}", type=_str2bool, default=val)
        else:
            parser.add_argument(f"--{field}", type=type(val), default=val)
    args = parser.parse_args()
    for field in vars(cfg):
        setattr(cfg, field, getattr(args, field))
    return cfg

def build_lora_config(cfg: VLMConfig):
    from peft import LoraConfig
    return LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_list, bias="none", task_type="CAUSAL_LM",
    )

def make_optimizer(vlm, cfg: VLMConfig, fused: bool, train_vision: bool):
    """projector(LoRA, vision encoder) 학습 파라미터로 param group 구성."""
    proj_params = [p for p in vlm.connector.parameters() if p.requires_grad]

    groups = []
    proj_lr = cfg.projector_lr if cfg.stage == 'instruct' else cfg.base_lr
    groups.append({"params": proj_params, "lr": proj_lr})
    if cfg.stage == 'instruct':
        if cfg.lora_enabled: # LM LoRA
            lora_params = [p for p in vlm.language_model.parameters() if p.requires_grad]
            groups.append({"params": lora_params, "lr": cfg.lm_lora_lr})
        else:
            lm_params = [p for p in vlm.language_model.parameters() if p.requires_grad]
            groups.append({"params": lm_params, "lr": cfg.lm_lr})
    if train_vision: # vision encoder
        vis_params = [p for p in vlm.vision_encoder.backbone.parameters() if p.requires_grad]
        groups.append({"params": vis_params, "lr": cfg.vision_lr})
    return torch.optim.AdamW(
        groups, betas=(cfg.beta1, cfg.beta2), eps=1e-8,
        weight_decay=cfg.weight_decay, fused=fused
    )

def save_ckpt(
    accelerator, vlm, cfg: VLMConfig, tag: str, step: int,
    optimizer=None, scheduler=None, epoch: int = 0, wandb_id: str | None = None
):
    accelerator.wait_for_everyone()

    out = os.path.join(cfg.ckpt_dir, tag)
    os.makedirs(out, exist_ok=True)

    if cfg.lm_mode == "full": # 샤딩 통합
        full_state_dict = accelerator.get_state_dict(vlm)
        if accelerator.is_main_process:
            # projector
            proj_state_dict = {
                k[len("connector."):]: v.detach().to(torch.float32).cpu().clone()
                for k, v in full_state_dict.items() if k.startswith("connector.")
            }
            save_file(proj_state_dict, os.path.join(out, "projector.safetensors"))

            # LM
            lm_state_dict = {
                k[len("language_model."):]: v.detach().to(torch.float32).cpu().clone()
                for k, v in full_state_dict.items() if k.startswith("language_model.")
            }
            save_file(lm_state_dict, os.path.join(out, "language_model.safetensors"))

            # vision encoder
            if cfg.train_vision:
                vis_state_dict = {
                    k[len("vision_encoder.backbone."):]: v.detach().to(torch.float32).cpu().clone()
                    for k, v in full_state_dict.items() if k.startswith("vision_encoder.backbone.")
                }
                save_file(vis_state_dict, os.path.join(out, "vision_encoder.safetensors"))
            
            meta = {
                "stage": cfg.stage, "exaone_id": cfg.exaone_id,
                "vision_backbone": cfg.vision_backbone,
                # SigLIP-2 재구성용.
                "siglip_model_id": cfg.siglip_model_id, "siglip_num_patches": cfg.siglip_num_patches,
                # LM
                "lm_mode": cfg.lm_mode,
                "lora": cfg.lora_enabled, "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha,
                "lora_targets": cfg.lora_target_list if cfg.lora_enabled else [],
                "img_size": cfg.img_size, "vision_hidden": vlm.vision_encoder.hidden_size,
                "train_vision": cfg.train_vision, "vision_encoder_finetuned": cfg.train_vision,
                "step": step, "data_manifest": cfg.data_manifest,
            }
            json.dump(meta, open(os.path.join(out, "meta.json"), "w"), ensure_ascii=False, indent=4)
            
        accelerator.save_state(os.path.join(out, "accel_state")) # optimizer, scheduler, rng 포함
        synchronize(str(accelerator.device))
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            json.dump(
                {"global_step": step, "epoch": epoch, "wandb_id": wandb_id,},
                open(os.path.join(out, "resume.json"), "w"),
            )
        return

    if not accelerator.is_main_process: # main process에서만 저장
        return
    unwrapped = accelerator.unwrap_model(vlm)

    # projector(fp32로 저장)
    proj_state_dict = {
        k: v.detach().to(torch.float32).cpu()
        for k, v in unwrapped.connector.state_dict().items()
    }
    save_file(proj_state_dict, os.path.join(out, "projector.safetensors"))

    # LoRA adapter(stage 2)
    if cfg.stage == 'instruct':
        if cfg.lora_enabled:
            unwrapped.language_model.save_pretrained(os.path.join(out, "adapter"))
        
    # Vision encoder
    train_vision = cfg.train_vision and cfg.stage == "instruct"
    if train_vision:
        vis_state_dict = {
            k: v.detach().to(torch.float32).cpu()
            for k, v in unwrapped.vision_encoder.backbone.state_dict().items()
        }
        save_file(vis_state_dict, os.path.join(out, "vision_encoder.safetensors"))

    meta = {
        "stage": cfg.stage, "exaone_id": cfg.exaone_id,
        "vision_backbone": cfg.vision_backbone,
        # SigLIP-2 재구성용.
        "siglip_model_id": cfg.siglip_model_id, "siglip_num_patches": cfg.siglip_num_patches,
        "lora": cfg.lora_enabled, "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha,
        "lora_targets": cfg.lora_target_list if cfg.lora_enabled else [],
        "img_size": cfg.img_size,
        "vision_hidden": unwrapped.vision_encoder.hidden_size,
        "train_vision": train_vision, "vision_encoder_finetuned": train_vision,
        "step": step, "data_manifest": cfg.data_manifest,
    }
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), ensure_ascii=False, indent=4)
    # 모델 가중치와 별도로 학습 재개에 필요한 상태를 저장한다
    if optimizer is not None and scheduler is not None:
        trainer_state = {
            "global_step": step, "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "wandb_id": wandb_id,
            # 재개 시 데이터, 스케줄 정합 검증용
            "grad_accum_steps": cfg.grad_accum_steps, "micro_batch_size": cfg.micro_batch_size,
            "num_processes": accelerator.num_processes, "epochs": cfg.epochs,
        }
        torch.save(trainer_state, os.path.join(out, TRAINER_STATE_FILE))
    tags = (
        "projector"
        + (" + adapter" if cfg.lora_enabled else "") + (" + vision" if train_vision else "")
        + (" + trainer_state" if optimizer is not None else "")
    )
    print(f"[ckpt] saved: {out} ({tags})", flush=True)


def load_resume_weights(vlm, cfg: VLMConfig, dtype, is_main: bool) -> dict:
    """resume ckpt 디렉터리에서 학습 대상 가중치(projector, vision, adapter)를 로드."""
    resume_dir = cfg.resume
    if not os.path.isdir(resume_dir):
        raise FileNotFoundError(f"--resume 경로가 디렉터리가 아닙니다: {resume_dir}")

    meta = {}
    meta_path = os.path.join(resume_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    # 아키텍처 정합 검증
    for key, cur in (("vision_backbone", cfg.vision_backbone), ("stage", cfg.stage),
                     ("lora_r", cfg.lora_r), ("lora_alpha", cfg.lora_alpha)):
        if key in meta and meta[key] != cur:
            raise ValueError(f"[Resume] '{key}' 불일치: ckpt={meta[key]} vs cfg={cur}. 동일 설정으로 재개하세요.")

    # projector
    proj_path = os.path.join(resume_dir, "projector.safetensors")
    if os.path.exists(proj_path):
        vlm.connector.load_state_dict({k: v.to(dtype) for k, v in load_file(proj_path).items()}, strict=True)

    # vision backbone(stage 2 train_vision일 때만 저장). 만약 있으면 기존 체크포인트 대신 stage 2의 가중치로 덮어씀
    vision_path = os.path.join(resume_dir, "vision_encoder.safetensors")
    if os.path.exists(vision_path):
        vision_dtype = next(vlm.vision_encoder.backbone.parameters()).dtype
        vlm.vision_encoder.backbone.load_state_dict(
            {k: v.to(vision_dtype) for k, v in load_file(vision_path).items()}, strict=True)
    elif meta.get("vision_encoder_finetuned") or meta.get("train_vision"):
        if is_main:
            raise RuntimeError(
                f"[Resume 경고] meta엔 vision 미세조정 기록이 있으나 {vision_path} 없음: "
                f"vision 백본이 미세조정 이전 상태로 복원됨(진행분 유실 가능)."
            )
        
    # LoRA adapter
    adapter_dir = os.path.join(resume_dir, "adapter")
    if cfg.lora_enabled and os.path.isdir(adapter_dir):
        from peft import load_peft_weights, set_peft_model_state_dict
        adapter_device = str(next(vlm.language_model.parameters()).device)
        incompat = set_peft_model_state_dict(
            vlm.language_model, load_peft_weights(adapter_dir, device=adapter_device))
        miss = list(getattr(incompat, "missing_keys", []) or [])
        lora_missing = [k for k in miss if "lora_" in k]
        if lora_missing:
            raise RuntimeError(
                f"[Resume 경고] adapter LoRA 키 누락 {len(lora_missing)}개(예: {lora_missing[:3]}). "
                f"lora_targets, r, alpha가 ckpt와 다를 수 있음."
            )
    elif cfg.lora_enabled and is_main:
        raise RuntimeError(f"[Resume 경고] adapter 디렉터리 없음: {adapter_dir} (LoRA가 랜덤에서 시작됨)", flush=True)
    if is_main:
        print(f"[Resume] weights: {resume_dir} (saved step={meta.get('step')})", flush=True)
    return meta


def load_resume_trainer_state(cfg: VLMConfig, optimizer, scheduler, accelerator, is_main: bool):
    """trainer_state.pt에서 optimizer, scheduler, step, RNG를 복원. 반환: (global_step, epoch, wandb_id)."""
    trainer_state_path = os.path.join(cfg.resume, TRAINER_STATE_FILE)
    if not os.path.exists(trainer_state_path):
        if is_main:
            print(
                f"[Resume 경고] {TRAINER_STATE_FILE} 없음: optimizer, step 복원 불가. "
                f"가중치만 이어받고 step 0, 새 optimizer로 시작합니다.",
                flush=True
            )
        return 0, 0, None
    state_dict = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
    # 스케줄, 데이터 정합 검증
    for key, cur in (("grad_accum_steps", cfg.grad_accum_steps), ("num_processes", accelerator.num_processes), ("epochs", cfg.epochs)):
        saved = state_dict.get(key)
        if saved is not None and saved != cur and is_main:
            print(
                f"[Resume 경고] '{key}' 불일치: ckpt={saved} vs 현재={cur}. "
                f"LR 스케줄과 데이터 위치가 원본과 달라질 수 있음.",
                flush=True
            )
    # optimizer, scheulder, RNG 로드
    optimizer.load_state_dict(state_dict["optimizer"])
    scheduler.load_state_dict(state_dict["scheduler"])
    if state_dict.get("torch_rng") is not None:
        torch.set_rng_state(state_dict["torch_rng"])
    if state_dict.get("cuda_rng") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state_dict["cuda_rng"])
        except Exception:
            pass
    return int(state_dict.get("global_step", 0)), int(state_dict.get("epoch", 0)), state_dict.get("wandb_id")


def main():
    cfg = parse_option()
    dtype = _DTYPES[cfg.param_dtype]
    fsdp_plugin = None
    if cfg.lm_mode == "full":
        fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            reshard_after_forward=True, # ZeRO-3
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["Exaone4DecoderLayer"],
            mixed_precision_policy=MixedPrecisionPolicy(
                param_dtype=dtype,
                reduce_dtype=torch.float32
            ),
            state_dict_type=StateDictType.FULL_STATE_DICT,
        )
    accelerator = Accelerator(
        mixed_precision=(
            "bf16" if cfg.param_dtype == "bfloat16"
            else "fp16" if cfg.param_dtype == "float16" else "no"
        ),
        gradient_accumulation_steps=cfg.grad_accum_steps,
        log_with=("wandb" if cfg.use_wandb else None),
        project_dir=cfg.run_dir,
        fsdp_plugin=fsdp_plugin,
    )
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    is_main = accelerator.is_main_process

    log_fh = None
    if is_main:
        # run_dir(runs/<run_name>) 경로에 산출물 정리
        os.makedirs(cfg.run_dir, exist_ok=True)
        with open(os.path.join(cfg.run_dir, "config.json"), "w") as f: # config 정리
            json.dump({**vars(cfg), "run_dir": cfg.run_dir, "ckpt_dir": cfg.ckpt_dir, "num_processes": accelerator.num_processes},
                f, indent=4, ensure_ascii=False
            )
        log_fh = open(os.path.join(cfg.run_dir, "log.jsonl"), "a", buffering=1) # log
        print(f"[VLM] outputs > {cfg.run_dir}/(config.json, log.jsonl, ckpts/)", flush=True)

    tokenizer = load_tokenizer(cfg.tokenizer_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    lora_config = build_lora_config(cfg) if cfg.lora_enabled else None
    vlm = build_vlm(
        exaone_id=cfg.exaone_id,
        vision_backbone=cfg.vision_backbone,
        siglip_cfg=Siglip2Config(
            model_id=cfg.siglip_model_id,
            max_num_patches=cfg.siglip_num_patches,
        ),
        lora_config=lora_config,
        lm_master_dtype=torch.float32 if cfg.lm_mode == 'full' else torch.bfloat16,
        vis_master_dtype=torch.float32 if cfg.train_vision else torch.bfloat16,
        con_master_dtype=torch.float32,
        device=str(accelerator.device), attn_implementation=cfg.attn_implementation, verbose=is_main,
    )
    if is_main:
        print(
            f"[VLM] vision backbone=SigLIP-2 NaFlex: id={cfg.siglip_model_id} "
            f"vision_hidden={vlm.vision_encoder.hidden_size} seq_len={cfg.siglip_num_patches}",
            flush=True
        )

    # Stage 1에서 학습한 projector 로드
    if cfg.projector_init:
        state_dict = {k: v.to(dtype) for k, v in load_file(cfg.projector_init).items()}
        vlm.connector.load_state_dict(state_dict, strict=True)
        if is_main:
            print(f"[VLM] projector_init: {cfg.projector_init} (keys={len(state_dict)})", flush=True)

    # Stage 2에서 사전 학습된 projector를 지정하지 않은 경우 경고 출력
    if cfg.stage == "instruct" and not cfg.projector_init and not cfg.resume and is_main:
        print("[VLM] WARNING: instruct without --projector_init: projector starts RANDOM ", flush=True)

    # resume: 저장된 ckpt의 학습 가중치로 위 stage 1 init을 덮어써 이어 학습 준비(가중치만, optimizer는 뒤에서)
    resume_meta = load_resume_weights(vlm, cfg, dtype, is_main) if cfg.resume and cfg.lm_mode != "full" else None

    train_vision = cfg.train_vision and cfg.stage == "instruct"
    vlm.set_trainable(
        train_lm=(False if cfg.lm_mode == 'frozen' else None if cfg.lm_mode == 'lora' else True), train_projector=True,
        train_vision=train_vision
    )
    if cfg.activation_checkpointing:
        vlm.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if cfg.lora_enabled:
            vlm.language_model.enable_input_require_grads() # frozen base+adapter에서 grad 전파 보장
        if train_vision:
            vlm.vision_encoder.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    records = load_manifest(cfg.data_manifest)
    ds = VLMDataset(
        records, cfg.tokenizer_name, stage=cfg.stage, img_size=cfg.img_size,
        max_len=cfg.max_len, train_aug=cfg.train_aug,
        add_default_system=cfg.add_default_system,
        vision_backbone=cfg.vision_backbone, siglip_model_id=cfg.siglip_model_id,
        max_num_patches=cfg.siglip_num_patches
    )
    # 셔플 재현성용 generator(각 프로세스에서 동일 seed: resume 시 skip_first_batches가 정확히 사용 부분만 건너뜀).
    data_gen = torch.Generator()
    data_gen.manual_seed(cfg.seed)
    loader = DataLoader(
        ds, batch_size=cfg.micro_batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        generator=data_gen,
        collate_fn=lambda b: collate_vlm(b, pad_id),
    )

    optimizer = make_optimizer(
        vlm, cfg, train_vision=train_vision,
        fused=(cfg.fused_optimizer and accelerator.device.type == "cuda")
    )

    # model, optimizer, data loader 준비
    vlm, optimizer, loader = accelerator.prepare(vlm, optimizer, loader)
    nproc = accelerator.num_processes
    opt_steps_per_epoch = max(1, math.ceil(len(loader) / cfg.grad_accum_steps))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else cfg.epochs * opt_steps_per_epoch
    warmup = int(cfg.warmup_ratio * total_steps)
    scheduler = make_lr_scheduler(optimizer, warmup * nproc, total_steps * nproc, cfg.min_lr_ratio)
    scheduler = accelerator.prepare(scheduler)

    # resume 지점 계산(optimizer, scheduler, step)
    resume_global_step, start_epoch, resume_skip_micro, resume_wandb_id = 0, 0, 0, None
    if cfg.resume:
        if cfg.lm_mode == "full":
            accelerator.load_state(os.path.join(cfg.resume, "accel_state"))
            resume_json = json.load(open(os.path.join(cfg.resume, "resume.json")))
            resume_global_step = int(resume_json["global_step"])
            resume_wandb_id = resume_json.get("wandb_id")
        else:
            resume_global_step, _saved_epoch, resume_wandb_id = load_resume_trainer_state(
            cfg, optimizer, scheduler, accelerator, is_main)
        start_epoch = resume_global_step // opt_steps_per_epoch
        resume_skip_micro = (resume_global_step % opt_steps_per_epoch) * cfg.grad_accum_steps
        if is_main:
            print(
                f"[Resume] global_step={resume_global_step}: start_epoch={start_epoch} "
                f"skip_micro_batches={resume_skip_micro}",
                flush=True
            )

    wandb_run_id = None
    if cfg.use_wandb:
        wb = {"name": cfg.wandb_run_name or cfg.run_name, "entity": cfg.wandb_entity, "dir": cfg.run_dir}
        if resume_wandb_id: # 같은 wandb run에 이어 로깅
            wb["id"] = resume_wandb_id
            wb["resume"] = "allow"
        accelerator.init_trackers(cfg.wandb_project, config=vars(cfg), init_kwargs={"wandb": wb})
        try: # 이후 체크포인트에 저장해 재개 시 run 연속성 유지
            wandb_run_id = accelerator.get_tracker("wandb").run.id
        except Exception:
            wandb_run_id = resume_wandb_id

    trainable = [p for p in vlm.parameters() if p.requires_grad]
    if is_main:
        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in vlm.parameters())
        global_batch = cfg.micro_batch_size * cfg.grad_accum_steps * nproc
        print(
                f"[VLM-{cfg.stage}] lora={cfg.lora_enabled} trainable={n_train/1e9:.3f}B, "
            f"total={n_total/1e9:.3f}B | #data={len(records):,} | nproc={nproc} | "
            f"global_batch={global_batch} ({cfg.micro_batch_size} micro × {cfg.grad_accum_steps} accum × {nproc} GPUs) | "
            f"opt_steps/epoch={opt_steps_per_epoch} total_steps={total_steps} warmup={warmup}",
            flush=True
        )

    vlm.train()
    # vlm.train()이 vision encoder까지 train()으로 되돌리므로 단계별 freeze을 재적용
    accelerator.unwrap_model(vlm).set_trainable(
        train_lm=(False if cfg.lm_mode == 'frozen' else None if cfg.lm_mode == 'lora' else True),
        train_projector=True, train_vision=train_vision
    )

    global_step = resume_global_step
    t0 = time.perf_counter()
    stop = False
    epoch = start_epoch # 루프가 한 번도 안 돌 때(완료된 run 재개)의 fin 저장용 기본값
    for epoch in range(start_epoch, cfg.epochs):
        # 처음 실행과 재개 모두 같은 셔플 순서를 사용한다
        data_gen.manual_seed(cfg.seed + epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch) # accelerate seedable sampler
        epoch_loader = loader
        if epoch == start_epoch and resume_skip_micro > 0:
            # 이미 사용한 micro-batch만 sampler 레벨에서 건너뜀
            epoch_loader = accelerator.skip_first_batches(loader, resume_skip_micro)
            if is_main:
                print(f"[Resume] epoch {epoch}: skip_first_batches({resume_skip_micro})", flush=True)
        for batch in epoch_loader:
            with accelerator.accumulate(vlm):
                out = vlm(
                    batch["input_ids"], batch["attention_mask"],
                    labels=batch["labels"], pixel_values=batch["pixel_values"],
                    pixel_attention_mask=batch.get("pixel_attention_mask"),
                    spatial_shapes=batch.get("spatial_shapes")
                )
                loss = out["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(trainable, cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % cfg.log_every == 0:
                    loss_g = accelerator.gather(loss.detach()).mean().item()
                    lr = scheduler.get_last_lr()[0]
                    if is_main:
                        dt = time.perf_counter() - t0
                        print(
                            f"Step {global_step}/{total_steps} Loss {loss_g:.4f} "
                            f"LR {lr:.2e} Grad Norm {float(grad_norm):.3f} "
                            f"{cfg.log_every/dt:.2f} it/s {max_memory_allocated_gib(str(accelerator.device)):.1f} GiB",
                            flush=True
                        )
                        rec = {
                            "step": global_step, "loss": loss_g, "lr": lr,
                            "grad_norm": float(grad_norm), "epoch": epoch
                        }
                        if log_fh:
                            log_fh.write(json.dumps(rec) + "\n")
                        if cfg.use_wandb:
                            accelerator.log(rec, step=global_step)
                        t0 = time.perf_counter()
                if cfg.save_every_steps > 0 and global_step % cfg.save_every_steps == 0:
                    save_ckpt(accelerator, vlm, cfg, "last", global_step, # 덮어쓰기
                              optimizer=optimizer, scheduler=scheduler, epoch=epoch, wandb_id=wandb_run_id)
                if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                    stop = True
                    break
        if cfg.save_every_epoch and not stop:
            save_ckpt(accelerator, vlm, cfg, f"epoch_{epoch}", global_step,
                      optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1, wandb_id=wandb_run_id)
        if stop:
            break

    save_ckpt(accelerator, vlm, cfg, "fin", global_step,
              optimizer=optimizer, scheduler=scheduler, epoch=epoch, wandb_id=wandb_run_id)
    if cfg.use_wandb:
        accelerator.end_training()
    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()
