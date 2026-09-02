#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)" # repo root 가정
NPROC="${NPROC:-4}"

RUN_NAME="${RUN_NAME:-klava_instruct_784_r64}"
DATA_MANIFEST="${DATA_MANIFEST:-$ROOT/data/vlm/stage2_instruct.jsonl}"
EXAONE_ID="${EXAONE_ID:-LGAI-EXAONE/EXAONE-4.0-1.2B}"

VISION_BACKBONE="${VISION_BACKBONE:-siglip2}" # siglip2(SigLIP-2 So400m NaFlex)만 지원
SIGLIP_MODEL_ID="${SIGLIP_MODEL_ID:-google/siglip2-so400m-patch16-naflex}"
SIGLIP_NUM_PATCHES="${SIGLIP_NUM_PATCHES:-784}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-klava_align_784}" # vlm_pretrain.sh의 RUN_NAME 기본값과 일치 필요
STAGE1_DIR="${STAGE1_DIR:-$ROOT/runs/$STAGE1_RUN_NAME/ckpts/fin}"
PROJECTOR_INIT="${PROJECTOR_INIT:-$STAGE1_DIR/projector.safetensors}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}" # eff batch size -> 128
NUM_WORKERS="${NUM_WORKERS:-12}"
EPOCHS="${EPOCHS:-1}"
PROJECTOR_LR="${PROJECTOR_LR:-2e-5}"
LM_LORA_LR="${LM_LORA_LR:-1e-4}"
TRAIN_VISION="${TRAIN_VISION:-true}"
VISION_LR="${VISION_LR:-2e-6}"

source "$ROOT/.venv/bin/activate"
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTHONPATH="$ROOT/src"
MULTI=""; [ "$NPROC" -gt 1 ] && MULTI="--multi_gpu"

echo "[vlm_finetune_lora.sh] stage=instruct(LoRA r64/a128) NPROC=$NPROC manifest=$DATA_MANIFEST"
accelerate launch --num_processes "$NPROC" $MULTI --mixed_precision bf16 \
  -m kava.train.vlm \
  --stage instruct --lora_enable true --lora_r 64 --lora_alpha 128 \
  --exaone_id "$EXAONE_ID" \
  --vision_backbone "$VISION_BACKBONE" --siglip_model_id "$SIGLIP_MODEL_ID" \
  --siglip_num_patches "$SIGLIP_NUM_PATCHES" \
  --data_manifest "$DATA_MANIFEST" --run_name "$RUN_NAME" \
  --img_size "${IMG_SIZE:-576}" --max_len "${MAX_LEN:-4096}" \
  --micro_batch_size "$MICRO_BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM_STEPS" --num_workers "$NUM_WORKERS" \
  --projector_init "$PROJECTOR_INIT" \
  --epochs "$EPOCHS" --projector_lr "$PROJECTOR_LR" --lm_lora_lr "$LM_LORA_LR" \
  --save_every_steps "${SAVE_EVERY_STEPS:-2000}" \
  --warmup_ratio 0.03 --grad_clip 1.0 \
  --train_vision "$TRAIN_VISION" --vision_lr "$VISION_LR" --vision_bn_mode "${VISION_BN_MODE:-freeze}" \
  --param_dtype bfloat16 --activation_checkpointing true --use_wandb "${USE_WANDB:-true}" \
  --wandb_project "${WANDB_PROJECT:-kor_chatbot}" --wandb_entity "${WANDB_ENTITY:-kksoo1769}" "$@"