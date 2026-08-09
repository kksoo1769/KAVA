#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)" # repo root 가정
NPROC="${NPROC:-4}"

RUN_NAME="${RUN_NAME:-klava_align_784}"
DATA_MANIFEST="${DATA_MANIFEST:-$ROOT/data/vlm/align_stage1.train.jsonl}"
EXAONE_ID="${EXAONE_ID:-LGAI-EXAONE/EXAONE-4.0-1.2B}"

VISION_BACKBONE="${VISION_BACKBONE:-siglip2}" # siglip2(SigLIP-2 So400m NaFlex)만 지원
SIGLIP_MODEL_ID="${SIGLIP_MODEL_ID:-google/siglip2-so400m-patch16-naflex}"
SIGLIP_NUM_PATCHES="${SIGLIP_NUM_PATCHES:-784}" # 시각 토큰 시퀀스 길이(siglip2 전용)

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}" # effctive batch size: 256
NUM_WORKERS="${NUM_WORKERS:-12}"
EPOCHS="${EPOCHS:-1}"
BASE_LR="${BASE_LR:-1e-3}" # projector lr

source "$ROOT/.venv/bin/activate"
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTHONPATH="$ROOT/src"
MULTI=""; [ "$NPROC" -gt 1 ] && MULTI="--multi_gpu"

echo "[vlm_pretrain.sh] stage=align NPROC=$NPROC manifest=$DATA_MANIFEST"
accelerate launch --num_processes "$NPROC" $MULTI --mixed_precision bf16 -m kor_chat.train.vlm \
  --stage align --exaone_id "$EXAONE_ID" \
  --vision_backbone "$VISION_BACKBONE" --siglip_model_id "$SIGLIP_MODEL_ID" \
  --siglip_num_patches "$SIGLIP_NUM_PATCHES" \
  --data_manifest "$DATA_MANIFEST" --run_name "$RUN_NAME" \
  --img_size "${IMG_SIZE:-384}" --max_len 4096 \
  --micro_batch_size "$MICRO_BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM_STEPS" --num_workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" --base_lr "$BASE_LR" --warmup_ratio 0.03 --grad_clip 1.0 \
  --param_dtype bfloat16 --use_wandb "${USE_WANDB:-true}" \
  --wandb_project "${WANDB_PROJECT:-kor_chatbot}" --wandb_entity "${WANDB_ENTITY:-kksoo1769}" "$@"