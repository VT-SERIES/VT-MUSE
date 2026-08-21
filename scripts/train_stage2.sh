#!/usr/bin/env bash
set -euo pipefail

TRAIN_DATA="${1:?usage: train_stage2.sh TRAIN_DIR VAL_DIR STAGE1_CKPT [OUTPUT_DIR]}"
VAL_DATA="${2:?usage: train_stage2.sh TRAIN_DIR VAL_DIR STAGE1_CKPT [OUTPUT_DIR]}"
STAGE1_CKPT="${3:?usage: train_stage2.sh TRAIN_DIR VAL_DIR STAGE1_CKPT [OUTPUT_DIR]}"
OUTPUT_DIR="${4:-checkpoints/stage2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m vt_muse.train \
  --train_data "${TRAIN_DATA}" \
  --val_data "${VAL_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --stage 2 \
  --resume "${STAGE1_CKPT}" \
  --epochs_stage2 20 \
  --lr_stage2 1e-4 \
  --batch_size "${BATCH_SIZE}" \
  --history_len 5 \
  --sample_stride 5 \
  --num_tail_frames 2 \
  --latent_dim 512 \
  --num_memory_layers 4 \
  --num_latent_layers 4 \
  --max_delta_t 32 \
  --tactile_temporal_mode raw \
  --tactile_flow_target_mode depth_delta \
  --depth_delta_clip 0.5 \
  --recon_weight 1.0 \
  --tactile_flow_weight 1.0 \
  --kl_weight 0.001 \
  --perceptual_weight 0.0 \
  --val_random_tail_mask \
  --no_action_conditioning \
  --stage2_reuse_frozen_encoder_tokens \
  --stage2_frozen_encoders_eval
