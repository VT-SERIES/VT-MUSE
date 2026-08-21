#!/usr/bin/env bash
set -euo pipefail

TRAIN_DATA="${1:?usage: train_stage1.sh TRAIN_DIR VAL_DIR [OUTPUT_DIR]}"
VAL_DATA="${2:?usage: train_stage1.sh TRAIN_DIR VAL_DIR [OUTPUT_DIR]}"
OUTPUT_DIR="${3:-checkpoints/stage1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m vt_muse.train \
  --train_data "${TRAIN_DATA}" \
  --val_data "${VAL_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --stage 1 \
  --epochs_stage1 30 \
  --lr_stage1 1e-4 \
  --batch_size "${BATCH_SIZE}" \
  --history_len 5 \
  --sample_stride 5 \
  --num_tail_frames 2 \
  --latent_dim 512 \
  --num_memory_layers 4 \
  --num_latent_layers 4 \
  --max_delta_t 32 \
  --tactile_temporal_mode raw \
  --no_action_conditioning \
  --contrastive_temp 0.07 \
  --stage1_cross_weight 1.0 \
  --stage1_temporal_weight 1.0 \
  --stage1_consistency_weight 1.0 \
  --ddp_contrastive_all_gather \
  --trainable_vit_layers 3
