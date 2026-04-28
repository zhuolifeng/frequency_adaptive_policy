#!/bin/bash

TASK="dual_switch"
DATASET_VERSION="v4.2"
ROUND="1"

# ============================================================
# Checkpoint path
# ============================================================
CKPT_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/frequency_adaptive_policy/data/outputs/2026.04.13/17.02.13_train_flow_matching_transformer_image_dual_switch_v4_fm_transformer_round_1/checkpoints/latest.ckpt"

# Dataset path
DATASET_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr"

# GPU device
GPU_DEVICE="cuda:0"

# Video settings
VIDEO_FPS=1

# Output directory for videos
OUTPUT_DIR="eval_fm_transformer_output_5step"

echo "=========================================="
echo "Flow Matching Transformer Policy Evaluation"
echo "=========================================="
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "GPU device: ${GPU_DEVICE}"
echo "Video FPS:  ${VIDEO_FPS}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

export CKPT_PATH="${CKPT_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"
python eval_fm.py \
    --config-name train_fm_workspace \
    task=real_${TASK}_image_wrench_offline_10fps \
    task.dataset_path=${DATASET_PATH} \
    +device=${GPU_DEVICE} \
    +video_fps=${VIDEO_FPS} \
    +use_full_dataset=True \
    task.dataset.val_ratio=0.0 \
    policy.obs_encoder.transforms='[{type: CenterCrop}]'

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
