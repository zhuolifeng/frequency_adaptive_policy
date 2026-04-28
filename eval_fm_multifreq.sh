#!/bin/bash

TASK="dual_switch"
DATASET_VERSION="v4.2"
ROUND="1"

CONFIG_NAME="train_fm_multifreq_workspace"

# Checkpoint path (update to actual path)
CKPT_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/frequency_adaptive_policy/data/outputs/2026.04.19/14.11.21_train_fm_multifreq_workspace_dual_switch_v4_fm_multifreq/checkpoints/latest.ckpt"

# Dataset path
DATASET_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr"

# GPU device
GPU_DEVICE="cuda:0"

# Video settings
VIDEO_FPS=1
NUM_INFERENCE_STEPS=1
USE_DISTILLATION=True

# Output directory
OUTPUT_DIR="evaluations_fm_multifreq_stage2_1step_standard"

echo "=========================================="
echo "FM MultiFreq Policy Evaluation"
echo "=========================================="
echo "Config:     ${CONFIG_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "GPU device: ${GPU_DEVICE}"
echo "Inference steps: ${NUM_INFERENCE_STEPS}"
echo "Use distillation: ${USE_DISTILLATION}"
echo "Video FPS:  ${VIDEO_FPS}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

export CKPT_PATH="${CKPT_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"
python eval_fm_multifreq.py \
    --config-name ${CONFIG_NAME} \
    task=real_${TASK}_image_wrench_offline_10fps_acp \
    task.dataset_path=${DATASET_PATH} \
    +use_full_dataset=True \
    +num_inference_steps=${NUM_INFERENCE_STEPS} \
    +use_distillation=${USE_DISTILLATION} \
    +video_fps=${VIDEO_FPS} \
    +device=${GPU_DEVICE} \
    task.dataset.val_ratio=0.0 \
    'policy.obs_encoder.transforms=[{type: CenterCrop}]'

echo ""
echo "=========================================="
echo "Complete!"
