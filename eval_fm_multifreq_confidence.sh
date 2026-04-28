#!/bin/bash

TASK="dual_switch"
DATASET_VERSION="v4.2"
ROUND="1"

CONFIG_NAME="train_fm_multifreq_confidence_workspace"

# Checkpoint path (update to actual path)
CKPT_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/frequency_adaptive_policy/data/outputs/2026.04.21/03.36.18_train_fm_multifreq_confidence_workspace_dual_switch_v4.1_fm_multifreq_confidence_stage3/checkpoints/latest.ckpt"

# Dataset path
DATASET_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr"

# GPU device
GPU_DEVICE="cuda:0"

# Video settings
VIDEO_FPS=1
NUM_INFERENCE_STEPS=5

# Output directory
OUTPUT_DIR="evaluations_fm_multifreq_confidence_900x400"

echo "=========================================="
echo "FM MultiFreq Confidence Policy Evaluation"
echo "=========================================="
echo "Config:     ${CONFIG_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "GPU device: ${GPU_DEVICE}"
echo "Inference steps: ${NUM_INFERENCE_STEPS}"
echo "Video FPS:  ${VIDEO_FPS}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

export CKPT_PATH="${CKPT_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"
python eval_fm_multifreq_confidence.py \
    --config-name ${CONFIG_NAME} \
    task=real_${TASK}_image_wrench_offline_10fps_acp \
    task.dataset_path=${DATASET_PATH} \
    +use_full_dataset=True \
    +num_inference_steps=${NUM_INFERENCE_STEPS} \
    +video_fps=${VIDEO_FPS} \
    +device=${GPU_DEVICE} \
    task.dataset.val_ratio=0.0 \
    'policy.obs_encoder.transforms=[{type: CenterCrop}]'

echo ""
echo "=========================================="
echo "Complete!"
