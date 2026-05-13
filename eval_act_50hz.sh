#!/bin/bash

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"
CONFIG_NAME="${CONFIG_NAME:-train_act_50hz_workspace}"
TASK_CONFIG="${TASK_CONFIG:-real_${TASK}_image_wrench_offline_10fps_50hz}"

CKPT_PATH="${CKPT_PATH:-/mnt/workspace/users/lifeng/frequency_adaptive_policy_flow/data/outputs/2026.05.12/23.16.03_train_act_50hz_workspace_dual_switch_v6_act_50hz/checkpoints/latest.ckpt}"
DATASET_PATH="${DATASET_PATH:-/mnt/workspace/users/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
VIDEO_FPS="${VIDEO_FPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations_act_50hz}"
MAX_EPISODES="${MAX_EPISODES:-10}"

echo "=========================================="
echo "ACT 50Hz evaluation"
echo "  Config:     ${CONFIG_NAME}"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  Dataset:    ${DATASET_PATH}"
echo "  Device:     ${GPU_DEVICE}"
echo "  Output:     ${OUTPUT_DIR}"
echo "=========================================="

export CKPT_PATH="${CKPT_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"

python eval_act_50hz.py \
    --config-name ${CONFIG_NAME} \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    +use_full_dataset=True \
    +video_fps=${VIDEO_FPS} \
    +device=${GPU_DEVICE} \
    +max_episodes=${MAX_EPISODES} \
    task.dataset.val_ratio=0.0 \
    "policy.obs_encoder.transforms=[{type: CenterCrop}]"

echo ""
echo "=========================================="
echo "Complete!"
