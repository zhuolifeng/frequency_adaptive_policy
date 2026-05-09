#!/bin/bash

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"

# This script is for FMMultiFreqShortcutPolicy:
#   10Hz head -> 5 inference steps
#   50Hz head -> 1 inference step
POLICY_TYPE="shortcut"
CONFIG_NAME="${CONFIG_NAME:-train_fm_multifreq_shortcut_workspace}"
if [ "${CONFIG_NAME}" != "train_fm_multifreq_shortcut_workspace" ]; then
    echo "This script expects CONFIG_NAME=train_fm_multifreq_shortcut_workspace."
    echo "Got CONFIG_NAME=${CONFIG_NAME}"
    exit 1
fi

# Checkpoint path. Override with CKPT_PATH=/path/to/checkpoint.ckpt.
CKPT_PATH="${CKPT_PATH:-/mnt/workspace/users/lifeng/frequency_adaptive_policy_flow/data/outputs/2026.05.08/15.49.50_train_fm_multifreq_shortcut_workspace_dual_switch_v6_fm_multifreq_shortcut/checkpoints/latest.ckpt}"

# Dataset path. Override with DATASET_PATH=/path/to/zarr.
DATASET_PATH="${DATASET_PATH:-/mnt/workspace/users/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"

GPU_DEVICE="${GPU_DEVICE:-cuda:6}"

VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS_50HZ="${NUM_INFERENCE_STEPS_50HZ:-1}"
NUM_INFERENCE_STEPS_10HZ="${NUM_INFERENCE_STEPS_10HZ:-5}"
SHORTCUT_VERSION="${SHORTCUT_VERSION:-old}"

OUTPUT_DIR="${OUTPUT_DIR:-evaluations_fm_multifreq_shortcut_10hz5_50hz1}"

echo "=========================================="
echo "FM MultiFreq Shortcut Policy Evaluation"
echo "=========================================="
echo "Policy type:       ${POLICY_TYPE}"
echo "Config:            ${CONFIG_NAME}"
echo "Checkpoint:        ${CKPT_PATH}"
echo "Dataset:           ${DATASET_PATH}"
echo "GPU device:        ${GPU_DEVICE}"
echo "10Hz steps:        ${NUM_INFERENCE_STEPS_10HZ}"
echo "50Hz steps:        ${NUM_INFERENCE_STEPS_50HZ}"
echo "Shortcut version:  ${SHORTCUT_VERSION}"
echo "Video FPS:         ${VIDEO_FPS}"
echo "Output dir:        ${OUTPUT_DIR}"
echo ""

export CKPT_PATH="${CKPT_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"

ARGS=(
    --config-name "${CONFIG_NAME}"
    "task=real_${TASK}_image_wrench_offline_10fps_acp"
    "task.dataset_path=${DATASET_PATH}"
    "+use_full_dataset=True"
    "+num_inference_steps=${NUM_INFERENCE_STEPS_50HZ}"
    "+num_inference_steps_10hz=${NUM_INFERENCE_STEPS_10HZ}"
    "policy.shortcut_version=${SHORTCUT_VERSION}"
    "+video_fps=${VIDEO_FPS}"
    "+device=${GPU_DEVICE}"
    "task.dataset.val_ratio=0.0"
    "policy.obs_encoder.transforms=[{type: CenterCrop}]"
)

python eval_fm_multifreq.py "${ARGS[@]}"

echo ""
echo "=========================================="
echo "Complete!"
