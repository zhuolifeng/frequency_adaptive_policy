#!/usr/bin/env bash
set -eo pipefail

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"
CONFIG_NAME="${CONFIG_NAME:-train_act_30hz_workspace}"
TASK_CONFIG="${TASK_CONFIG:-real_${TASK}_image_wrench_offline_10fps_30hz}"

: "${CKPT_PATH:?Set CKPT_PATH to the policy checkpoint}"
export DATASET_PATH="${DATASET_PATH:-data/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
VIDEO_FPS="${VIDEO_FPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations_act_30hz}"
MAX_EPISODES="${MAX_EPISODES:-10}"

echo "=========================================="
echo "ACT 30Hz evaluation"
echo "  Config:     ${CONFIG_NAME}"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  Dataset:    ${DATASET_PATH}"
echo "  Device:     ${GPU_DEVICE}"
echo "  Output:     ${OUTPUT_DIR}"
echo "=========================================="

export CKPT_PATH
export OUTPUT_DIR

python eval_act_30hz.py \
    --config-name "${CONFIG_NAME}" \
    "task=${TASK_CONFIG}" \
    'task.dataset_path=${oc.env:DATASET_PATH}' \
    +use_full_dataset=True \
    "+video_fps=${VIDEO_FPS}" \
    "+device=${GPU_DEVICE}" \
    "+max_episodes=${MAX_EPISODES}" \
    task.dataset.val_ratio=0.0 \
    "policy.obs_encoder.transforms=[{type: CenterCrop}]" \
    "$@"
