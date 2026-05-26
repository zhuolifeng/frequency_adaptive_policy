#!/bin/bash

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"

CONFIG_NAME="${CONFIG_NAME:-train_dp_multifreq_cd_50hz_workspace}"
CKPT_PATH="${CKPT_PATH:-/mnt/workspace/users/lifeng/frequency_adaptive_policy_flow/data/outputs/2026.05.24/dp_cd_sample/checkpoints/epoch=0070-train_loss=0.001.ckpt}"
DATASET_PATH="${DATASET_PATH:-/mnt/workspace/users/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"

GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS_10HZ="${NUM_INFERENCE_STEPS_10HZ:-5}"
NUM_INFERENCE_STEPS_50HZ="${NUM_INFERENCE_STEPS_50HZ:-1}"
PREDICTION_TYPE="${PREDICTION_TYPE:-sample}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations_dp_multifreq_cd_50hz_10hz${NUM_INFERENCE_STEPS_10HZ}steps_50hz${NUM_INFERENCE_STEPS_50HZ}steps_wo10hz}"

if [ -z "${CKPT_PATH}" ]; then
    echo "Error: set CKPT_PATH=/path/to/checkpoints/latest.ckpt"
    exit 1
fi

echo "=========================================="
echo "DP MultiFreq 50Hz CD Evaluation"
echo "=========================================="
echo "Config:          ${CONFIG_NAME}"
echo "Checkpoint:      ${CKPT_PATH}"
echo "Dataset:         ${DATASET_PATH}"
echo "GPU device:      ${GPU_DEVICE}"
echo "10Hz steps:      ${NUM_INFERENCE_STEPS_10HZ}"
echo "50Hz steps:      ${NUM_INFERENCE_STEPS_50HZ}"
echo "Prediction:      ${PREDICTION_TYPE}"
echo "Output dir:      ${OUTPUT_DIR}"
echo ""

export CKPT_PATH="${CKPT_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"

python eval_dp_multifreq_cd_50hz.py \
    --config-name "${CONFIG_NAME}" \
    "task=real_${TASK}_image_wrench_offline_10fps_acp" \
    "task.dataset_path=${DATASET_PATH}" \
    "prediction_type=${PREDICTION_TYPE}" \
    "+use_full_dataset=True" \
    "+num_inference_steps_10hz=${NUM_INFERENCE_STEPS_10HZ}" \
    "+num_inference_steps_50hz=${NUM_INFERENCE_STEPS_50HZ}" \
    "+video_fps=${VIDEO_FPS}" \
    "+device=${GPU_DEVICE}" \
    "task.dataset.val_ratio=0.0" \
    "policy.obs_encoder.transforms=[{type: CenterCrop}]" \
    "$@"

echo ""
echo "=========================================="
echo "Complete!"
