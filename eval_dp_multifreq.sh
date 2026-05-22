#!/bin/bash

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"

CONFIG_NAME="${CONFIG_NAME:-train_dp_multifreq_workspace}"

# Checkpoint path. Override with CKPT_PATH=/path/to/checkpoint.ckpt.
CKPT_PATH="${CKPT_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/right/frequency_adaptive_policy/data/outputs/2026.04.30/17.25.24_train_dp_multifreq_workspace_dual_switch_v6_dp_multifreq/checkpoints/latest.ckpt}"

# Dataset path. Override with DATASET_PATH=/path/to/zarr.
DATASET_PATH="${DATASET_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"

GPU_DEVICE="${GPU_DEVICE:-cuda:2}"
VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-2}"
PREDICTION_TYPE="${PREDICTION_TYPE:-v_prediction}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations_dp_multifreq_2steps}"

echo "=========================================="
echo "DP MultiFreq Policy Evaluation"
echo "=========================================="
echo "Config:        ${CONFIG_NAME}"
echo "Checkpoint:    ${CKPT_PATH}"
echo "Dataset:       ${DATASET_PATH}"
echo "GPU device:    ${GPU_DEVICE}"
echo "Inference steps: ${NUM_INFERENCE_STEPS}"
echo "Prediction:    ${PREDICTION_TYPE}"
echo "Video FPS:     ${VIDEO_FPS}"
echo "Output dir:    ${OUTPUT_DIR}"
echo ""

export CKPT_PATH="${CKPT_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"

python eval_dp_multifreq.py \
    --config-name "${CONFIG_NAME}" \
    "task=real_${TASK}_image_wrench_offline_10fps_acp" \
    "task.dataset_path=${DATASET_PATH}" \
    "prediction_type=${PREDICTION_TYPE}" \
    "+use_full_dataset=True" \
    "+num_inference_steps=${NUM_INFERENCE_STEPS}" \
    "+video_fps=${VIDEO_FPS}" \
    "+device=${GPU_DEVICE}" \
    "task.dataset.val_ratio=0.0" \
    "policy.obs_encoder.transforms=[{type: CenterCrop}]" \
    "$@"

echo ""
echo "=========================================="
echo "Complete!"
