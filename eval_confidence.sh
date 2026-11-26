#!/usr/bin/env bash
set -eo pipefail

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.2}"
ROUND="${ROUND:-1}"
CONFIG_NAME="${CONFIG_NAME:-train_hybrid_diffusion_act_confidence_workspace}"
: "${CKPT_PATH:?Set CKPT_PATH to the policy checkpoint}"
export DATASET_PATH="${DATASET_PATH:-data/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations_hybrid_confidence}"

export CKPT_PATH
export OUTPUT_DIR
python eval_confidence.py \
    --config-name "${CONFIG_NAME}" \
    "task=real_${TASK}_image_wrench_offline_10fps_acp" \
    'task.dataset_path=${oc.env:DATASET_PATH}' \
    +eval_batch_size=64 \
    +use_full_dataset=True \
    "+num_inference_steps=${NUM_INFERENCE_STEPS}" \
    "+video_fps=${VIDEO_FPS}" \
    "+device=${GPU_DEVICE}" \
    task.dataset.val_ratio=0.0 \
    'policy.obs_encoder.transforms=[{type: CenterCrop}]' \
    "$@"
