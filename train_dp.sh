#!/usr/bin/env bash
set -eo pipefail

export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29504}"
LOGGING_BACKEND="${LOGGING_BACKEND:-tensorboard}"
LOGGING_MODE="${LOGGING_MODE:-online}"

TASK="${TASK:-dual_box}"
export DATASET_PATH="${DATASET_PATH:-data/${TASK}_v5_round_1_zarr}"

# Train the diffusion policy
echo "Stage 1: training Policy..."
bash ./launch_training.sh \
    --config-name=train_diffusion_unet_real_image_workspace \
    "task=real_${TASK}_image_dp_10fps" \
    'task.dataset_path=${oc.env:DATASET_PATH}' \
    "task.name=real_${TASK}_image_dp_10fps_v2" \
    "logging.mode=${LOGGING_MODE}" \
    "logging.backend=${LOGGING_BACKEND}" \
    "$@"
