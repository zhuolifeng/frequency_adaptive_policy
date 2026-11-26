#!/usr/bin/env bash
set -eo pipefail

TASK="${TASK:-dual_box}"
DATASET_VERSION="${DATASET_VERSION:-v1}"
ROUND="${ROUND:-1}"
DATASET_TAG="${TASK}_${DATASET_VERSION}_round_${ROUND}"
export DATASET_PATH="${DATASET_PATH:-data/${DATASET_TAG}_zarr}"
TASK_CONFIG="${TASK_CONFIG:-real_${TASK}_image_wrench_offline_10fps_acp}"

USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
POLICY_LR="${POLICY_LR:-1.0e-4}"
LOGGING_BACKEND="${LOGGING_BACKEND:-tensorboard}"
LOGGING_MODE="${LOGGING_MODE:-online}"

# Stage 1: Train Policy
echo "Stage 1: training Policy..."
POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_policy_bigger_acp_round_${ROUND}
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}" bash ./launch_training.sh \
    --config-name=train_reactive_diffusion_transformer_real_image_workspace \
    "task=${TASK_CONFIG}" \
    'task.dataset_path=${oc.env:DATASET_PATH}' \
    "task.name=${POLICY_TASK_NAME}" \
    policy.noise_scheduler.prediction_type=v_prediction \
    "policy.use_amp_for_inference=${USE_AMP_FOR_POLICY}" \
    "policy.amp_dtype=${AMP_DTYPE_FOR_POLICY}" \
    "training.use_amp=${USE_AMP_FOR_POLICY}" \
    "training.amp_dtype=${AMP_DTYPE_FOR_POLICY}" \
    "training.debug=${TRAINING_DEBUG}" \
    "optimizer.lr=${POLICY_LR}" \
    "logging.backend=${LOGGING_BACKEND}" \
    "logging.mode=${LOGGING_MODE}" \
    "$@"
