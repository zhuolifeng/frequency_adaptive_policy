#!/usr/bin/env bash
set -eo pipefail

TASK="${TASK:-dual_switch}"
STAGE1_DATASET_VERSION="${STAGE1_DATASET_VERSION:-v4}"
STAGE2_DATASET_VERSION="${STAGE2_DATASET_VERSION:-v4.1}"
TASK_CONFIG="${TASK_CONFIG:-real_${TASK}_image_wrench_offline_10fps_acp}"

# Three-stage training epochs
STAGE1_EPOCHS="${STAGE1_EPOCHS:-300}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-600}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-250}"
TOTAL_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS))

export STAGE1_DATASET_PATH="${STAGE1_DATASET_PATH:-${DATASET_PATH:-data/${TASK}_${STAGE1_DATASET_VERSION}_round_1_zarr}}"
export STAGE2_DATASET_PATH="${STAGE2_DATASET_PATH:-data/${TASK}_${STAGE2_DATASET_VERSION}_round_1_zarr}"

USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
POLICY_LR="${POLICY_LR:-1.0e-4}"
LOGGING_MODE="${LOGGING_MODE:-online}"

echo "Stage 1: training Policy..."
POLICY_TASK_NAME=${TASK}_${STAGE1_DATASET_VERSION}_policy_bigger_acp
bash ./launch_training.sh \
    --config-name=train_hybrid_diffusion_act_confidence_stagewise_workspace \
    "task=${TASK_CONFIG}" \
    'task.dataset_path=${oc.env:STAGE1_DATASET_PATH}' \
    "task.name=${POLICY_TASK_NAME}" \
    policy.noise_scheduler.prediction_type=v_prediction \
    "policy.use_amp_for_inference=${USE_AMP_FOR_POLICY}" \
    "policy.amp_dtype=${AMP_DTYPE_FOR_POLICY}" \
    "training.num_epochs=${TOTAL_EPOCHS}" \
    "training.stage1_epochs=${STAGE1_EPOCHS}" \
    "training.stage2_epochs=${STAGE2_EPOCHS}" \
    "training.stage3_epochs=${STAGE3_EPOCHS}" \
    'training.stage3_dataset_path=${oc.env:STAGE2_DATASET_PATH}' \
    "training.use_amp=${USE_AMP_FOR_POLICY}" \
    "training.amp_dtype=${AMP_DTYPE_FOR_POLICY}" \
    "training.debug=${TRAINING_DEBUG}" \
    policy.training_stage=1 \
    "optimizer.lr=${POLICY_LR}" \
    "logging.mode=${LOGGING_MODE}" \
    "$@"
