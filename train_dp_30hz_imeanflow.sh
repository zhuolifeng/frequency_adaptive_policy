#!/usr/bin/env bash
set -eo pipefail

export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29505}"

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v8}"
ROUND="${ROUND:-1}"
export DATASET_PATH="${DATASET_PATH:-data/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
TASK_CONFIG="${TASK_CONFIG:-real_${TASK}_image_wrench_offline_10fps_30hz}"
SEED="${SEED:-42}"

EXP_NAME="${EXP_NAME:-dp_30hz_imeanflow}"
POLICY_TASK_NAME="${TASK}_${DATASET_VERSION}_dp_30hz_imeanflow"
NUM_EPOCHS="${NUM_EPOCHS:-800}"
POLICY_LR="${POLICY_LR:-1.0e-4}"
OBS_ENCODER_LR="${OBS_ENCODER_LR:-1.0e-5}"

IMEANFLOW_NUM_INFERENCE_STEPS="${IMEANFLOW_NUM_INFERENCE_STEPS:-1}"
IMEANFLOW_P_MEAN="${IMEANFLOW_P_MEAN:--0.4}"
IMEANFLOW_P_STD="${IMEANFLOW_P_STD:-1.0}"
IMEANFLOW_DATA_PROPORTION="${IMEANFLOW_DATA_PROPORTION:-0.5}"

USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
RESUME="${RESUME:-False}"
# Quote the checkpoint for Hydra, including names such as epoch=0100.ckpt.
pretrained_override=null
if [[ -n "${PRETRAINED_CKPT_PATH:-}" && "$PRETRAINED_CKPT_PATH" != null ]]; then
    pretrained_override='${oc.env:PRETRAINED_CKPT_PATH}'
fi
LOGGING_BACKEND="${LOGGING_BACKEND:-tensorboard}"
LOGGING_MODE="${LOGGING_MODE:-online}"

echo "=========================================="
echo "DP 30Hz iMeanFlow Training"
echo "=========================================="
echo "  Task config:       ${TASK_CONFIG}"
echo "  Dataset:           ${DATASET_PATH}"
echo "  Epochs:            ${NUM_EPOCHS}"
echo "  LR:                ${POLICY_LR}"
echo "  iMeanFlow steps:   ${IMEANFLOW_NUM_INFERENCE_STEPS}"
echo "=========================================="

bash ./launch_training.sh \
    --config-name=train_dp_30hz_imeanflow_workspace \
    "task=${TASK_CONFIG}" \
    'task.dataset_path=${oc.env:DATASET_PATH}' \
    "task.name=${POLICY_TASK_NAME}" \
    "exp_name=${EXP_NAME}" \
    "training.seed=${SEED}" \
    "training.num_epochs=${NUM_EPOCHS}" \
    "training.pretrained_ckpt_path=$pretrained_override" \
    "training.use_amp=${USE_AMP_FOR_POLICY}" \
    "training.amp_dtype=${AMP_DTYPE_FOR_POLICY}" \
    "training.debug=${TRAINING_DEBUG}" \
    "training.resume=${RESUME}" \
    "optimizer.lr=${POLICY_LR}" \
    "optimizer.obs_encoder_lr=${OBS_ENCODER_LR}" \
    "policy.use_amp_for_inference=${USE_AMP_FOR_POLICY}" \
    "policy.amp_dtype=${AMP_DTYPE_FOR_POLICY}" \
    "meanflow.num_inference_steps=${IMEANFLOW_NUM_INFERENCE_STEPS}" \
    "meanflow.p_mean=${IMEANFLOW_P_MEAN}" \
    "meanflow.p_std=${IMEANFLOW_P_STD}" \
    "meanflow.data_proportion=${IMEANFLOW_DATA_PROPORTION}" \
    "logging.backend=${LOGGING_BACKEND}" \
    "logging.mode=${LOGGING_MODE}" \
    "$@"
