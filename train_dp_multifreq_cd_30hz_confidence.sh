#!/usr/bin/env bash
set -eo pipefail

TASK="${TASK:-dual_box}"
DATASET_VERSION="${DATASET_VERSION:-v2}"
ROUND="${ROUND:-1}"
export DATASET_PATH="${DATASET_PATH:-data/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
TASK_CONFIG="${TASK_CONFIG:-real_${TASK}_image_wrench_offline_10fps_acp}"
SEED="${SEED:-42}"

EXP_NAME="${EXP_NAME:-dp_multifreq_cd_30hz_confidence}"
POLICY_TASK_NAME="${TASK}_${DATASET_VERSION}_dp_multifreq_cd_30hz_confidence"
LOSS_TYPE="${LOSS_TYPE:-eps}"

# Stage 2 calibration dataset (optional, defaults to same as stage 1)
STAGE2_DATASET_VERSION="${STAGE2_DATASET_VERSION:-}"
STAGE2_ROUND="${STAGE2_ROUND:-1}"
export STAGE2_DATASET_PATH="${STAGE2_DATASET_PATH:-}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-800}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-250}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-150}"

STAGE1_LR="${STAGE1_LR:-1.0e-4}"
STAGE1_OBS_ENCODER_LR="${STAGE1_OBS_ENCODER_LR:-3.0e-5}"
STAGE2_CONFIDENCE_LR="${STAGE2_CONFIDENCE_LR:-1.0e-4}"
STAGE3_LR="${STAGE3_LR:-1.0e-5}"

CONFIDENCE_N_SAMPLES="${CONFIDENCE_N_SAMPLES:-8}"
CONFIDENCE_NUM_INFERENCE_STEPS="${CONFIDENCE_NUM_INFERENCE_STEPS:-3}"

USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
RESUME="${RESUME:-False}"
# Quote the checkpoint for Hydra, including names such as epoch=0100.ckpt.
pretrained_override=null
if [[ -n "${PRETRAINED_CKPT_PATH:-}" && "$PRETRAINED_CKPT_PATH" != null ]]; then
    pretrained_override='${oc.env:PRETRAINED_CKPT_PATH}'
fi
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29505}"

NUM_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS))

# Build stage 2 dataset path if STAGE2_DATASET_VERSION is set
if [ -z "${STAGE2_DATASET_PATH}" ] && [ -n "${STAGE2_DATASET_VERSION}" ]; then
    STAGE2_DATASET_PATH="data/${TASK}_${STAGE2_DATASET_VERSION}_round_${STAGE2_ROUND}_zarr"
fi

calibration_override=null
if [[ -n "$STAGE2_DATASET_PATH" ]]; then
    calibration_override='${oc.env:STAGE2_DATASET_PATH}'
fi

echo "=========================================="
echo "DP MultiFreq 30Hz CD Confidence Training"
echo "=========================================="
echo "  Task config:        ${TASK_CONFIG}"
echo "  Dataset (Stage 1):  ${DATASET_PATH}"
echo "  Dataset (Stage 2):  ${STAGE2_DATASET_PATH:-same as Stage 1}"
echo "  Stage epochs:       ${STAGE1_EPOCHS}/${STAGE2_EPOCHS}/${STAGE3_EPOCHS}"
echo "  LR:                 ${STAGE1_LR}/${STAGE2_CONFIDENCE_LR}/${STAGE3_LR}"
echo "  Loss type:          ${LOSS_TYPE}"
echo "  Confidence:         samples=${CONFIDENCE_N_SAMPLES}, steps=${CONFIDENCE_NUM_INFERENCE_STEPS}"
echo "=========================================="

bash ./launch_training.sh \
    --config-name=train_dp_multifreq_cd_30hz_confidence_workspace \
    "task=${TASK_CONFIG}" \
    'task.dataset_path=${oc.env:DATASET_PATH}' \
    "task.name=${POLICY_TASK_NAME}" \
    "exp_name=${EXP_NAME}" \
    "loss_type=${LOSS_TYPE}" \
    "training.seed=${SEED}" \
    "training.stage1_epochs=${STAGE1_EPOCHS}" \
    "training.stage2_epochs=${STAGE2_EPOCHS}" \
    "training.stage3_epochs=${STAGE3_EPOCHS}" \
    "training.num_epochs=${NUM_EPOCHS}" \
    "training.pretrained_ckpt_path=$pretrained_override" \
    "training.use_amp=${USE_AMP_FOR_POLICY}" \
    "training.amp_dtype=${AMP_DTYPE_FOR_POLICY}" \
    "training.debug=${TRAINING_DEBUG}" \
    "training.resume=${RESUME}" \
    "optimizer.stage1_lr=${STAGE1_LR}" \
    "optimizer.stage1_obs_encoder_lr=${STAGE1_OBS_ENCODER_LR}" \
    "optimizer.stage2_confidence_lr=${STAGE2_CONFIDENCE_LR}" \
    "optimizer.stage3_lr=${STAGE3_LR}" \
    "policy.use_amp_for_inference=${USE_AMP_FOR_POLICY}" \
    "policy.amp_dtype=${AMP_DTYPE_FOR_POLICY}" \
    "policy.confidence_n_samples=${CONFIDENCE_N_SAMPLES}" \
    "policy.confidence_num_inference_steps=${CONFIDENCE_NUM_INFERENCE_STEPS}" \
    "training.stage2_dataset_path=$calibration_override" \
    "$@"
