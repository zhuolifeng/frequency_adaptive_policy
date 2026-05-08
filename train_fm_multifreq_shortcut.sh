#!/bin/bash

#--------------------------------
# Config
GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6}"
ROUND="${ROUND:-1}"
TASK_CONFIG="${TASK_CONFIG:-real_${TASK}_image_wrench_offline_10fps_acp}"

EPOCHS="${EPOCHS:-800}"
DATASET_PATH="${DATASET_PATH:-/mnt/data/users/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"

USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
POLICY_LR="${POLICY_LR:-1.0e-4}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
NUM_INFERENCE_STEPS_10HZ="${NUM_INFERENCE_STEPS_10HZ:-10}"
SHORTCUT_DENOISE_STEPS="${SHORTCUT_DENOISE_STEPS:-128}"
SHORTCUT_BOOTSTRAP_EVERY="${SHORTCUT_BOOTSTRAP_EVERY:-8}"
SHORTCUT_VERSION="${SHORTCUT_VERSION:-new}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-fm_multifreq_shortcut_policy}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_ID="${WANDB_ID:-}"
#--------------------------------

echo "=========================================="
echo "Multi-frequency FM shortcut training"
echo "  Epochs: ${EPOCHS}"
echo "  Dataset: ${DATASET_PATH}"
echo "  50Hz inference steps: ${NUM_INFERENCE_STEPS}"
echo "  10Hz inference steps: ${NUM_INFERENCE_STEPS_10HZ}"
echo "  Shortcut denoise steps: ${SHORTCUT_DENOISE_STEPS}"
echo "  Shortcut version: ${SHORTCUT_VERSION}"
echo "  W&B project/mode: ${WANDB_PROJECT}/${WANDB_MODE}"
echo "=========================================="

POLICY_TASK_NAME="${TASK}_${DATASET_VERSION}_fm_multifreq_shortcut"
ARGS=(
    --config-name=train_fm_multifreq_shortcut_workspace
    task=${TASK_CONFIG}
    task.dataset_path=${DATASET_PATH}
    task.name=${POLICY_TASK_NAME}
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY}
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY}
    policy.shortcut_version=${SHORTCUT_VERSION}
    policy.num_inference_steps=${NUM_INFERENCE_STEPS}
    policy.num_inference_steps_10hz=${NUM_INFERENCE_STEPS_10HZ}
    policy.shortcut_denoise_steps=${SHORTCUT_DENOISE_STEPS}
    policy.shortcut_bootstrap_every=${SHORTCUT_BOOTSTRAP_EVERY}
    training.num_epochs=${EPOCHS}
    training.use_amp=${USE_AMP_FOR_POLICY}
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY}
    training.debug=${TRAINING_DEBUG}
    optimizer.lr=${POLICY_LR}
    logging.project=${WANDB_PROJECT}
    logging.mode=${WANDB_MODE}
)

if [ -n "${WANDB_NAME}" ]; then
    ARGS+=(logging.name="${WANDB_NAME}")
fi
if [ -n "${WANDB_GROUP}" ]; then
    ARGS+=(logging.group="${WANDB_GROUP}")
fi
if [ -n "${WANDB_ID}" ]; then
    ARGS+=(logging.id="${WANDB_ID}" logging.resume=True)
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --main_process_port 29501 train.py "${ARGS[@]}"

exit 0
