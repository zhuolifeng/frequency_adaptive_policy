#!/bin/bash

GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"
TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v7}"
ROUND="${ROUND:-1}"
TASK_CONFIG="${TASK_CONFIG:-real_${TASK}_image_wrench_offline_10fps_50hz}"
DATASET_PATH="${DATASET_PATH:-/mnt/data/users/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"

POLICY_TASK_NAME="${POLICY_TASK_NAME:-${TASK}_${DATASET_VERSION}_act_50hz}"
USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
POLICY_LR="${POLICY_LR:-1.0e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-800}"
LOGGING_BACKEND="${LOGGING_BACKEND:-tensorboard}"
LOGGING_MODE="${LOGGING_MODE:-online}"

echo "=========================================="
echo "ACT 50Hz training"
echo "  Config:  train_act_50hz_workspace"
echo "  Task:    ${TASK_CONFIG}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Epochs:  ${NUM_EPOCHS}"
echo "  LR:      ${POLICY_LR}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --main_process_port 29502 train.py \
    --config-name=train_act_50hz_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.num_epochs=${NUM_EPOCHS} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    optimizer.lr=${POLICY_LR} \
    logging.backend=${LOGGING_BACKEND} \
    logging.mode=${LOGGING_MODE}

exit 0
