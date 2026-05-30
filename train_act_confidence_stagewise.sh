#!/bin/bash

GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"
TASK="${TASK:-dual_switch}"
TASK_CONFIG="${TASK_CONFIG:-real_${TASK}_image_wrench_offline_10fps_acp}"

# Stage 1 dataset
STAGE1_DATASET_VERSION="${STAGE1_DATASET_VERSION:-v3}"
STAGE1_ROUND="${STAGE1_ROUND:-1}"
STAGE1_DATASET_PATH="${STAGE1_DATASET_PATH:-/mnt/data/users/lifeng/${TASK}_${STAGE1_DATASET_VERSION}_round_${STAGE1_ROUND}_zarr}"

# Stage 2 dataset (if different from stage 1)
#v3.1
#v6.1
STAGE2_DATASET_VERSION="${STAGE2_DATASET_VERSION:-v3.4}"
STAGE2_ROUND="${STAGE2_ROUND:-1}"
STAGE2_DATASET_PATH="${STAGE2_DATASET_PATH:-/mnt/data/users/lifeng/${TASK}_${STAGE2_DATASET_VERSION}_round_${STAGE2_ROUND}_zarr}"

# Training epochs
STAGE1_EPOCHS="${STAGE1_EPOCHS:-800}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-250}"
TOTAL_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS))

POLICY_TASK_NAME="${POLICY_TASK_NAME:-${TASK}_${STAGE1_DATASET_VERSION}_act_confidence_stagewise}"
USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
POLICY_LR="${POLICY_LR:-1.0e-4}"
CONFIDENCE_LR="${CONFIDENCE_LR:-1.0e-4}"  # Learning rate for Stage 2 (Confidence Head)
CONFIDENCE_WEIGHT_DECAY="${CONFIDENCE_WEIGHT_DECAY:-1.0e-6}"

echo "=========================================="
echo "ACT Confidence Stagewise training"
echo "  Config:  train_act_confidence_stagewise_workspace"
echo "  Task:    ${TASK_CONFIG}"
echo "  Stage 1: ${STAGE1_DATASET_PATH}"
echo "  Stage 2: ${STAGE2_DATASET_PATH:-same as Stage 1}"
echo "  Epochs:  Stage1=${STAGE1_EPOCHS}, Stage2=${STAGE2_EPOCHS}, Total=${TOTAL_EPOCHS}"
echo "  LR:      Stage1=${POLICY_LR}, Stage2=${CONFIDENCE_LR}"
echo "=========================================="

STAGE2_DATASET_ARG=""
if [ -n "${STAGE2_DATASET_PATH}" ]; then
    STAGE2_DATASET_ARG="training.stage2_dataset_path=${STAGE2_DATASET_PATH}"
elif [ -n "${STAGE2_DATASET_VERSION}" ]; then
    STAGE2_DATASET_PATH="/mnt/data/users/lifeng/${TASK}_${STAGE2_DATASET_VERSION}_round_${STAGE2_ROUND}_zarr"
    STAGE2_DATASET_ARG="training.stage2_dataset_path=${STAGE2_DATASET_PATH}"
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --main_process_port 29501 train.py \
    --config-name=train_act_confidence_stagewise_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${STAGE1_DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    ${STAGE2_DATASET_ARG} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    optimizer.lr=${POLICY_LR} \
    optimizer.confidence_lr=${CONFIDENCE_LR} \
    optimizer.confidence_weight_decay=${CONFIDENCE_WEIGHT_DECAY}

exit 0
