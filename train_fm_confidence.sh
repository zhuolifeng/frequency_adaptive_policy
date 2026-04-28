#!/bin/bash

#--------------------------------
# Config
GPU_ID=0,1,2,3

TASK="dual_switch"
STAGE1_DATASET_VERSION="v4"
STAGE2_DATASET_VERSION="v4.1"
ACTION_TYPE="right_arm_6DOF_virtual_target_stiffness"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps"

# Two-stage training epochs
STAGE1_EPOCHS=600   # FM velocity matching
STAGE2_EPOCHS=250   # Confidence head training
TOTAL_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS))

STAGE1_DATASET_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${STAGE1_DATASET_VERSION}_round_1_zarr"
STAGE2_DATASET_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${STAGE2_DATASET_VERSION}_round_1_zarr"

USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
TRAINING_DEBUG=False
POLICY_LR=3.0e-4
#--------------------------------

echo "=========================================="
echo "FM + Confidence two-stage training"
echo "  Stage 1: ${STAGE1_EPOCHS} epochs (FM velocity matching)"
echo "  Stage 2: ${STAGE2_EPOCHS} epochs (Confidence on one-step FM MSE)"
echo "  Total:   ${TOTAL_EPOCHS} epochs"
echo "  Stage 1 dataset: ${STAGE1_DATASET_PATH}"
echo "  Stage 2 dataset: ${STAGE2_DATASET_PATH}"
echo "=========================================="

POLICY_TASK_NAME=${TASK}_${STAGE1_DATASET_VERSION}_fm_confidence
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_fm_confidence_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${STAGE1_DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.num_epochs=${TOTAL_EPOCHS} \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.stage2_dataset_path=${STAGE2_DATASET_PATH} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    policy.training_stage=1 \
    optimizer.lr=${POLICY_LR} \

exit 0
