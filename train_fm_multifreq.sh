#!/bin/bash

#--------------------------------
# Config
GPU_ID=0,1,2,3,4,5,6,7

TASK="dual_switch"
DATASET_VERSION="v4"
ACTION_TYPE="right_arm_6DOF_virtual_target_stiffness"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_acp"

# Distillation control
# Set USE_DISTILLATION=True for Stage 1 + Stage 2 (CD distillation)
# Set USE_DISTILLATION=False for Stage 1 only (FM velocity matching)
USE_DISTILLATION=True

# Training epochs
STAGE1_EPOCHS=600    # FM velocity matching (10hz + 50hz)
STAGE2_EPOCHS=300       # CD distillation for 50hz (0 = disabled when USE_DISTILLATION=False)


TOTAL_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS))

DATASET_PATH="/mnt/data/users/lifeng/${TASK}_${DATASET_VERSION}_round_1_zarr"

USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
TRAINING_DEBUG=False
POLICY_LR=1.0e-4
#--------------------------------

echo "=========================================="
echo "Multi-frequency FM training"
echo "  Use distillation: ${USE_DISTILLATION}"
echo "  Stage 1: ${STAGE1_EPOCHS} epochs (FM velocity matching)"
if [ "${USE_DISTILLATION}" = "True" ]; then
echo "  Stage 2: ${STAGE2_EPOCHS} epochs (CD distillation for 50hz)"
fi
echo "  Total:   ${TOTAL_EPOCHS} epochs"
echo "  Dataset: ${DATASET_PATH}"
echo "=========================================="

POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_fm_multifreq
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_fm_multifreq_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    policy.use_distillation=${USE_DISTILLATION} \
    training.num_epochs=${TOTAL_EPOCHS} \
    training.use_distillation=${USE_DISTILLATION} \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    policy.training_stage=1 \
    optimizer.lr=${POLICY_LR} \

exit 0
