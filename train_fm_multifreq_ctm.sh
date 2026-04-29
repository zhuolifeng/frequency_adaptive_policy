#!/bin/bash

# Two-stage multi-frequency FM-CTM training with a new stop-time backbone.

GPU_ID=0,1,2,3,4,5,6,7

TASK="dual_switch"
DATASET_VERSION="v4"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_acp"

STAGE1_EPOCHS=600
STAGE2_EPOCHS=200
POLICY_LR=1.0e-4
STAGE2_LR=1.0e-5

DATASET_PATH="/mnt/data/users/lifeng/${TASK}_${DATASET_VERSION}_round_1_zarr"

USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
TRAINING_DEBUG=False

echo "=========================================="
echo "Multi-frequency FM-CTM training"
echo "  Stage 1 epochs: ${STAGE1_EPOCHS}"
echo "  Stage 2 epochs: ${STAGE2_EPOCHS}"
echo "  Dataset: ${DATASET_PATH}"
echo "=========================================="

POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_fm_multifreq_ctm
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_fm_multifreq_ctm_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    training.resume=False \
    optimizer.lr=${POLICY_LR} \
    optimizer.stage2_lr=${STAGE2_LR}

exit 0
