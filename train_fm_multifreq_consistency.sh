#!/bin/bash

#--------------------------------
# Config
GPU_ID=0,1,2,3,4,5,6,7

TASK="dual_switch"
DATASET_VERSION="v4"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_acp"

NUM_EPOCHS=900
POLICY_LR=1.0e-4

DATASET_PATH="/mnt/data/users/lifeng/${TASK}_${DATASET_VERSION}_round_1_zarr"

USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
TRAINING_DEBUG=False
#--------------------------------

echo "=========================================="
echo "Teacher-free multi-frequency consistency training"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Time sampling: beta(1.5, 1.0)"
echo "  Dataset: ${DATASET_PATH}"
echo "=========================================="

POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_fm_multifreq_consistency
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_fm_multifreq_consistency_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.num_epochs=${NUM_EPOCHS} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    training.resume=False \
    training.pretrained_ckpt_path=null \
    optimizer.lr=${POLICY_LR}

exit 0
