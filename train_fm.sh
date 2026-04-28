#!/bin/bash

#--------------------------------
# Config
GPU_ID=0,1,2,3

TASK="dual_switch"
DATASET_VERSION="v4"
ROUND=1
DATASET_TAG="${TASK}_${DATASET_VERSION}_round_${ROUND}"
DATASET_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${DATASET_TAG}_zarr"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps"

TRAINING_DEBUG=False
POLICY_LR=1.0e-4
LOGGING_MODE="online"
#--------------------------------

# Train Flow Matching Transformer Policy
echo "Training Flow Matching Transformer Policy..."
POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_fm_transformer_round_${ROUND}
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_fm_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    training.debug=${TRAINING_DEBUG} \
    optimizer.lr=${POLICY_LR} \
    logging.mode=${LOGGING_MODE}

exit 0
