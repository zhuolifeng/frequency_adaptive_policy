#!/bin/bash

#--------------------------------
# Config
GPU_ID=0,1,2,3,4,5,6,7

TASK="dual_switch"
STAGE1_DATASET_VERSION="v4"
STAGE2_DATASET_VERSION="v4.1"
ACTION_TYPE="right_arm_6DOF_virtual_target_stiffness"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_acp"

# Three-stage training epochs
STAGE1_EPOCHS=300
STAGE2_EPOCHS=600
STAGE3_EPOCHS=250
TOTAL_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS))

STAGE1_DATASET_PATH="/mnt/data/users/lifeng/${TASK}_${STAGE1_DATASET_VERSION}_round_1_zarr"
STAGE2_DATASET_PATH="/mnt/data/users/lifeng/${TASK}_${STAGE2_DATASET_VERSION}_round_1_zarr"

USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
TRAINING_DEBUG=False
POLICY_LR=1.0e-4
LOGGING_MODE="online"
#--------------------------------

# Stage 0: Post-process data
echo "Stage 0: post-processing data..."
#python post_process_data.py --tag ${TASK}_${STAGE1_DATASET_VERSION}_round_1 --action_type ${ACTION_TYPE} --skip_static_frames

echo "Stage 1: training Policy..."
POLICY_TASK_NAME=${TASK}_${STAGE1_DATASET_VERSION}_policy_bigger_acp
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_hybrid_diffusion_act_confidence_stagewise_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${STAGE1_DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    policy.noise_scheduler.prediction_type=v_prediction \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.num_epochs=${TOTAL_EPOCHS} \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.stage3_epochs=${STAGE3_EPOCHS} \
    training.stage3_dataset_path=${STAGE2_DATASET_PATH} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    policy.training_stage=1 \
    optimizer.lr=${POLICY_LR} \
    logging.mode=${LOGGING_MODE} \

exit 0
