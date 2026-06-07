#!/bin/bash

#--------------------------------
# Config
GPU_ID=0,1,2,3,4,5,6,7

TASK="dual_switch"
DATASET_VERSION="v7"
ROUND=1
# ACTION_TYPE="right_arm_6DOF" # No Auxiliary Task
# ACTION_TYPE="right_arm_6DOF_wrench" # FP
ACTION_TYPE="right_arm_6DOF_virtual_target_stiffness" # VRR
DATASET_TAG="${TASK}_${DATASET_VERSION}_round_${ROUND}"
DATASET_PATH="/mnt/workspace/users/lifeng/${DATASET_TAG}_zarr"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_acp"

USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
TRAINING_DEBUG=False
POLICY_LR=1.0e-4
LOGGING_BACKEND="${LOGGING_BACKEND:-wandb}"
LOGGING_MODE="${LOGGING_MODE:-online}"
#--------------------------------

# Stage 0: Post-process data
echo "Stage 0: post-processing data..."
#python post_process_data.py --tag ${DATASET_TAG} --action_type ${ACTION_TYPE} --skip_static_frames

# Stage 1: Train Policy
echo "Stage 1: training Policy..."
POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_policy_bigger_acp_round_${ROUND}
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --main_process_port 29501 train.py \
    --config-name=train_reactive_diffusion_transformer_real_image_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    policy.noise_scheduler.prediction_type=v_prediction \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    optimizer.lr=${POLICY_LR} \
    logging.backend=${LOGGING_BACKEND} \
    logging.mode=${LOGGING_MODE}

exit 0
