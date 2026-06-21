#!/bin/bash

GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29504}"
LOGGING_BACKEND="${LOGGING_BACKEND:-wandb}"
LOGGING_MODE="${LOGGING_MODE:-online}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_ID}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ARRAY[@]}}"
ACCELERATE_ARGS=(--main_process_port "${MAIN_PROCESS_PORT}")
if [ "${NUM_PROCESSES}" -gt 1 ]; then
    ACCELERATE_ARGS=(--multi_gpu --num_processes "${NUM_PROCESSES}" "${ACCELERATE_ARGS[@]}")
fi

# Stage 0: Post-process data
echo "Stage 0: post-processing data..."
#python post_process_data.py --tag flip_v2 --action_type right_arm_6DOF --skip_static_frames

# Stage 1: Train Policy
echo "Stage 1: training Policy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch "${ACCELERATE_ARGS[@]}" train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_dual_box_image_dp_10fps \
    task.dataset_path=/mnt/workspace/users/lifeng/dual_box_v5_round_1_zarr \
    task.name=real_dual_box_image_dp_10fps_v2 \
    logging.mode=${LOGGING_MODE} \
    logging.backend=${LOGGING_BACKEND}
