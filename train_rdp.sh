#!/bin/bash

GPU_ID="${GPU_ID:-0,1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"

TASK="dual_box"
DATASET_VERSION="v2"
ACTION_TYPE="right_arm_6DOF_wrench"
DATASET_PATH="/mnt/workspace/users/lifeng/${TASK}_${DATASET_VERSION}_round_1_zarr"
TRAINING_DEBUG=False
LOGGING_BACKEND="${LOGGING_BACKEND:-tensorboard}"
LOGGING_MODE="${LOGGING_MODE:-online}"

IDENTIFIER=$(date +%m%d%H%M%S)
SEARCH_PATH="./data/outputs"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_ID}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ARRAY[@]}}"
ACCELERATE_ARGS=(--main_process_port "${MAIN_PROCESS_PORT}")
if [ "${NUM_PROCESSES}" -gt 1 ]; then
    ACCELERATE_ARGS=(--multi_gpu --num_processes "${NUM_PROCESSES}" "${ACCELERATE_ARGS[@]}")
fi

# Stage 0: Post-process data
echo "Stage 0: post-processing data..."
#python post_process_data.py --tag ${TASK}_${DATASET_VERSION} --action_type ${ACTION_TYPE} --skip_static_frames

# Stage 1: Train Asymmetric Tokenizer
# echo "Stage 1: training Asymmetric Tokenizer..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
#     --config-name=train_at_workspace \
#     task=real_${TASK}_image_wrench_at_10fps \
#     task.dataset_path=${DATASET_PATH} \
#     task.name=real_${TASK}_image_wrench_at_10fps_${IDENTIFIER} \
#     at=at_${TASK} \
#     training.debug=${TRAINING_DEBUG} \
#     logging.mode=${LOGGING_MODE}

# find the latest checkpoint
echo ""
echo "Searching for the latest AT checkpoint..."
if [ -z "${AT_LOAD_DIR}" ]; then
    AT_LOAD_DIR=$(find "${SEARCH_PATH}" -maxdepth 4 \
        -path "*train_vae_real_${TASK}_image_wrench_at_10fps*/checkpoints/latest.ckpt" \
        -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
fi

if [ ! -f "${AT_LOAD_DIR}" ]; then
    echo "Error: VAE checkpoint not found at ${AT_LOAD_DIR}"
    echo "Set AT_LOAD_DIR=/path/to/latest.ckpt or run Stage 1 AT training first."
    exit 1
fi

# Stage 2: Train Latent Diffusion Policy
echo ""
echo "Stage 2: training Latent Diffusion Policy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch "${ACCELERATE_ARGS[@]}" train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    task=real_${TASK}_image_wrench_ldp_10fps \
    task.dataset_path=${DATASET_PATH} \
    task.name=real_${TASK}_image_wrench_ldp_10fps_${IDENTIFIER} \
    at=at_${TASK} \
    at_load_dir=${AT_LOAD_DIR} \
    training.debug=${TRAINING_DEBUG} \
    logging.backend=${LOGGING_BACKEND} \
    logging.mode=${LOGGING_MODE}
