#!/usr/bin/env bash
set -eo pipefail

export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"

TASK="${TASK:-dual_box}"
DATASET_VERSION="${DATASET_VERSION:-v2}"
export DATASET_PATH="${DATASET_PATH:-data/${TASK}_${DATASET_VERSION}_round_1_zarr}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
LOGGING_BACKEND="${LOGGING_BACKEND:-tensorboard}"
LOGGING_MODE="${LOGGING_MODE:-online}"

IDENTIFIER=$(date +%m%d%H%M%S)
SEARCH_PATH="${SEARCH_PATH:-./data/outputs}"

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

# Use the most recent tokenizer checkpoint for this task if none was supplied.
if [ -z "${AT_LOAD_DIR:-}" ]; then
    for candidate in "$SEARCH_PATH"/*/*train_vae_real_${TASK}_image_wrench_at_10fps*/checkpoints/latest.ckpt; do
        if [[ -f "$candidate" && ( -z "${AT_LOAD_DIR:-}" || "$candidate" -nt "$AT_LOAD_DIR" ) ]]; then
            AT_LOAD_DIR="$candidate"
        fi
    done
fi

if [ ! -f "${AT_LOAD_DIR}" ]; then
    echo "Error: VAE checkpoint not found at ${AT_LOAD_DIR}"
    echo "Set AT_LOAD_DIR=/path/to/latest.ckpt or run Stage 1 AT training first."
    exit 1
fi

export AT_LOAD_DIR

# Stage 2: Train Latent Diffusion Policy
echo ""
echo "Stage 2: training Latent Diffusion Policy..."
bash ./launch_training.sh \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    "task=real_${TASK}_image_wrench_ldp_10fps" \
    'task.dataset_path=${oc.env:DATASET_PATH}' \
    "task.name=real_${TASK}_image_wrench_ldp_10fps_${IDENTIFIER}" \
    "at=at_${TASK}" \
    'at_load_dir=${oc.env:AT_LOAD_DIR}' \
    "training.debug=${TRAINING_DEBUG}" \
    "logging.backend=${LOGGING_BACKEND}" \
    "logging.mode=${LOGGING_MODE}" \
    "$@"
