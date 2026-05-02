#!/bin/bash

# Evaluate iMF (improved MeanFlow) multi-frequency policy.
# Generates MSE videos and action chunk videos for both 10hz and 50hz.

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"

CONFIG_NAME="train_fm_multifreq_imf_workspace"

# Checkpoint path. Override with CKPT_PATH=/path/to/checkpoint.ckpt.
CKPT_PATH="${CKPT_PATH:-}"
if [ -z "${CKPT_PATH}" ]; then
    echo "ERROR: Set CKPT_PATH to point to your iMF checkpoint."
    echo "  e.g.: CKPT_PATH=/path/to/imf_final.ckpt bash eval_fm_multifreq_imf.sh"
    exit 1
fi

# Dataset path. Override with DATASET_PATH=/path/to/zarr.
DATASET_PATH="${DATASET_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"

GPU_DEVICE="${GPU_DEVICE:-cuda:6}"

VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-evaluations_fm_multifreq_imf_${NUM_INFERENCE_STEPS}steps}"

echo "=========================================="
echo "FM MultiFreq iMF Policy Evaluation"
echo "=========================================="
echo "Config:     ${CONFIG_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "GPU device: ${GPU_DEVICE}"
echo "Inference steps: ${NUM_INFERENCE_STEPS}"
echo "Video FPS:  ${VIDEO_FPS}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

export CKPT_PATH="${CKPT_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"

python eval_fm_multifreq_imf.py \
    --config-name "${CONFIG_NAME}" \
    "task=real_${TASK}_image_wrench_offline_10fps_acp" \
    "task.dataset_path=${DATASET_PATH}" \
    "+use_full_dataset=True" \
    "+num_inference_steps=${NUM_INFERENCE_STEPS}" \
    "+video_fps=${VIDEO_FPS}" \
    "+device=${GPU_DEVICE}" \
    "task.dataset.val_ratio=0.0" \
    "policy.obs_encoder.transforms=[{type: CenterCrop}]"

echo ""
echo "=========================================="
echo "Complete!"
