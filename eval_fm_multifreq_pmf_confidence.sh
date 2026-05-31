#!/bin/bash

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"

CONFIG_NAME="${CONFIG_NAME:-train_fm_multifreq_pmf_confidence_workspace}"

# Override with CKPT_PATH=/path/to/latest.ckpt
CKPT_PATH="${CKPT_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/right/frequency_adaptive_policy/data/outputs/latest_pmf_confidence/checkpoints/latest.ckpt}"

DATASET_PATH="${DATASET_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"

VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
MAX_EPISODES="${MAX_EPISODES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations_fm_multifreq_pmf_confidence}"

echo "=========================================="
echo "FM MultiFreq PMF Confidence Policy Evaluation"
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

ARGS=(
    --config-name "${CONFIG_NAME}"
    "task=real_${TASK}_image_wrench_offline_10fps_acp"
    "task.dataset_path=${DATASET_PATH}"
    "+use_full_dataset=True"
    "+num_inference_steps=${NUM_INFERENCE_STEPS}"
    "+video_fps=${VIDEO_FPS}"
    "+device=${GPU_DEVICE}"
    "task.dataset.val_ratio=0.0"
    "policy.obs_encoder.transforms=[{type: CenterCrop}]"
)

if [ -n "${MAX_EPISODES}" ]; then
    ARGS+=("+max_episodes=${MAX_EPISODES}")
fi

python eval_fm_multifreq_pmf_confidence.py "${ARGS[@]}"

echo ""
echo "=========================================="
echo "Complete!"
