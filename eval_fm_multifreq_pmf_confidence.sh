#!/bin/bash

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.2}"
ROUND="${ROUND:-1}"

CONFIG_NAME="${CONFIG_NAME:-train_fm_multifreq_pmf_confidence_workspace}"

# Override with CKPT_PATH=/path/to/latest.ckpt
CKPT_PATH="${CKPT_PATH:-/mnt/workspace/users/lifeng/frequency_adaptive_policy_flow/data/outputs/2026.06.01/16.51.22_train_fm_multifreq_pmf_confidence_workspace_dual_switch_v6_pmf_confidence/checkpoints/epoch=0550-train_loss=-0.494.ckpt}"

DATASET_PATH="${DATASET_PATH:-/mnt/workspace/users/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"

VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
TRAINING_STAGE="${TRAINING_STAGE:-2}"
CONFIDENCE_SELECTION="${CONFIDENCE_SELECTION:-True}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-3.0}"
MAX_EPISODES="${MAX_EPISODES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations_fm_multifreq_pmf_confidence}"

TRAINING_STAGE_ARG=""
if [ "${TRAINING_STAGE}" != "auto" ]; then
    TRAINING_STAGE_ARG="+training_stage=${TRAINING_STAGE}"
fi

echo "=========================================="
echo "FM MultiFreq PMF Confidence Policy Evaluation"
echo "=========================================="
echo "Config:     ${CONFIG_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "GPU device: ${GPU_DEVICE}"
echo "Inference steps: ${NUM_INFERENCE_STEPS}"
echo "Stage:      ${TRAINING_STAGE}"
echo "Confidence selection: ${CONFIDENCE_SELECTION}"
echo "Video FPS:  ${VIDEO_FPS}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

export CKPT_PATH="${CKPT_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"

ARGS=(
    --config-name "${CONFIG_NAME}"
    "task=real_${TASK}_image_wrench_offline_10fps_acp"
    "task.dataset_path=${DATASET_PATH}"
    ${TRAINING_STAGE_ARG}
    "+use_full_dataset=True"
    "+num_inference_steps=${NUM_INFERENCE_STEPS}"
    "+video_fps=${VIDEO_FPS}"
    "+device=${GPU_DEVICE}"
    "policy.use_confidence_based_head_selection=${CONFIDENCE_SELECTION}"
    "policy.confidence_threshold_for_head_selection=${CONFIDENCE_THRESHOLD}"
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
