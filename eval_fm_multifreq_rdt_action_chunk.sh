#!/bin/bash

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"

CONFIG_NAME="${CONFIG_NAME:-train_fm_multifreq_rdt_action_chunk_workspace}"
POLICY_TYPE="${POLICY_TYPE:-rdt_action_chunk}"

# Checkpoint path. Override with CKPT_PATH=/path/to/checkpoint.ckpt.
CKPT_PATH="${CKPT_PATH:-/mnt/workspace/users/lifeng/frequency_adaptive_policy_flow/data/outputs/2026.05.08/15.49.50_train_fm_multifreq_rdt_action_chunk_workspace_dual_switch_v6_fm_multifreq_rdt_action_chunk/checkpoints}"

# Dataset path. Override with DATASET_PATH=/path/to/zarr.
DATASET_PATH="${DATASET_PATH:-/mnt/workspace/users/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"

GPU_DEVICE="${GPU_DEVICE:-cuda:6}"

VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
NUM_INFERENCE_STEPS_10HZ="${NUM_INFERENCE_STEPS_10HZ:-10}"
TRAINING_STAGE="${TRAINING_STAGE:-2}"

OUTPUT_DIR="${OUTPUT_DIR:-evaluations_fm_multifreq_ctm_1steps}"

echo "=========================================="
echo "FM MultiFreq RDT Action Chunk Evaluation"
echo "=========================================="
echo "Policy type:${POLICY_TYPE}"
echo "Config:     ${CONFIG_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "GPU device: ${GPU_DEVICE}"
echo "Inference steps: ${NUM_INFERENCE_STEPS}"
echo "10Hz steps: ${NUM_INFERENCE_STEPS_10HZ}"
echo "Training stage: ${TRAINING_STAGE}"
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
    "+num_inference_steps_10hz=${NUM_INFERENCE_STEPS_10HZ}"
    "+video_fps=${VIDEO_FPS}"
    "+device=${GPU_DEVICE}"
    "task.dataset.val_ratio=0.0"
    "policy.obs_encoder.transforms=[{type: CenterCrop}]"
)

ARGS+=("+training_stage=${TRAINING_STAGE}")

python eval_fm_multifreq_rdt_action_chunk.py "${ARGS[@]}"

echo ""
echo "=========================================="
echo "Complete!"
