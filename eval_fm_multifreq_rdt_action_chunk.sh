#!/bin/bash

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"

CONFIG_NAME="${CONFIG_NAME:-train_fm_multifreq_rdt_action_chunk_workspace}"
POLICY_TYPE="${POLICY_TYPE:-rdt_action_chunk}"

# Checkpoint path. Override with CKPT_PATH=/path/to/checkpoint.ckpt.
CKPT_PATH="${CKPT_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/right/frequency_adaptive_policy/data/outputs/2026.05.11/15.59.48_train_fm_multifreq_rdt_action_chunk_workspace_dual_switch_v6_fm_multifreq_rdt_action_chunk/checkpoints/latest.ckpt}"

# Dataset path. Override with DATASET_PATH=/path/to/zarr.
DATASET_PATH="${DATASET_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"

GPU_DEVICE="${GPU_DEVICE:-cuda:5}"

VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
TRAINING_STAGE="${TRAINING_STAGE:-2}"
INFERENCE_SOLVER="${INFERENCE_SOLVER:-euler}"

OUTPUT_DIR="${OUTPUT_DIR:-evaluations_fm_multifreq_rdt_action_chunk_${NUM_INFERENCE_STEPS}steps_${INFERENCE_SOLVER}_new_latest}"

echo "=========================================="
echo "FM MultiFreq RDT Action Chunk Evaluation"
echo "=========================================="
echo "Policy type:${POLICY_TYPE}"
echo "Config:     ${CONFIG_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "GPU device: ${GPU_DEVICE}"
echo "Inference steps: ${NUM_INFERENCE_STEPS}"
echo "Training stage: ${TRAINING_STAGE}"
echo "Inference solver: ${INFERENCE_SOLVER}"
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
    "+inference_solver=${INFERENCE_SOLVER}"
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
