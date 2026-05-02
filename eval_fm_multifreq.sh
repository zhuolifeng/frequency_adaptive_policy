#!/bin/bash

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"

# POLICY_TYPE options:
#   fm          -> train_fm_multifreq_workspace
#   consistency -> train_fm_multifreq_consistency_workspace
#   ctm         -> train_fm_multifreq_ctm_workspace
POLICY_TYPE="${POLICY_TYPE:-ctm}"
if [ -z "${CONFIG_NAME}" ]; then
    case "${POLICY_TYPE}" in
        fm)
            CONFIG_NAME="train_fm_multifreq_workspace"
            ;;
        consistency)
            CONFIG_NAME="train_fm_multifreq_consistency_workspace"
            ;;
        ctm)
            CONFIG_NAME="train_fm_multifreq_ctm_workspace"
            ;;
        *)
            echo "Unsupported POLICY_TYPE=${POLICY_TYPE}. Use fm, consistency, or ctm."
            exit 1
            ;;
    esac
fi

# Checkpoint path. Override with CKPT_PATH=/path/to/checkpoint.ckpt.
CKPT_PATH="${CKPT_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/right/frequency_adaptive_policy/data/outputs/2026.04.30/17.43.24_train_fm_multifreq_ctm_workspace_dual_switch_v6_fm_multifreq_ctm/checkpoints/ctm_final.ckpt}"

# Dataset path. Override with DATASET_PATH=/path/to/zarr.
DATASET_PATH="${DATASET_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"

GPU_DEVICE="${GPU_DEVICE:-cuda:6}"

VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
USE_DISTILLATION="${USE_DISTILLATION:-True}"

OUTPUT_DIR="${OUTPUT_DIR:-evaluations_fm_multifreq_ctm_1steps}"

echo "=========================================="
echo "FM MultiFreq Policy Evaluation"
echo "=========================================="
echo "Policy type:${POLICY_TYPE}"
echo "Config:     ${CONFIG_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "GPU device: ${GPU_DEVICE}"
echo "Inference steps: ${NUM_INFERENCE_STEPS}"
echo "Use distillation: ${USE_DISTILLATION}"
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

if [ "${CONFIG_NAME}" = "train_fm_multifreq_workspace" ]; then
    ARGS+=("+use_distillation=${USE_DISTILLATION}")
fi

if [ "${CONFIG_NAME}" = "train_fm_multifreq_ctm_workspace" ]; then
    ARGS+=("+training_stage=2")
fi

python eval_fm_multifreq.py "${ARGS[@]}"

echo ""
echo "=========================================="
echo "Complete!"
