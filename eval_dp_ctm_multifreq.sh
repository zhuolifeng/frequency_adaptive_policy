#!/bin/bash
#
# Offline evaluation for DP-CTM Multi-Frequency Policy.
#
# Evaluates both 10Hz (Karras Heun ODE) and 50Hz (CTM single-step) heads,
# generating:
#   - Per-episode MSE visualization videos
#   - Per-episode action chunk comparison videos (GT vs Pred)
#   - Aggregated metrics.json
#
# Usage:
#   # Basic (all defaults)
#   bash eval_dp_ctm_multifreq.sh
#
#   # Custom checkpoint & dataset
#   CKPT_PATH=/path/to/ckpt DATASET_PATH=/path/to/zarr bash eval_dp_ctm_multifreq.sh
#
#   # Evaluate 50Hz only, with chaining, on GPU 3
#   EVAL_HEAD=50hz USE_CHAINING=1 GPU_DEVICE=cuda:3 bash eval_dp_ctm_multifreq.sh
#
#   # Use full dataset (val_ratio=0.0) with custom output dir
#   USE_FULL_DATASET=True OUTPUT_DIR=my_eval bash eval_dp_ctm_multifreq.sh
#
# All configurable variables (override via environment):
#   TASK              Task name prefix                         [default: dual_switch]
#   DATASET_VERSION   Dataset version                          [default: v6.1]
#   ROUND             Dataset round                            [default: 1]
#   CKPT_PATH         Checkpoint .ckpt path                    [default: see below]
#   DATASET_PATH      Zarr dataset path                        [default: auto from TASK/VERSION/ROUND]
#   GPU_DEVICE        CUDA device                              [default: cuda:0]
#   VIDEO_FPS         Video playback FPS                       [default: 1]
#   NUM_INFERENCE_STEPS  Number of inference steps (10hz ODE)  [default: 1]
#   USE_CHAINING      Enable CTM chaining for 50Hz (0|1)       [default: 0]
#   EVAL_HEAD         Which head to evaluate (10hz|50hz|both)  [default: both]
#   USE_FULL_DATASET  Evaluate on full dataset (True|False)    [default: True]
#   OUTPUT_DIR        Output directory for videos & metrics    [default: auto]
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
cd "$ROOT_DIR"

# ---- Configurable defaults ----
TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"

# Policy type: edm (Stage 1, multi-step ODE) or ctm (Stage 2, single-step CTM)
POLICY_TYPE="${POLICY_TYPE:-edm}"

# Checkpoint path
CKPT_PATH="${CKPT_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/right/frequency_adaptive_policy/data/outputs/2026.05.05/14.36.41_train_dp_ctm_multifreq_workspace_dual_switch_v6_dp_ctm_multifreq/checkpoints/stage1_final.ckpt}"

# Dataset path
DATASET_PATH="${DATASET_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"

GPU_DEVICE="${GPU_DEVICE:-cuda:0}"

VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
USE_CHAINING="${USE_CHAINING:-0}"
EVAL_HEAD="${EVAL_HEAD:-both}"
USE_FULL_DATASET="${USE_FULL_DATASET:-True}"

OUTPUT_DIR="${OUTPUT_DIR:-evaluations_dp_${POLICY_TYPE}_multifreq_${NUM_INFERENCE_STEPS}steps}"

# ---- Validate ----
if [[ "$POLICY_TYPE" != "edm" && "$POLICY_TYPE" != "ctm" ]]; then
    echo "Error: POLICY_TYPE must be edm or ctm. Got: $POLICY_TYPE"
    exit 1
fi

if [[ "$EVAL_HEAD" != "10hz" && "$EVAL_HEAD" != "50hz" && "$EVAL_HEAD" != "both" ]]; then
    echo "Error: EVAL_HEAD must be 10hz, 50hz, or both. Got: $EVAL_HEAD"
    exit 1
fi

echo "=========================================="
echo " DP-CTM Multi-Frequency Evaluation"
echo "=========================================="
echo "  Policy type:      ${POLICY_TYPE}"
echo "  Checkpoint:       ${CKPT_PATH}"
echo "  Dataset:          ${DATASET_PATH}"
echo "  GPU device:       ${GPU_DEVICE}"
echo "  Eval head:        ${EVAL_HEAD}"
echo "  Inference steps:  ${NUM_INFERENCE_STEPS}"
echo "  Use chaining:     ${USE_CHAINING}"
echo "  Use full dataset: ${USE_FULL_DATASET}"
echo "  Video FPS:        ${VIDEO_FPS}"
echo "  Output dir:       ${OUTPUT_DIR}"
echo ""

export CKPT_PATH="${CKPT_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"
export EVAL_HEAD="${EVAL_HEAD}"
export USE_CHAINING="${USE_CHAINING}"
export POLICY_TYPE="${POLICY_TYPE}"

# Set training_stage based on policy type
if [ "${POLICY_TYPE}" = "edm" ]; then
    TRAINING_STAGE=1
else
    TRAINING_STAGE=2
fi

ARGS=(
    "task=real_${TASK}_image_wrench_offline_10fps_acp"
    "task.dataset_path=${DATASET_PATH}"
    "+use_full_dataset=${USE_FULL_DATASET}"
    "+num_inference_steps=${NUM_INFERENCE_STEPS}"
    "+video_fps=${VIDEO_FPS}"
    "+device=${GPU_DEVICE}"
    "task.dataset.val_ratio=0.0"
    "policy.obs_encoder.transforms=[{type: CenterCrop}]"
    "+training_stage=${TRAINING_STAGE}"
)

python eval_dp_ctm_multifreq.py "${ARGS[@]}"

echo ""
echo "=========================================="
echo " Evaluation complete!"
echo "=========================================="
