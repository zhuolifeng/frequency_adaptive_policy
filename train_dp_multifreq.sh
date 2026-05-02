#!/bin/bash

#--------------------------------
# Config
GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6}"
ROUND="${ROUND:-1}"
DATASET_PATH="${DATASET_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_acp"
SEED="${SEED:-42}"

EXP_NAME="${EXP_NAME:-dp_multifreq}"
POLICY_TASK_NAME="${TASK}_${DATASET_VERSION}_dp_multifreq"

# Training epochs
STAGE1_EPOCHS="${STAGE1_EPOCHS:-600}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-300}"
TOTAL_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS))

USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
POLICY_LR="${POLICY_LR:-1.0e-4}"

# Accelerate port (avoid conflict with other jobs)
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29502}"
#--------------------------------

echo "=========================================="
echo "DP MultiFreq Training (accelerate multi-GPU)"
echo "=========================================="
echo "  Task:        ${TASK}"
echo "  Dataset:     ${DATASET_PATH}"
echo "  GPUs:        ${GPU_ID}"
echo "  Seed:        ${SEED}"
echo "  Exp name:    ${EXP_NAME}"
echo "  Stage 1:     ${STAGE1_EPOCHS} epochs (diffusion policy)"
echo "  Stage 2:     ${STAGE2_EPOCHS} epochs (CTM distillation)"
echo "  Total:       ${TOTAL_EPOCHS} epochs"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --main_process_port ${MAIN_PROCESS_PORT} train.py \
    --config-name=train_dp_multifreq_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path="${DATASET_PATH}" \
    task.name="${POLICY_TASK_NAME}" \
    exp_name="${EXP_NAME}" \
    training.seed="${SEED}" \
    training.num_epochs=${TOTAL_EPOCHS} \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    policy.training_stage=1 \
    optimizer.lr=${POLICY_LR} \
    "$@"

exit 0
