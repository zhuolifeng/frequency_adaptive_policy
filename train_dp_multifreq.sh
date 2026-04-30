#!/bin/bash

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v4.2}"
ROUND="${ROUND:-1}"
DATASET_PATH="${DATASET_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
SEED="${SEED:-42}"

EXP_NAME="${EXP_NAME:-dp_multifreq}"
POLICY_TASK_NAME="${TASK}_${DATASET_VERSION}_dp_multifreq"

echo "=========================================="
echo "DP MultiFreq Training"
echo "=========================================="
echo "Task:        ${TASK}"
echo "Dataset:     ${DATASET_PATH}"
echo "GPU device:  ${GPU_DEVICE}"
echo "Seed:        ${SEED}"
echo "Exp name:    ${EXP_NAME}"
echo ""

python ImplicitRDP/workspace/train_dp_multifreq_workspace.py \
    --config-name=train_dp_multifreq_workspace \
    task=real_${TASK}_image_wrench_offline_10fps_acp \
    task.dataset_path="${DATASET_PATH}" \
    training.device="${GPU_DEVICE}" \
    training.seed="${SEED}" \
    exp_name="${EXP_NAME}" \
    task_name="${POLICY_TASK_NAME}" \
    "$@"
