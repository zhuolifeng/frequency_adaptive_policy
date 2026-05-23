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

EXP_NAME="${EXP_NAME:-dp_multifreq_cd_50hz}"
POLICY_TASK_NAME="${TASK}_${DATASET_VERSION}_dp_multifreq_cd_50hz"

STAGE1_CKPT_PATH="${STAGE1_CKPT_PATH:-}"
STAGE1_SOURCE_KEY="${STAGE1_SOURCE_KEY:-ema_model}"

NUM_EPOCHS="${NUM_EPOCHS:-500}"
PREDICTION_TYPE="${PREDICTION_TYPE:-sample}"
CD_TIMESTEP_GRID="${CD_TIMESTEP_GRID:-[99,79,59,39,19,0]}"
CD_WEIGHT="${CD_WEIGHT:-1.0}"
DSM_WEIGHT="${DSM_WEIGHT:-0.0}"

POLICY_LR="${POLICY_LR:-1.0e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
EMA_MAX_VALUE="${EMA_MAX_VALUE:-0.999}"
EMA_POWER="${EMA_POWER:-0.75}"

USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
RESUME="${RESUME:-False}"

MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29503}"
#--------------------------------

if [ -z "${STAGE1_CKPT_PATH}" ]; then
    echo "Error: set STAGE1_CKPT_PATH=/path/to/stage1_final.ckpt"
    exit 1
fi

echo "=========================================="
echo "DP MultiFreq 50Hz CD Training"
echo "=========================================="
echo "  Task:        ${TASK}"
echo "  Dataset:     ${DATASET_PATH}"
echo "  GPUs:        ${GPU_ID}"
echo "  Seed:        ${SEED}"
echo "  Stage1 ckpt: ${STAGE1_CKPT_PATH}"
echo "  Source key:  ${STAGE1_SOURCE_KEY}"
echo "  Epochs:      ${NUM_EPOCHS}"
echo "  Prediction:  ${PREDICTION_TYPE}"
echo "  CD grid:     ${CD_TIMESTEP_GRID}"
echo "  CD weight:   ${CD_WEIGHT}"
echo "  DSM weight:  ${DSM_WEIGHT}"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --main_process_port ${MAIN_PROCESS_PORT} train.py \
    --config-name=train_dp_multifreq_cd_50hz_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path="${DATASET_PATH}" \
    task.name="${POLICY_TASK_NAME}" \
    exp_name="${EXP_NAME}" \
    prediction_type="${PREDICTION_TYPE}" \
    training.seed="${SEED}" \
    training.num_epochs=${NUM_EPOCHS} \
    training.stage1_ckpt_path="${STAGE1_CKPT_PATH}" \
    training.stage1_source_key="${STAGE1_SOURCE_KEY}" \
    training.lr_warmup_steps=${LR_WARMUP_STEPS} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    training.resume=${RESUME} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    policy.cd_timestep_grid="${CD_TIMESTEP_GRID}" \
    policy.cd_weight=${CD_WEIGHT} \
    policy.dsm_weight=${DSM_WEIGHT} \
    optimizer.lr=${POLICY_LR} \
    optimizer.obs_encoder_lr=${POLICY_LR} \
    ema.max_value=${EMA_MAX_VALUE} \
    ema.power=${EMA_POWER} \
    "$@"

exit 0
