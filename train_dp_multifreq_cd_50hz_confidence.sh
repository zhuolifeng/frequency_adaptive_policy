#!/bin/bash

GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"

TASK="${TASK:-dual_switch}"
DATASET_VERSION="${DATASET_VERSION:-v6}"
ROUND="${ROUND:-1}"
DATASET_PATH="${DATASET_PATH:-/mnt/workspace/users/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_acp"
SEED="${SEED:-42}"

EXP_NAME="${EXP_NAME:-dp_multifreq_cd_50hz_confidence}"
POLICY_TASK_NAME="${TASK}_${DATASET_VERSION}_dp_multifreq_cd_50hz_confidence"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-600}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-250}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-100}"

STAGE1_LR="${STAGE1_LR:-1.0e-4}"
STAGE1_OBS_ENCODER_LR="${STAGE1_OBS_ENCODER_LR:-3.0e-5}"
STAGE2_CONFIDENCE_LR="${STAGE2_CONFIDENCE_LR:-1.0e-4}"
STAGE3_LR="${STAGE3_LR:-1.0e-5}"

CONFIDENCE_N_SAMPLES="${CONFIDENCE_N_SAMPLES:-8}"
CONFIDENCE_NUM_INFERENCE_STEPS="${CONFIDENCE_NUM_INFERENCE_STEPS:-3}"

USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
RESUME="${RESUME:-False}"
PRETRAINED_CKPT_PATH="${PRETRAINED_CKPT_PATH:-null}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29504}"

NUM_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS))

echo "=========================================="
echo "DP MultiFreq 50Hz CD Confidence Training"
echo "=========================================="
echo "  Dataset:      ${DATASET_PATH}"
echo "  Stage epochs: ${STAGE1_EPOCHS}/${STAGE2_EPOCHS}/${STAGE3_EPOCHS}"
echo "  LR:           ${STAGE1_LR}/${STAGE2_CONFIDENCE_LR}/${STAGE3_LR}"
echo "  Confidence:   samples=${CONFIDENCE_N_SAMPLES}, steps=${CONFIDENCE_NUM_INFERENCE_STEPS}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --main_process_port ${MAIN_PROCESS_PORT} train.py \
    --config-name=train_dp_multifreq_cd_50hz_confidence_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path="${DATASET_PATH}" \
    task.name="${POLICY_TASK_NAME}" \
    exp_name="${EXP_NAME}" \
    training.seed="${SEED}" \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.stage3_epochs=${STAGE3_EPOCHS} \
    training.num_epochs=${NUM_EPOCHS} \
    training.pretrained_ckpt_path=${PRETRAINED_CKPT_PATH} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    training.resume=${RESUME} \
    optimizer.stage1_lr=${STAGE1_LR} \
    optimizer.stage1_obs_encoder_lr=${STAGE1_OBS_ENCODER_LR} \
    optimizer.stage2_confidence_lr=${STAGE2_CONFIDENCE_LR} \
    optimizer.stage3_lr=${STAGE3_LR} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    policy.confidence_n_samples=${CONFIDENCE_N_SAMPLES} \
    policy.confidence_num_inference_steps=${CONFIDENCE_NUM_INFERENCE_STEPS} \
    "$@"
