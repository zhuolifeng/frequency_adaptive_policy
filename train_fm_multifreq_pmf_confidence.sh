#!/bin/bash

# Two-stage multi-frequency PMF + confidence training.

GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"

TASK="${TASK:-dual_switch}"
STAGE1_DATASET_VERSION="${STAGE1_DATASET_VERSION:-v4.2}"
STAGE2_DATASET_VERSION="${STAGE2_DATASET_VERSION:-v6.1}"
ROUND="${ROUND:-1}"
TASK_CONFIG="${TASK_CONFIG:-real_${TASK}_image_wrench_offline_10fps_acp}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-4}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-4}"

STAGE1_LR="${STAGE1_LR:-1.0e-4}"
STAGE1_OBS_ENCODER_LR="${STAGE1_OBS_ENCODER_LR:-3.0e-5}"
STAGE2_CONFIDENCE_LR="${STAGE2_CONFIDENCE_LR:-1.0e-4}"

DATASET_PATH="${DATASET_PATH:-/mnt/workspace/users/lifeng/${TASK}_${STAGE1_DATASET_VERSION}_round_${ROUND}_zarr}"
STAGE2_DATASET_PATH="${STAGE2_DATASET_PATH:-/mnt/workspace/users/lifeng/${TASK}_${STAGE2_DATASET_VERSION}_round_${ROUND}_zarr}"

USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
CONFIDENCE_N_SAMPLES="${CONFIDENCE_N_SAMPLES:-1}"
CONFIDENCE_NUM_INFERENCE_STEPS="${CONFIDENCE_NUM_INFERENCE_STEPS:-3}"

POLICY_TASK_NAME="${POLICY_TASK_NAME:-${TASK}_${STAGE1_DATASET_VERSION}_pmf_confidence}"

echo "=========================================="
echo "Multi-frequency PMF Confidence training"
echo "  Stage 1 epochs: ${STAGE1_EPOCHS}"
echo "  Stage 2 epochs: ${STAGE2_EPOCHS}"
echo "  Stage 1 dataset: ${DATASET_PATH}"
echo "  Stage 2 dataset: ${STAGE2_DATASET_PATH}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_fm_multifreq_pmf_confidence_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    exp_name=pmf_confidence \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.num_epochs=$((STAGE1_EPOCHS + STAGE2_EPOCHS)) \
    training.stage2_dataset_path=${STAGE2_DATASET_PATH} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    training.resume=False \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    policy.confidence_n_samples=${CONFIDENCE_N_SAMPLES} \
    policy.confidence_num_inference_steps=${CONFIDENCE_NUM_INFERENCE_STEPS} \
    optimizer.stage1_lr=${STAGE1_LR} \
    optimizer.stage1_obs_encoder_lr=${STAGE1_OBS_ENCODER_LR} \
    optimizer.stage2_confidence_lr=${STAGE2_CONFIDENCE_LR}

exit 0
