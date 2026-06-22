#!/bin/bash

GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29506}"

TASK="${TASK:-dual_button}"
DATASET_VERSION="${DATASET_VERSION:-v2}"
ROUND="${ROUND:-1}"
DATASET_PATH="${DATASET_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_30hz"
SEED="${SEED:-42}"

EXP_NAME="${EXP_NAME:-dp_30hz_consistency}"
POLICY_TASK_NAME="${TASK}_${DATASET_VERSION}_dp_30hz_consistency"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-600}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
NUM_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS))

STAGE1_LR="${STAGE1_LR:-1.0e-4}"
STAGE1_OBS_ENCODER_LR="${STAGE1_OBS_ENCODER_LR:-3.0e-5}"
STAGE2_LR="${STAGE2_LR:-1.0e-5}"

CONSISTENCY_NUM_INFERENCE_STEPS="${CONSISTENCY_NUM_INFERENCE_STEPS:-1}"
SIGMA_DATA="${SIGMA_DATA:-0.5}"
SIGMA_MIN="${SIGMA_MIN:-0.02}"
SIGMA_MAX="${SIGMA_MAX:-80.0}"
RHO="${RHO:-7.0}"
NUM_SCALES="${NUM_SCALES:-40}"
CTM_ODE_STEPS_MAX="${CTM_ODE_STEPS_MAX:-1}"
WEIGHT_SCHEDULE="${WEIGHT_SCHEDULE:-karras}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
DSM_WEIGHT="${DSM_WEIGHT:-1.0}"
SIGMA_SAMPLE_P_MEAN="${SIGMA_SAMPLE_P_MEAN:--1.2}"
SIGMA_SAMPLE_P_STD="${SIGMA_SAMPLE_P_STD:-1.2}"
DELTA="${DELTA:--1.0}"

USE_AMP_FOR_POLICY="${USE_AMP_FOR_POLICY:-False}"
AMP_DTYPE_FOR_POLICY="${AMP_DTYPE_FOR_POLICY:-bf16}"
TRAINING_DEBUG="${TRAINING_DEBUG:-False}"
RESUME="${RESUME:-False}"
PRETRAINED_CKPT_PATH="${PRETRAINED_CKPT_PATH:-null}"
LOGGING_BACKEND="${LOGGING_BACKEND:-tensorboard}"
LOGGING_MODE="${LOGGING_MODE:-online}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_ID}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ARRAY[@]}}"
ACCELERATE_ARGS=(--main_process_port "${MAIN_PROCESS_PORT}")
if [ "${NUM_PROCESSES}" -gt 1 ]; then
    ACCELERATE_ARGS=(--multi_gpu --num_processes "${NUM_PROCESSES}" "${ACCELERATE_ARGS[@]}")
fi

echo "=========================================="
echo "DP 30Hz Consistency Training"
echo "=========================================="
echo "  Task config:       ${TASK_CONFIG}"
echo "  Dataset:           ${DATASET_PATH}"
echo "  Stage epochs:      ${STAGE1_EPOCHS}/${STAGE2_EPOCHS}"
echo "  LR:                ${STAGE1_LR}/${STAGE2_LR}"
echo "  Num scales:        ${NUM_SCALES}"
echo "  CTM ODE steps:     ${CTM_ODE_STEPS_MAX}"
echo "  Weights CM/DSM:    ${CONSISTENCY_WEIGHT}/${DSM_WEIGHT}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch "${ACCELERATE_ARGS[@]}" train.py \
    --config-name=train_dp_30hz_consistency_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path="${DATASET_PATH}" \
    task.name="${POLICY_TASK_NAME}" \
    exp_name="${EXP_NAME}" \
    training.seed="${SEED}" \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.num_epochs=${NUM_EPOCHS} \
    training.pretrained_ckpt_path=${PRETRAINED_CKPT_PATH} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    training.resume=${RESUME} \
    optimizer.stage1_lr=${STAGE1_LR} \
    optimizer.stage1_obs_encoder_lr=${STAGE1_OBS_ENCODER_LR} \
    optimizer.stage2_lr=${STAGE2_LR} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    consistency.num_inference_steps=${CONSISTENCY_NUM_INFERENCE_STEPS} \
    consistency.sigma_data=${SIGMA_DATA} \
    consistency.sigma_min=${SIGMA_MIN} \
    consistency.sigma_max=${SIGMA_MAX} \
    consistency.rho=${RHO} \
    consistency.num_scales=${NUM_SCALES} \
    consistency.ctm_ode_steps_max=${CTM_ODE_STEPS_MAX} \
    consistency.weight_schedule=${WEIGHT_SCHEDULE} \
    consistency.consistency_weight=${CONSISTENCY_WEIGHT} \
    consistency.dsm_weight=${DSM_WEIGHT} \
    consistency.sigma_sample_p_mean=${SIGMA_SAMPLE_P_MEAN} \
    consistency.sigma_sample_p_std=${SIGMA_SAMPLE_P_STD} \
    consistency.delta=${DELTA} \
    logging.backend=${LOGGING_BACKEND} \
    logging.mode=${LOGGING_MODE} \
    "$@"

exit 0
