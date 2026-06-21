#!/bin/bash

GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29505}"

TASK="${TASK:-dual_button}"
DATASET_VERSION="${DATASET_VERSION:-v2}"
ROUND="${ROUND:-1}"
DATASET_PATH="${DATASET_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_30hz"
SEED="${SEED:-42}"

EXP_NAME="${EXP_NAME:-dp_30hz_imeanflow}"
POLICY_TASK_NAME="${TASK}_${DATASET_VERSION}_dp_30hz_imeanflow"
NUM_EPOCHS="${NUM_EPOCHS:-800}"
POLICY_LR="${POLICY_LR:-1.0e-4}"
OBS_ENCODER_LR="${OBS_ENCODER_LR:-3.0e-5}"

IMEANFLOW_NUM_INFERENCE_STEPS="${IMEANFLOW_NUM_INFERENCE_STEPS:-1}"
IMEANFLOW_P_MEAN="${IMEANFLOW_P_MEAN:--0.4}"
IMEANFLOW_P_STD="${IMEANFLOW_P_STD:-1.0}"
IMEANFLOW_DATA_PROPORTION="${IMEANFLOW_DATA_PROPORTION:-0.5}"

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
echo "DP 30Hz iMeanFlow Training"
echo "=========================================="
echo "  Task config:       ${TASK_CONFIG}"
echo "  Dataset:           ${DATASET_PATH}"
echo "  Epochs:            ${NUM_EPOCHS}"
echo "  LR:                ${POLICY_LR}"
echo "  iMeanFlow steps:   ${IMEANFLOW_NUM_INFERENCE_STEPS}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch "${ACCELERATE_ARGS[@]}" train.py \
    --config-name=train_dp_30hz_imeanflow_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path="${DATASET_PATH}" \
    task.name="${POLICY_TASK_NAME}" \
    exp_name="${EXP_NAME}" \
    training.seed="${SEED}" \
    training.num_epochs=${NUM_EPOCHS} \
    training.pretrained_ckpt_path=${PRETRAINED_CKPT_PATH} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    training.resume=${RESUME} \
    optimizer.lr=${POLICY_LR} \
    optimizer.obs_encoder_lr=${OBS_ENCODER_LR} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    meanflow.num_inference_steps=${IMEANFLOW_NUM_INFERENCE_STEPS} \
    meanflow.p_mean=${IMEANFLOW_P_MEAN} \
    meanflow.p_std=${IMEANFLOW_P_STD} \
    meanflow.data_proportion=${IMEANFLOW_DATA_PROPORTION} \
    logging.backend=${LOGGING_BACKEND} \
    logging.mode=${LOGGING_MODE} \
    "$@"

exit 0
