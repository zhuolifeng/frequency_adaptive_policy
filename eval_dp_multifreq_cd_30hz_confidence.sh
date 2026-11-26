#!/usr/bin/env bash
set -eo pipefail

TASK="${TASK:-dual_button}"
DATASET_VERSION="${DATASET_VERSION:-v1.1}"
ROUND="${ROUND:-1}"

CONFIG_NAME="${CONFIG_NAME:-train_dp_multifreq_cd_30hz_confidence_workspace}"
: "${CKPT_PATH:?Set CKPT_PATH to the policy checkpoint}"
export DATASET_PATH="${DATASET_PATH:-data/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr}"

GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
VIDEO_FPS="${VIDEO_FPS:-1}"
NUM_INFERENCE_STEPS_10HZ="${NUM_INFERENCE_STEPS_10HZ:-3}"
NUM_INFERENCE_STEPS_30HZ="${NUM_INFERENCE_STEPS_30HZ:-3}"
PREDICTION_TYPE="${PREDICTION_TYPE:-sample}"
LOSS_TYPE="${LOSS_TYPE:-eps}"
TRAINING_STAGE="${TRAINING_STAGE:-2}"
CONFIDENCE_SELECTION="${CONFIDENCE_SELECTION:-True}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-3.5}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations_dp_multifreq_cd_30hz_confidence/10hz_${NUM_INFERENCE_STEPS_10HZ}steps_30hz_${NUM_INFERENCE_STEPS_30HZ}steps_${TRAINING_STAGE}stage}"

TRAINING_STAGE_ARGS=()
if [ "${TRAINING_STAGE}" != "auto" ]; then
    TRAINING_STAGE_ARGS=("+training_stage=${TRAINING_STAGE}")
fi

echo "=========================================="
echo "DP MultiFreq 30Hz CD Confidence Evaluation"
echo "=========================================="
echo "Config:     ${CONFIG_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset:    ${DATASET_PATH}"
echo "Stage:      ${TRAINING_STAGE}"
echo "10Hz steps: ${NUM_INFERENCE_STEPS_10HZ}"
echo "30Hz steps: ${NUM_INFERENCE_STEPS_30HZ}"
echo "Loss type:  ${LOSS_TYPE}"
echo "Confidence selection: ${CONFIDENCE_SELECTION}"
echo "Output dir: ${OUTPUT_DIR}"

export CKPT_PATH
export OUTPUT_DIR

python eval_dp_multifreq_cd_30hz_confidence.py \
    --config-name "${CONFIG_NAME}" \
    "task=real_${TASK}_image_wrench_offline_10fps_acp" \
    'task.dataset_path=${oc.env:DATASET_PATH}' \
    "prediction_type=${PREDICTION_TYPE}" \
    "loss_type=${LOSS_TYPE}" \
    "${TRAINING_STAGE_ARGS[@]}" \
    "+use_full_dataset=True" \
    "+num_inference_steps_10hz=${NUM_INFERENCE_STEPS_10HZ}" \
    "+num_inference_steps_30hz=${NUM_INFERENCE_STEPS_30HZ}" \
    "+video_fps=${VIDEO_FPS}" \
    "+device=${GPU_DEVICE}" \
    "policy.use_confidence_based_head_selection=${CONFIDENCE_SELECTION}" \
    "policy.confidence_threshold_for_head_selection=${CONFIDENCE_THRESHOLD}" \
    "task.dataset.val_ratio=0.0" \
    "policy.obs_encoder.transforms=[{type: CenterCrop}]" \
    "$@"
