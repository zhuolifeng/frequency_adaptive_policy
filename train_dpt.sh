#!/usr/bin/env bash
set -eo pipefail

# Stage 0: Post-process data
TASK="${TASK:-flip}"
DATASET_TAG="${DATASET_TAG:-${TASK}_v2}"
if [ -z "${DATASET_PATH:-}" ]; then
    python post_process_data.py --tag "$DATASET_TAG" --action_type right_arm_6DOF_wrench --skip_static_frames --resume
    DATASET_PATH="data/${DATASET_TAG}_zarr"
fi
export DATASET_PATH

# Stage 1: Train Policy
echo "Stage 1: training Policy..."
bash ./launch_training.sh \
    --config-name=train_diffusion_transformer_real_image_workspace \
    "task=real_${TASK}_image_wrench_dpt_10fps" \
    'task.dataset_path=${oc.env:DATASET_PATH}' \
    "task.name=real_${TASK}_image_wrench_dpt_10fps_v2" \
    policy.noise_scheduler.prediction_type=v_prediction \
    "logging.mode=${LOGGING_MODE:-online}" \
    "$@"
