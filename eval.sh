#!/usr/bin/env bash
set -euo pipefail

# Real robot evaluation. Choose the workspace and task matching your checkpoint.
# For a 30 Hz iMeanFlow run, for example:
#   CONFIG_NAME=train_dp_30hz_imeanflow_workspace \
#   TASK_CONFIG=real_dual_switch_image_wrench_offline_10fps_30hz \
#   CKPT_PATH=data/checkpoints/latest.ckpt bash eval.sh task.env_runner.action_type=right_arm_6DOF
: "${CONFIG_NAME:?Set CONFIG_NAME to the training workspace used for the checkpoint}"
: "${TASK_CONFIG:?Set TASK_CONFIG to the task with your configured robot runner}"
: "${CKPT_PATH:?Set CKPT_PATH to the policy checkpoint}"
export CKPT_PATH
export OUTPUT_DIR="${OUTPUT_DIR:-data/evaluations/${TASK_CONFIG}}"

exec python eval_real_robot_flexiv.py \
    --config-name "$CONFIG_NAME" \
    "task=$TASK_CONFIG" \
    '++ckpt_path=${oc.env:CKPT_PATH}' \
    '++task.env_runner.output_dir=${oc.env:OUTPUT_DIR}' \
    "$@"
