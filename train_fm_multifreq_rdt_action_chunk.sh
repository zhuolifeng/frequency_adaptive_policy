#!/bin/bash

#--------------------------------
# Config
GPU_ID=0,1,2,3,4,5,6,7
NUM_PROCESSES=$(awk -F, '{print NF}' <<< "${GPU_ID}")

TASK="dual_switch"
DATASET_VERSION="v6"
ACTION_TYPE="right_arm_6DOF_virtual_target_stiffness"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_acp"

# RDT-style distillation control
USE_DISTILLATION="${USE_DISTILLATION:-True}"

# Training epochs
STAGE1_EPOCHS="${STAGE1_EPOCHS:-800}"    # x0 prediction loss (10hz + 50hz)
STAGE2_EPOCHS="${STAGE2_EPOCHS:-300}"    # 50hz x0 distillation + 10hz x0 loss

TOTAL_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS))

DATASET_PATH="/mnt/data/users/lifeng/${TASK}_${DATASET_VERSION}_round_1_zarr"

USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
TRAINING_DEBUG=False
POLICY_LR=1.0e-4
STAGE2_LR=1.0e-5
TEACHER_STEPS=10
TEACHER_SOLVER="${TEACHER_SOLVER:-euler}"
DISTILL_GT_WEIGHT=1
LOSS_TYPE="${LOSS_TYPE:-velocity}"
STAGE1_CKPT_PATH="${STAGE1_CKPT_PATH:-null}"
#--------------------------------

echo "=========================================="
echo "RDT-style multi-frequency FM training"
echo "  Use distillation: ${USE_DISTILLATION}"
echo "  Stage 1: ${STAGE1_EPOCHS} epochs (10Hz + 50Hz x0 prediction)"
if [ "${USE_DISTILLATION}" = "True" ]; then
echo "  Stage 2: ${STAGE2_EPOCHS} epochs (10Hz x0 + 50Hz x0 distillation)"
fi
echo "  Total:   ${TOTAL_EPOCHS} epochs"
echo "  Dataset: ${DATASET_PATH}"
echo "  Teacher solver: ${TEACHER_SOLVER}"
echo "  Loss type: ${LOSS_TYPE}"
echo "  Stage 1 ckpt path: ${STAGE1_CKPT_PATH}"
echo "  GPUs: ${GPU_ID} (${NUM_PROCESSES} processes)"
echo "=========================================="

POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_fm_multifreq_rdt_action_chunk
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --num_processes ${NUM_PROCESSES} --main_process_port 29502 train.py \
    --config-name=train_fm_multifreq_rdt_action_chunk_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    policy.teacher_num_inference_steps=${TEACHER_STEPS} \
    policy.teacher_solver=${TEACHER_SOLVER} \
    policy.loss_type=${LOSS_TYPE} \
    +policy.distill_gt_weight=${DISTILL_GT_WEIGHT} \
    training.num_epochs=${TOTAL_EPOCHS} \
    training.use_distillation=${USE_DISTILLATION} \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.stage1_ckpt_path=${STAGE1_CKPT_PATH} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    training.freeze_condition_modules_stage2=True \
    policy.training_stage=1 \
    optimizer.lr=${POLICY_LR} \
    optimizer.stage2_lr=${STAGE2_LR}

exit 0
