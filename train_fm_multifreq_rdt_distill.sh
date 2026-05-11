#!/bin/bash

#--------------------------------
# Config
GPU_ID=0,1,2,3,4,5,6,7

TASK="dual_switch"
DATASET_VERSION="v6"
ACTION_TYPE="right_arm_6DOF_virtual_target_stiffness"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_acp"

# RDT-style distillation control
USE_DISTILLATION=True

# Training epochs
STAGE1_EPOCHS=700    # FM velocity matching (10hz + 50hz)
STAGE2_EPOCHS=300    # 50hz endpoint distillation + 10hz FM

TOTAL_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS))

DATASET_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_1_zarr"

USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
TRAINING_DEBUG=False
POLICY_LR=1.0e-4
STAGE2_LR=1.0e-5
DISTILL_GT_WEIGHT=0
TEACHER_STEPS=50
TEACHER_SOLVER="euler"
USE_SKIP_CONNECTION=True
#--------------------------------

echo "=========================================="
echo "RDT-style multi-frequency FM training"
echo "  Use distillation: ${USE_DISTILLATION}"
echo "  Use skip connection: ${USE_SKIP_CONNECTION}"
echo "  Stage 1: ${STAGE1_EPOCHS} epochs (10Hz + 50Hz FM velocity matching)"
if [ "${USE_DISTILLATION}" = "True" ]; then
echo "  Stage 2: ${STAGE2_EPOCHS} epochs (10Hz FM + 50Hz endpoint distillation)"
fi
echo "  Total:   ${TOTAL_EPOCHS} epochs"
echo "  Dataset: ${DATASET_PATH}"
echo "=========================================="

POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_fm_multifreq_rdt_distill
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --main_process_port 29501 train.py \
    --config-name=train_fm_multifreq_rdt_distill_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    policy.teacher_num_inference_steps=${TEACHER_STEPS} \
    policy.teacher_solver=${TEACHER_SOLVER} \
    policy.distill_gt_weight=${DISTILL_GT_WEIGHT} \
    training.num_epochs=${TOTAL_EPOCHS} \
    training.use_distillation=${USE_DISTILLATION} \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    training.freeze_condition_modules_stage2=True \
    policy.training_stage=1 \
    optimizer.lr=${POLICY_LR} \
    optimizer.stage2_lr=${STAGE2_LR} \
    policy.model.use_skip_connection=${USE_SKIP_CONNECTION}

exit 0
