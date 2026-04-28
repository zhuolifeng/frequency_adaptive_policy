#!/bin/bash

#--------------------------------
# Config
GPU_ID=0,1,2,3,4,5,6,7

TASK="dual_switch"
STAGE1_DATASET_VERSION="v4"
STAGE2_DATASET_VERSION="v4.1"
ACTION_TYPE="right_arm_6DOF_virtual_target_stiffness"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_acp"

# Stage 2: whether to keep 10hz FM loss (set False to train CD only)
STAGE2_USE_10HZ_LOSS=False

# Stage 2: optional GT delta auxiliary loss on 50hz f_student
# Reduces action chunk jitter by penalizing inconsistent adjacent steps
STAGE2_USE_DELTA_LOSS=False
STAGE2_DELTA_LOSS_WEIGHT=0.1

# Stage 2: CD discretization scales (higher = finer Teacher ODE steps = more precise)
CD_NUM_SCALES=40

# Stage 2: EMA rate for CD Target model (higher = more stable, slower tracking)
STAGE2_EMA_RATE=0.9999

# Three-stage training epochs
STAGE1_EPOCHS=600   # FM velocity matching (10hz + 50hz)
STAGE2_EPOCHS=200   # CD distillation for 50hz
STAGE3_EPOCHS=250   # Confidence head training
TOTAL_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS))

STAGE1_DATASET_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${STAGE1_DATASET_VERSION}_round_1_zarr"
STAGE2_DATASET_PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${STAGE2_DATASET_VERSION}_round_1_zarr"

USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
TRAINING_DEBUG=False
POLICY_LR=2.0e-4
#--------------------------------

echo "=========================================="
echo "Multi-frequency FM + Confidence three-stage training"
echo "  Stage2 10hz loss: ${STAGE2_USE_10HZ_LOSS}"
echo "  Stage2 delta loss: ${STAGE2_USE_DELTA_LOSS} (weight=${STAGE2_DELTA_LOSS_WEIGHT})"
echo "  Stage2 CD scales: ${CD_NUM_SCALES}"
echo "  Stage2 EMA rate:  ${STAGE2_EMA_RATE}"
echo "  Stage 1: ${STAGE1_EPOCHS} epochs (FM velocity matching)"
echo "  Stage 2: ${STAGE2_EPOCHS} epochs (CD distillation for 50hz)"
echo "  Stage 3: ${STAGE3_EPOCHS} epochs (Confidence head on 10hz one-step FM MSE)"
echo "  Total:   ${TOTAL_EPOCHS} epochs"
echo "  Stage 1 dataset: ${STAGE1_DATASET_PATH}"
echo "  Stage 2 dataset: ${STAGE2_DATASET_PATH}"
echo "=========================================="

POLICY_TASK_NAME=${TASK}_${STAGE1_DATASET_VERSION}_fm_multifreq_confidence
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_fm_multifreq_confidence_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${STAGE1_DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    policy.stage2_use_10hz_loss=${STAGE2_USE_10HZ_LOSS} \
    policy.stage2_use_delta_loss=${STAGE2_USE_DELTA_LOSS} \
    policy.stage2_delta_loss_weight=${STAGE2_DELTA_LOSS_WEIGHT} \
    policy.cd_num_scales=${CD_NUM_SCALES} \
    training.stage2_ema_rate=${STAGE2_EMA_RATE} \
    training.num_epochs=${TOTAL_EPOCHS} \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.stage3_epochs=${STAGE3_EPOCHS} \
    training.stage2_dataset_path=${STAGE2_DATASET_PATH} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    policy.training_stage=1 \
    optimizer.lr=${POLICY_LR} \

exit 0
