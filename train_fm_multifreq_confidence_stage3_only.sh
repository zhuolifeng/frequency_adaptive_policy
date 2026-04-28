#!/bin/bash
#
# Stage-3-only training: Load a pretrained FM checkpoint (stage2_final or
# FMMultiFreqPolicy latest.ckpt), freeze FM backbone, train Confidence Head.
#
# Supports TWO scenarios:
#   A) From FMMultiFreqConfidencePolicy stage2_final.ckpt (has confidence head weights)
#   B) From FMMultiFreqPolicy latest.ckpt (NO confidence head, randomly initialized)
#
# Usage:
#   bash train_fm_multifreq_confidence_stage3_only.sh
#
# Or override the checkpoint path:
#   PRETRAINED_CKPT=/path/to/your/checkpoint.ckpt bash train_fm_multifreq_confidence_stage3_only.sh
#

set -e

#==========================================================================
# GPU Config
#==========================================================================
GPU_ID=${GPU_ID:-"0,1,2,3,4,5,6,7"}

#==========================================================================
# Task / Dataset Config
#==========================================================================
TASK="dual_switch"
DATASET_VERSION="v4.1"
ACTION_TYPE="right_arm_6DOF_virtual_target_stiffness"
TASK_CONFIG="real_${TASK}_image_wrench_offline_10fps_acp"

# Dataset path (used for normalizer computation and training data)
DATASET_PATH="${DATASET_PATH:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/${TASK}_${DATASET_VERSION}_round_1_zarr}"

#==========================================================================
# Pretrained Checkpoint Path
#   - Scenario A: stage2_final.ckpt from FMMultiFreqConfidencePolicy training
#   - Scenario B: latest.ckpt from FMMultiFreqPolicy training (no confidence head)
#==========================================================================
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/frequency_adaptive_policy/data/outputs/2026.04.20/06.34.30_train_fm_multifreq_confidence_workspace_dual_switch_v4_fm_multifreq_confidence/checkpoints/stage2_final.ckpt}"

#==========================================================================
# Stage 3 Training Config
#==========================================================================
STAGE3_EPOCHS=250

# Learning rate for confidence head
# (same as default optimizer.lr, only confidence head params are optimized)
CONFIDENCE_LR=3.0e-4

# LR warmup steps for stage 3
LR_WARMUP_STEPS=500

#==========================================================================
# Confidence Head Config
#==========================================================================
CONFIDENCE_GAMMA=1.0        # Weight on precision term: gamma * MSE * sigma
CONFIDENCE_ALPHA=0.2        # Weight on entropy bonus: -alpha * log(sigma)
CONFIDENCE_ACTION_DIMS=6    # Only first 6 dims (xyz+rpy) used for confidence
NORMALIZE_CONFIDENCE_LOSS=False

# Confidence-based inference thresholds
CONFIDENCE_THRESHOLD=0.0                      # For _predict_action
CONFIDENCE_THRESHOLD_FOR_HEAD_SELECTION=3.5   # For reactive runner head selection

#==========================================================================
# Other Training Config
#==========================================================================
BATCH_SIZE=64
TRAINING_DEBUG=False
CHECKPOINT_EVERY=10
VAL_EVERY=1
SEED=42

#==========================================================================
# Print config summary
#==========================================================================
POLICY_TASK_NAME="${TASK}_${DATASET_VERSION}_fm_multifreq_confidence_stage3"

echo "=========================================="
echo "  Stage-3-only Confidence Head Training"
echo "=========================================="
echo "  Pretrained ckpt:  ${PRETRAINED_CKPT}"
echo "  Dataset:          ${DATASET_PATH}"
echo "  Stage 3 epochs:   ${STAGE3_EPOCHS}"
echo "  Confidence LR:    ${CONFIDENCE_LR}"
echo "  LR warmup:        ${LR_WARMUP_STEPS}"
echo "  Batch size:       ${BATCH_SIZE}"
echo "  GPUs:             ${GPU_ID}"
echo ""
echo "  Confidence config:"
echo "    gamma=${CONFIDENCE_GAMMA}, alpha=${CONFIDENCE_ALPHA}"
echo "    action_dims=${CONFIDENCE_ACTION_DIMS}"
echo "    normalize_loss=${NORMALIZE_CONFIDENCE_LOSS}"
echo "    threshold=${CONFIDENCE_THRESHOLD}"
echo "    head_selection_threshold=${CONFIDENCE_THRESHOLD_FOR_HEAD_SELECTION}"
echo "=========================================="

#==========================================================================
# Validate checkpoint exists
#==========================================================================
if [ ! -f "${PRETRAINED_CKPT}" ]; then
    echo "ERROR: Pretrained checkpoint not found: ${PRETRAINED_CKPT}"
    echo ""
    echo "Please set PRETRAINED_CKPT to one of:"
    echo "  - stage2_final.ckpt from a FMMultiFreqConfidencePolicy run"
    echo "  - latest.ckpt from a FMMultiFreqPolicy run (no confidence head)"
    echo ""
    echo "Example:"
    echo "  PRETRAINED_CKPT=/path/to/checkpoints/stage2_final.ckpt bash $0"
    exit 1
fi

#==========================================================================
# Launch training
#   stage1_epochs=0, stage2_epochs=0 → Direct Stage 3 mode
#   The workspace will:
#     1. Load pretrained ckpt (strict=False, handles missing confidence_head)
#     2. Freeze FM model, obs_encoder, rnn_lowfreq, rnn_highfreq
#     3. Rebuild optimizer with only confidence_head params
#     4. Train for stage3_epochs
#     5. Save latest.ckpt and stage3_final.ckpt
#==========================================================================
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_fm_multifreq_confidence_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    training.pretrained_ckpt_path=${PRETRAINED_CKPT} \
    training.stage1_epochs=0 \
    training.stage2_epochs=0 \
    training.stage3_epochs=${STAGE3_EPOCHS} \
    training.num_epochs=${STAGE3_EPOCHS} \
    training.seed=${SEED} \
    training.debug=${TRAINING_DEBUG} \
    training.resume=False \
    training.use_amp=False \
    training.amp_dtype="bf16" \
    training.lr_warmup_steps=${LR_WARMUP_STEPS} \
    training.checkpoint_every=${CHECKPOINT_EVERY} \
    training.val_every=${VAL_EVERY} \
    training.use_ema=True \
    optimizer.lr=${CONFIDENCE_LR} \
    dataloader.batch_size=${BATCH_SIZE} \
    val_dataloader.batch_size=${BATCH_SIZE} \
    policy.training_stage=1 \
    policy.use_confidence_head=True \
    policy.confidence_head_type=transformer \
    policy.confidence_gamma=${CONFIDENCE_GAMMA} \
    policy.confidence_alpha=${CONFIDENCE_ALPHA} \
    policy.confidence_action_dims=${CONFIDENCE_ACTION_DIMS} \
    policy.normalize_confidence_loss=${NORMALIZE_CONFIDENCE_LOSS} \
    policy.confidence_threshold=${CONFIDENCE_THRESHOLD} \
    policy.confidence_threshold_for_head_selection=${CONFIDENCE_THRESHOLD_FOR_HEAD_SELECTION} \
    policy.use_confidence_based_head_selection=True \
    policy.use_amp_for_inference=False \
    policy.amp_dtype="bf16" \

exit 0
