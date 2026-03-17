#!/bin/bash

TASK="dual_switch"
DATASET_VERSION="v3.1"
ROUND="1"

# ============================================================
# 切换策略：只需修改 CONFIG_NAME 一处即可
# 可选值（与 ImplictRDP/config/ 下的 yaml 文件名对应）：
#   train_act_confidence_dual_encoder_stagewise_workspace
#   train_act_confidence_dual_encoder_workspace
#   train_act_confidence_workspace
#   train_act_confidence_stagewise_workspace
#   train_hybrid_diffusion_act_workspace
#   train_hybrid_diffusion_act_stagewise_workspace
# ============================================================
CONFIG_NAME="train_act_confidence_stagewise_workspace"

# ============================================================
# Checkpoint path（根据使用的策略切换）
# ============================================================
CKPT_PATH="/mnt/workspace/users/lifeng/reactive_transformer_diffusion_policy/data/outputs/2026.03.16/22.42-07_dual_switch_v3_to_v3_4_act_confidence_stagewise_policy_round_1_stage2/checkpoints/latest.ckpt"

# Dataset path
DATASET_PATH="/mnt/data/users/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr"

# GPU device
GPU_DEVICE="cuda:6"

# Video settings
VIDEO_FPS=1
NUM_INFERENCE_STEPS=8

# Output directory for videos
OUTPUT_DIR="evaluations_stagewise_800_250_collabation_act"

# ============================================================
# 根据 CONFIG_NAME 自动推导 transforms override，关闭随机数据增强
#
# 双编码器策略（dual_encoder_stagewise / dual_encoder）：
#   YAML 中 act_encoder 与 confidence_encoder 各自含
#   RandomRotation + ColorJitter，需分别 override 两个键。
#
# 单编码器策略（act_confidence / hybrid_diffusion_act / act_confidence_stagewise / hybrid_diffusion_act_stagewise）：
# ============================================================
if [ "$CONFIG_NAME" = "train_act_confidence_dual_encoder_stagewise_workspace" ] || \
   [ "$CONFIG_NAME" = "train_act_confidence_dual_encoder_workspace" ]; then
    TRANSFORMS_OVERRIDE=(
        'policy.act_encoder.transforms=[{type: CenterCrop}]'
        'policy.confidence_encoder.transforms=[{type: CenterCrop}]'
    )
else
    TRANSFORMS_OVERRIDE=(
        'policy.obs_encoder.transforms=[{type: CenterCrop}]'
    )
fi

echo "=========================================="
echo "Model evaluation - supports HybridDiffusionACTPolicy and ACTConfidencePolicy"
echo "=========================================="
echo "Config:    ${CONFIG_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset:   ${DATASET_PATH}"
echo "GPU device: ${GPU_DEVICE}"
echo "Diffusion inference steps: ${NUM_INFERENCE_STEPS}"
echo "Video FPS: ${VIDEO_FPS}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

export CKPT_PATH="${CKPT_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR}"
python evaluate_confidence.py \
    --config-name ${CONFIG_NAME} \
    task=real_${TASK}_image_wrench_offline_rl_10fps_acp \
    task.dataset_path=${DATASET_PATH} \
    +eval_batch_size=64 \
    +use_full_dataset=True \
    +num_inference_steps=${NUM_INFERENCE_STEPS} \
    +video_fps=${VIDEO_FPS} \
    +device=${GPU_DEVICE} \
    task.dataset.val_ratio=0.0 \
    task.dataset.only_use_success_episodes=False \
    task.dataset.use_noisy_episodes=False \
    task.dataset.resume_params_path=null \
    "${TRANSFORMS_OVERRIDE[@]}"

echo ""
echo "=========================================="
echo "Complete!"
