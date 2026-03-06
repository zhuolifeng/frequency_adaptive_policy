TASK="dual_switch"
DATASET_VERSION="v3.1"
ROUND="1"

#Checkpoint path
CKPT_PATH="/home/lifeng/reactive_transformer_diffusion_policy_copy/data/outputs/2026.02.24/22.46:34_train_dual_switch_act_confidence_dual_encoder_workspace_dual_switch_v3_act_confidence_dual_encoder_policy_round_1/checkpoints/latest.ckpt"

#Dataset path
DATASET_PATH="/mnt/data/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr"

#GPU device
GPU_DEVICE="cuda:0"

#Video settings
VIDEO_FPS=1
NUM_INFERENCE_STEPS=8

echo "=========================================="
echo "Model evaluation - supports HybridDiffusionACTPolicy and ACTConfidencePolicy"
echo "=========================================="
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "GPU device: ${GPU_DEVICE}"
echo "Diffusion inference steps: ${NUM_INFERENCE_STEPS}"
echo "Video FPS: ${VIDEO_FPS}"
echo ""

export CKPT_PATH="${CKPT_PATH}"
python eval_confidence.py \
    --config-name train_act_confidence_dual_encoder_workspace \
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
    task.dataset.resume_params_path=null

echo ""
echo "=========================================="
echo "Complete!"


