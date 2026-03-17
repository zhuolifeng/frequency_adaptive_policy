#!/bin/bash

#--------------------------------
# Config
NUM_GPUS=8  # Change this to use different number of GPUs: 1, 2, 4, 8

TASK="dual_switch"
STAGE1_DATASET_VERSION="v3"
STAGE2_DATASET_VERSION="v3.4"
ACTION_TYPE="right_arm_6DOF_virtual_target_stiffness"
TRAINING_DEBUG=False
USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
LOGGING_MODE="online"

# Policy: "act_confidence_stagewise"、"act_confidence_dual_encoder_stagewise"、"hybrid_stagewise"
POLICY_TYPE="act_confidence_stagewise"

ROUND=1
POLICY_PRETRAINED_CKPT_PATH="null"
REWARD_PRETRAINED_CKPT_PATH="null"
CRITIC_PRETRAINED_CKPT_PATH="null"
DATASET_RESUME_PARAMS_PATH="null"

# Stagewise training config
STAGE1_EPOCHS=800
STAGE2_EPOCHS=250 
STAGE3_EPOCHS=250 #only for hybrid_stagewise
TOTAL_EPOCHS=$((STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS))

STAGE1_RAW_DATASET_DIR="/mnt/data/users/lifeng/${TASK}_${STAGE1_DATASET_VERSION}_round_${ROUND}"
STAGE2_RAW_DATASET_DIR="/mnt/data/users/lifeng/${TASK}_${STAGE2_DATASET_VERSION}_round_${ROUND}"
STAGE1_DATASET_PATH="${STAGE1_RAW_DATASET_DIR}_zarr"
STAGE2_DATASET_PATH="${STAGE2_RAW_DATASET_DIR}_zarr"
#--------------------------------

# if round <= 1, use offline RL config
if [ ${ROUND} -le 1 ]; then
    echo "Training offline RL"
    RL_TYPE="offline"
    TASK_CONFIG="real_${TASK}_image_wrench_offline_rl_10fps_acp"
    POLICY_NUM_EPOCHS=${TOTAL_EPOCHS}
    REWARD_NUM_EPOCHS=200
    CRITIC_NUM_EPOCHS=200
    POLICY_LR=1.0e-4
    REWARD_LR=1.0e-4
    CRITIC_LR=1.0e-4
else
    echo "Training online RL"
    RL_TYPE="online"
    TASK_CONFIG="real_${TASK}_image_wrench_online_rl_10fps_acp"
    POLICY_NUM_EPOCHS=200
    REWARD_NUM_EPOCHS=200
    CRITIC_NUM_EPOCHS=200
    POLICY_LR=1.0e-5
    REWARD_LR=1.0e-5
    CRITIC_LR=1.0e-5
fi

SEARCH_PATH="./data/outputs"

post_process_dataset() {
    local dataset_version="$1"
    local raw_dataset_dir="/mnt/data/users/lifeng/${TASK}_${dataset_version}_round_${ROUND}"
    local dataset_tag="${TASK}_${dataset_version}_round_${ROUND}"

    if [ ${ROUND} -gt 1 ]; then
        local last_raw_dataset_dir="/mnt/data/users/lifeng/${TASK}_${dataset_version}_round_$((${ROUND}-1))"
        echo "Create soft links from ${last_raw_dataset_dir} to ${raw_dataset_dir}..."
        while IFS= read -r -d '' data_file; do
            ln -sf "$(realpath "${data_file}")" "${raw_dataset_dir}/${data_file##*/}"
        done < <(find "${last_raw_dataset_dir}" \( -type f -o -type l \) -name "*.pkl" -print0)
    fi

    echo "Post-processing dataset tag: ${dataset_tag}"
    python post_process_data.py \
        --tag "${dataset_tag}" \
        --action_type "${ACTION_TYPE}" \
        --skip_static_frames \
        --resume
}

find_checkpoint_in_output_dir() {
    local output_dir="$1"
    local latest_ckpt="${output_dir}/checkpoints/latest.ckpt"
    local fallback_ckpt

    if [ -f "${latest_ckpt}" ]; then
        echo "${latest_ckpt}"
        return 0
    fi

    if [ -d "${output_dir}/checkpoints" ]; then
        fallback_ckpt=$(find "${output_dir}/checkpoints" -maxdepth 1 -name "*.ckpt" -type f | sort | tail -n 1)
        if [ -n "${fallback_ckpt}" ]; then
            echo "${fallback_ckpt}"
            return 0
        fi
    fi

    return 1
}

# Stage 0: Post-process data
echo "Stage 0: post-processing data..."
post_process_dataset "${STAGE1_DATASET_VERSION}"
if [ "${STAGE2_DATASET_VERSION}" != "${STAGE1_DATASET_VERSION}" ]; then
    post_process_dataset "${STAGE2_DATASET_VERSION}"
fi


# Stage 1: Train Policy
echo "Stage 1: training Policy..."
STAGE1_DATASET_VERSION_TAG=${STAGE1_DATASET_VERSION//./_}
STAGE2_DATASET_VERSION_TAG=${STAGE2_DATASET_VERSION//./_}


if [ "${POLICY_TYPE}" = "act_confidence_dual_encoder_stagewise" ]; then
    echo "Using ACT + Confidence Dual Encoder Stagewise Policy..."
    echo "Stage 1: Train ACT with ${STAGE1_DATASET_VERSION} dataset (${STAGE1_EPOCHS} epochs)"
    echo "Stage 2: Train Confidence with ${STAGE2_DATASET_VERSION} dataset (${STAGE2_EPOCHS} epochs)"
    CONFIG_NAME="train_act_confidence_dual_encoder_stagewise_workspace"
    POLICY_TASK_NAME=${TASK}_${STAGE1_DATASET_VERSION_TAG}_to_${STAGE2_DATASET_VERSION_TAG}_act_confidence_dual_encoder_stagewise_policy_round_${ROUND}
elif [ "${POLICY_TYPE}" = "act_confidence_stagewise" ]; then
    echo "Using ACT + Confidence Stagewise Policy (Single Encoder)..."
    echo "Stage 1: Train ACT with ${STAGE1_DATASET_VERSION} dataset (${STAGE1_EPOCHS} epochs)"
    echo "Stage 2: Train Confidence with ${STAGE2_DATASET_VERSION} dataset (${STAGE2_EPOCHS} epochs)"
    CONFIG_NAME="train_act_confidence_stagewise_workspace"
    POLICY_TASK_NAME=${TASK}_${STAGE1_DATASET_VERSION_TAG}_to_${STAGE2_DATASET_VERSION_TAG}_act_confidence_stagewise_policy_round_${ROUND}
elif [ "${POLICY_TYPE}" = "hybrid_stagewise" ]; then
    echo "Using Hybrid Diffusion + ACT Stagewise Policy (Three-Stage Training)..."
    echo "Stage 1: Train ACT + encoder + rnn (${STAGE1_EPOCHS} epochs, freeze DP + Confidence)"
    echo "Stage 2: Train ACT + DP + encoder + rnn (${STAGE2_EPOCHS} epochs, freeze Confidence)"
    echo "Stage 3: Train Confidence only (${STAGE3_EPOCHS} epochs)"
    CONFIG_NAME="train_hybrid_diffusion_act_stagewise_workspace"
    POLICY_TASK_NAME=${TASK}_${STAGE1_DATASET_VERSION_TAG}_hybrid_stagewise_round_${ROUND}
else
    echo "Error: Unsupported POLICY_TYPE '${POLICY_TYPE}'. Use one of: act_confidence_stagewise | act_confidence_dual_encoder_stagewise | hybrid_stagewise"
    exit 1
fi

COMMON_TRAIN_ARGS=(
    --config-name=${CONFIG_NAME}
    task=${TASK_CONFIG}
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY}
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY}
    task.dataset.resume_params_path=${DATASET_RESUME_PARAMS_PATH}
    task.dataset.use_noisy_episodes=True
    task.dataset.only_use_success_episodes=True
    training.use_amp=${USE_AMP_FOR_POLICY}
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY}
    training.debug=${TRAINING_DEBUG}
    optimizer.lr=${POLICY_LR}
    logging.mode=${LOGGING_MODE}
)

if [ "${POLICY_TYPE}" = "hybrid_stagewise" ]; then
    COMMON_TRAIN_ARGS+=(policy.noise_scheduler.prediction_type=v_prediction)
fi


# Update accelerate config for multi-GPU training
echo "Setting up accelerate config for ${NUM_GPUS} GPU(s)..."
cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: ${NUM_GPUS}
num_machines: 1
machine_rank: 0
main_training_function: main
mixed_precision: "no"
deepspeed_config: {}
fsdp_config: {}
megatron_lm_config: {}
downcast_bf16: "no"
gpu_ids: all
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
dynamo_config: {}
EOF

# Set CUDA_VISIBLE_DEVICES based on NUM_GPUS
if [ ${NUM_GPUS} -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=0
elif [ ${NUM_GPUS} -eq 2 ]; then
    CUDA_VISIBLE_DEVICES=0,1
elif [ ${NUM_GPUS} -eq 4 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3
elif [ ${NUM_GPUS} -eq 8 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    echo "Error: Unsupported NUM_GPUS '${NUM_GPUS}'. Use one of: 1, 2, 4, 8"
    exit 1
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [ "${POLICY_TYPE}" = "hybrid_stagewise" ]; then
    STAGE1_TASK_NAME=${POLICY_TASK_NAME}_stage1
    STAGE2_TASK_NAME=${POLICY_TASK_NAME}_stage2
    STAGE3_TASK_NAME=${POLICY_TASK_NAME}_stage3
    RUN_DATE=$(date +%Y.%m.%d)
    RUN_TIME=$(date +%H.%M-%S)
    STAGE1_OUTPUT_DIR="${SEARCH_PATH}/${RUN_DATE}/${RUN_TIME}_${STAGE1_TASK_NAME}"
    STAGE2_OUTPUT_DIR="${SEARCH_PATH}/${RUN_DATE}/${RUN_TIME}_${STAGE2_TASK_NAME}"
    STAGE3_OUTPUT_DIR="${SEARCH_PATH}/${RUN_DATE}/${RUN_TIME}_${STAGE3_TASK_NAME}"

    # ---------------------------- Stage 1: ACT only (freeze DP + Confidence) ----------------------------
    echo ""
    echo "=========================================="
    echo "Stage 1 training starts (dataset: ${STAGE1_DATASET_PATH})..."
    echo "Training: ACT + encoder + rnn (freeze DP + Confidence)"
    echo "Epochs: ${STAGE1_EPOCHS}"
    echo "Stage 1 output dir: ${STAGE1_OUTPUT_DIR}"
    echo "=========================================="
    STAGE1_TRAIN_ARGS=(
        "${COMMON_TRAIN_ARGS[@]}"
        task.dataset_path=${STAGE1_DATASET_PATH}
        task.name=${STAGE1_TASK_NAME}
        training.num_epochs=${STAGE1_EPOCHS}
        training.pretrained_ckpt_path=${POLICY_PRETRAINED_CKPT_PATH}
        training.stage1_epochs=${STAGE1_EPOCHS}
        training.stage2_epochs=0
        training.stage3_epochs=0
        policy.training_stage=1
        hydra.run.dir=${STAGE1_OUTPUT_DIR}
    )
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch train.py "${STAGE1_TRAIN_ARGS[@]}"

    echo ""
    echo "Resolving Stage 1 checkpoint from output dir: ${STAGE1_OUTPUT_DIR}"
    STAGE1_CKPT_PATH=$(find_checkpoint_in_output_dir "${STAGE1_OUTPUT_DIR}")
    if [ -z "${STAGE1_CKPT_PATH}" ]; then
        echo "Error: Could not find Stage 1 checkpoint under ${STAGE1_OUTPUT_DIR}/checkpoints"
        exit 1
    fi
    echo "Found Stage 1 checkpoint: ${STAGE1_CKPT_PATH}"

    # ---------------------------- Stage 2: ACT + DP (freeze Confidence) ----------------------------
    echo ""
    echo "=========================================="
    echo "Stage 2 training starts (dataset: ${STAGE1_DATASET_PATH})..."
    echo "Training: ACT + DP + encoder + rnn (freeze Confidence)"
    echo "Epochs: ${STAGE2_EPOCHS}"
    echo "Stage 2 output dir: ${STAGE2_OUTPUT_DIR}"
    echo "=========================================="
    STAGE2_TRAIN_ARGS=(
        "${COMMON_TRAIN_ARGS[@]}"
        task.dataset_path=${STAGE1_DATASET_PATH}
        task.name=${STAGE2_TASK_NAME}
        training.num_epochs=${STAGE2_EPOCHS}
        training.pretrained_ckpt_path=${STAGE1_CKPT_PATH}
        training.stage1_epochs=0
        training.stage2_epochs=${STAGE2_EPOCHS}
        training.stage3_epochs=0
        policy.training_stage=2
        hydra.run.dir=${STAGE2_OUTPUT_DIR}
    )
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch train.py "${STAGE2_TRAIN_ARGS[@]}"

    echo ""
    echo "Resolving Stage 2 checkpoint from output dir: ${STAGE2_OUTPUT_DIR}"
    STAGE2_CKPT_PATH=$(find_checkpoint_in_output_dir "${STAGE2_OUTPUT_DIR}")
    if [ -z "${STAGE2_CKPT_PATH}" ]; then
        echo "Error: Could not find Stage 2 checkpoint under ${STAGE2_OUTPUT_DIR}/checkpoints"
        exit 1
    fi
    echo "Found Stage 2 checkpoint: ${STAGE2_CKPT_PATH}"

    # ---------------------------- Stage 3: Confidence only (freeze all others) ----------------------------
    echo ""
    echo "=========================================="
    echo "Stage 3 training starts (dataset: ${STAGE2_DATASET_PATH})..."
    echo "Training: Confidence only (freeze all others)"
    echo "Epochs: ${STAGE3_EPOCHS}"
    echo "Stage 3 output dir: ${STAGE3_OUTPUT_DIR}"
    echo "=========================================="
    STAGE3_TRAIN_ARGS=(
        "${COMMON_TRAIN_ARGS[@]}"
        task.dataset_path=${STAGE2_DATASET_PATH}
        task.name=${STAGE3_TASK_NAME}
        training.num_epochs=${STAGE3_EPOCHS}
        training.pretrained_ckpt_path=${STAGE2_CKPT_PATH}
        training.stage1_epochs=0
        training.stage2_epochs=0
        training.stage3_epochs=${STAGE3_EPOCHS}
        policy.training_stage=3
        hydra.run.dir=${STAGE3_OUTPUT_DIR}
    )
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch train.py "${STAGE3_TRAIN_ARGS[@]}"

    echo ""
    echo "=========================================="
    echo "All three stages completed!"
    echo "Stage 1: ${STAGE1_CKPT_PATH}"
    echo "Stage 2: ${STAGE2_CKPT_PATH}"
    echo "Stage 3: $(find_checkpoint_in_output_dir "${STAGE3_OUTPUT_DIR}")"
    echo "=========================================="

elif [ "${POLICY_TYPE}" = "act_confidence_dual_encoder_stagewise" ] || [ "${POLICY_TYPE}" = "act_confidence_stagewise" ]; then
    STAGE1_TASK_NAME=${POLICY_TASK_NAME}_stage1
    STAGE2_TASK_NAME=${POLICY_TASK_NAME}_stage2
    RUN_DATE=$(date +%Y.%m.%d)
    RUN_TIME=$(date +%H.%M-%S)
    STAGE1_OUTPUT_DIR="${SEARCH_PATH}/${RUN_DATE}/${RUN_TIME}_${STAGE1_TASK_NAME}"
    STAGE2_OUTPUT_DIR="${SEARCH_PATH}/${RUN_DATE}/${RUN_TIME}_${STAGE2_TASK_NAME}"

    echo ""
    echo "=========================================="
    echo "Stage 1 training starts (dataset: ${STAGE1_DATASET_PATH})..."
    echo "Stage 1 output dir: ${STAGE1_OUTPUT_DIR}"
    echo "=========================================="
    STAGE1_TRAIN_ARGS=(
        "${COMMON_TRAIN_ARGS[@]}"
        task.dataset_path=${STAGE1_DATASET_PATH}
        task.name=${STAGE1_TASK_NAME}
        training.num_epochs=${STAGE1_EPOCHS}
        training.pretrained_ckpt_path=${POLICY_PRETRAINED_CKPT_PATH}
        training.stage1_epochs=${STAGE1_EPOCHS}
        training.stage2_epochs=0
        policy.training_stage=1
        hydra.run.dir=${STAGE1_OUTPUT_DIR}
    )
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch train.py "${STAGE1_TRAIN_ARGS[@]}"

    echo ""
    echo "Resolving Stage 1 checkpoint from output dir: ${STAGE1_OUTPUT_DIR}"
    STAGE1_CKPT_PATH=$(find_checkpoint_in_output_dir "${STAGE1_OUTPUT_DIR}")
    if [ -z "${STAGE1_CKPT_PATH}" ]; then
        echo "Error: Could not find Stage 1 checkpoint under ${STAGE1_OUTPUT_DIR}/checkpoints"
        exit 1
    fi
    echo "Found Stage 1 checkpoint: ${STAGE1_CKPT_PATH}"

    echo ""
    echo "Stage 2 training starts (dataset: ${STAGE2_DATASET_PATH})..."
    echo "Stage 2 output dir: ${STAGE2_OUTPUT_DIR}"
    STAGE2_TRAIN_ARGS=(
        "${COMMON_TRAIN_ARGS[@]}"
        task.dataset_path=${STAGE2_DATASET_PATH}
        task.name=${STAGE2_TASK_NAME}
        training.num_epochs=${STAGE2_EPOCHS}
        training.pretrained_ckpt_path=${STAGE1_CKPT_PATH}
        training.stage1_epochs=0
        training.stage2_epochs=${STAGE2_EPOCHS}
        policy.training_stage=2
        hydra.run.dir=${STAGE2_OUTPUT_DIR}
    )
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch train.py "${STAGE2_TRAIN_ARGS[@]}"

    echo ""
    echo "All stages completed!"
else
    echo ""
    echo "Single-stage training starts (dataset: ${STAGE1_DATASET_PATH})..."
    SINGLE_TRAIN_ARGS=(
        "${COMMON_TRAIN_ARGS[@]}"
        task.dataset_path=${STAGE1_DATASET_PATH}
        task.name=${POLICY_TASK_NAME}
        training.num_epochs=${POLICY_NUM_EPOCHS}
        training.pretrained_ckpt_path=${POLICY_PRETRAINED_CKPT_PATH}
    )
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch train.py "${SINGLE_TRAIN_ARGS[@]}"
fi

exit 0
