#!/bin/bash

#--------------------------------
# Config
GPU_ID=1

TASK="dual_switch"
DATASET_VERSION="v3"
ACTION_TYPE="right_arm_6DOF_virtual_target_stiffness"
#ACTION_TYPE="right_arm_6DOF"
TRAINING_DEBUG=False
USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
LOGGING_MODE="online"

# Policy : "diffusion"、"hybrid"、"act_confidence"、"act_confidence_dual_encoder"
POLICY_TYPE="act_confidence_dual_encoder"  # "hybrid": diffusion + ACT；"diffusion": only diffusion；"act_confidence": only ACT + confidence；"act_confidence_dual_encoder": ACT + confidence with dual encoders

ROUND=1
POLICY_PRETRAINED_CKPT_PATH="null"
REWARD_PRETRAINED_CKPT_PATH="null"
CRITIC_PRETRAINED_CKPT_PATH="null"
DATASET_RESUME_PARAMS_PATH="null"
#--------------------------------

# if round <= 1, use offline RL config
if [ ${ROUND} -le 1 ]; then
    echo "Training offline RL"
    RL_TYPE="offline"
    DATASET_PATH="/mnt/data/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr"
    TASK_CONFIG="real_${TASK}_image_wrench_offline_rl_10fps_acp"
    POLICY_NUM_EPOCHS=600
    REWARD_NUM_EPOCHS=200
    CRITIC_NUM_EPOCHS=200
    POLICY_LR=1.0e-4
    REWARD_LR=1.0e-4
    CRITIC_LR=1.0e-4
else
    echo "Training online RL"
    RL_TYPE="online"
    DATASET_PATH="/mnt/data/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}_zarr"
    TASK_CONFIG="real_${TASK}_image_wrench_online_rl_10fps_acp"
    POLICY_NUM_EPOCHS=200
    REWARD_NUM_EPOCHS=200
    CRITIC_NUM_EPOCHS=200
    POLICY_LR=1.0e-5
    REWARD_LR=1.0e-5
    CRITIC_LR=1.0e-5
fi

SEARCH_PATH="./data/outputs"

# Stage 0: Post-process data
echo "Stage 0: post-processing data..."
if [ ${ROUND} -gt 1 ]; then
    echo "Create soft link of every data in the last dataset directory to the new dataset directory..."
    # find the last dataset directory
    LAST_RAW_DATASET_DIR="/mnt/data/lifeng/${TASK}_${DATASET_VERSION}_round_$((${ROUND}-1))"
    RAW_DATASET_DIR="/mnt/data/lifeng/${TASK}_${DATASET_VERSION}_round_${ROUND}"
    # create soft link of every data in the last dataset directory to the new dataset directory
    for data_file in $(find ${LAST_RAW_DATASET_DIR} \( -type f -o -type l \) -name "*.pkl"); do
        ln -sf $(realpath ${data_file}) ${RAW_DATASET_DIR}/${data_file##*/}
    done
fi
echo "Post-processing data..."
python post_process_data.py --tag ${TASK}_${DATASET_VERSION}_round_${ROUND} --action_type ${ACTION_TYPE} --skip_static_frames


# Stage 1: Train Policy
echo "Stage 1: training Policy..."
POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_policy_round_${ROUND}


if [ "${POLICY_TYPE}" = "hybrid" ]; then
    echo "Using Hybrid Diffusion + ACT Policy..."
    CONFIG_NAME="train_hybrid_diffusion_act_workspace"
    POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_hybrid_policy_round_${ROUND}
elif [ "${POLICY_TYPE}" = "diffusion" ]; then
    echo "Using Diffusion Policy..."
    CONFIG_NAME="train_diffusion_reactive_transformer_rnn_nsca_real_image_original_dataset_workspace"
elif [ "${POLICY_TYPE}" = "act_confidence" ]; then
    echo "Using ACT + Confidence Policy..."
    CONFIG_NAME="train_act_confidence_workspace"
    POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_act_confidence_policy_round_${ROUND}
elif [ "${POLICY_TYPE}" = "act_confidence_dual_encoder" ]; then
    echo "Using ACT + Confidence Dual Encoder Policy..."
    CONFIG_NAME="train_act_confidence_dual_encoder_workspace"
    POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_act_confidence_dual_encoder_policy_round_${ROUND}
else
    echo "Error: Unsupported POLICY_TYPE '${POLICY_TYPE}'. Use one of: diffusion | hybrid | act_confidence | act_confidence_dual_encoder"
    exit 1
fi

COMMON_TRAIN_ARGS=(
    --config-name=${CONFIG_NAME}
    task=${TASK_CONFIG}
    task.dataset_path=${DATASET_PATH}
    task.name=${POLICY_TASK_NAME}
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY}
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY}
    ++policy.obs_drop_type=gaussian
    ++policy.gaussian_start=20
    ++policy.obs_drop_keys=['right_wrist_img']
    task.dataset.resume_params_path=${DATASET_RESUME_PARAMS_PATH}
    task.dataset.use_noisy_episodes=True
    task.dataset.only_use_success_episodes=True
    training.num_epochs=${POLICY_NUM_EPOCHS}
    training.use_amp=${USE_AMP_FOR_POLICY}
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY}
    training.pretrained_ckpt_path=${POLICY_PRETRAINED_CKPT_PATH}
    training.debug=${TRAINING_DEBUG}
    optimizer.lr=${POLICY_LR}
    logging.mode=${LOGGING_MODE}
)

if [ "${POLICY_TYPE}" = "diffusion" ] || [ "${POLICY_TYPE}" = "hybrid" ]; then
    COMMON_TRAIN_ARGS+=(policy.noise_scheduler.prediction_type=v_prediction)
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py "${COMMON_TRAIN_ARGS[@]}"

exit 0

# find the latest policy checkpoint
echo ""
echo "Searching for the latest policy checkpoint..."
POLICY_CKPT_PATH=$(find "${SEARCH_PATH}" -maxdepth 2 -path "*${POLICY_TASK_NAME}*" -type d)/checkpoints/latest.ckpt

if [ ! -f "${POLICY_CKPT_PATH}" ]; then
    echo "Error: Policy checkpoint not found at ${POLICY_CKPT_PATH}"
    exit 1
fi
echo "Found policy checkpoint at ${POLICY_CKPT_PATH}"


echo "Updating the dataset resume params with the file in the policy checkpoint directory..."
DATASET_RESUME_PARAMS_PATH=$(find "${SEARCH_PATH}" -maxdepth 2 -path "*${POLICY_TASK_NAME}*" -type d)/dataset_resume_params.pkl

if [ ! -f "${DATASET_RESUME_PARAMS_PATH}" ]; then
    echo "Error: Dataset resume params not found at ${DATASET_RESUME_PARAMS_PATH}"
    exit 1
fi
echo "Using dataset resume params at ${DATASET_RESUME_PARAMS_PATH}"

# Stage 2: Train Reward Model
echo ""
echo "Stage 2: training Reward Model..."
REWARD_TASK_NAME=${TASK}_${DATASET_VERSION}_reward_round_${ROUND}
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_desired_force_reward_real_image_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${REWARD_TASK_NAME} \
    task.dataset.resume_params_path=${DATASET_RESUME_PARAMS_PATH} \
    task.dataset.only_use_human_data=True \
    training.num_epochs=${REWARD_NUM_EPOCHS} \
    training.pretrained_ckpt_path=${REWARD_PRETRAINED_CKPT_PATH} \
    training.debug=${TRAINING_DEBUG} \
    optimizer.lr=${REWARD_LR} \
    policy_ckpt_path=${POLICY_CKPT_PATH} \
    logging.mode=${LOGGING_MODE}

# find the latest reward checkpoint
echo ""
echo "Searching for the latest reward checkpoint..."
REWARD_CKPT_PATH=$(find "${SEARCH_PATH}" -maxdepth 2 -path "*${REWARD_TASK_NAME}*" -type d)/checkpoints/latest.ckpt

if [ ! -f "${REWARD_CKPT_PATH}" ]; then
    echo "Error: Reward checkpoint not found at ${REWARD_CKPT_PATH}"
    exit 1
fi
echo "Found reward checkpoint at ${REWARD_CKPT_PATH}"

# Stage 3: Train Critic Model
echo ""
echo "Stage 3: training Critic Model..."
CRITIC_TASK_NAME=${TASK}_${DATASET_VERSION}_critic_round_${ROUND}
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_diffusion_reactive_transformer_small_rnn_nsca_desired_force_reward_critic_real_image_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${CRITIC_TASK_NAME} \
    task.dataset.resume_params_path=${DATASET_RESUME_PARAMS_PATH} \
    task.dataset.random_append_obs=True \
    task.dataset.use_noisy_episodes=True \
    policy.noise_scheduler.prediction_type=v_prediction \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.num_epochs=${CRITIC_NUM_EPOCHS} \
    training.pretrained_ckpt_path=${CRITIC_PRETRAINED_CKPT_PATH} \
    training.debug=${TRAINING_DEBUG} \
    optimizer.lr=${CRITIC_LR} \
    reward_ckpt_path=${REWARD_CKPT_PATH} \
    policy_ckpt_path=${POLICY_CKPT_PATH} \
    logging.mode=${LOGGING_MODE}

echo "Finished training policy, reward, and critic for round ${ROUND}. Exiting..."