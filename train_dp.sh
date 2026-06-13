#!/bin/bash

# Stage 0: Post-process data
echo "Stage 0: post-processing data..."
#python post_process_data.py --tag flip_v2 --action_type right_arm_6DOF --skip_static_frames

# Stage 1: Train Policy
echo "Stage 1: training Policy..."
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_dual_box_image_dp_10fps \
    task.dataset_path=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/right/dual_box_v2_round_1_zarr \
    task.name=real_dual_box_image_dp_10fps_v2 \
    logging.mode=online \
    logging.backend=tensorboard