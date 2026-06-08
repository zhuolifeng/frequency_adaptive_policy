#!/bin/bash

# # ImplicitRDP 
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_dual_switch_image_wrench_offline_10fps  \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.06/23.53.08_train_reactive_diffusion_transformer_image_dual_switch_v7_policy_bigger_acp_round_1/checkpoints/latest.ckpt

# # DPMultiFreqCD50HzConfidencePolicy stage 1
python eval_real_robot_flexiv.py \
   --config-name train_dp_multifreq_cd_50hz_confidence_workspace \
   task=real_dual_switch_image_wrench_offline_10fps_50hz \
   prediction_type=sample \
   loss_type=velocity \
   task.env_runner.action_type=right_arm_6DOF \
   +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
   policy.training_stage=3 \
   'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.06/12.38.15_train_dp_multifreq_cd_50hz_confidence_workspace_dual_switch_v7_dp_multifreq_cd_50hz_confidence/checkpoints/epoch=1100-stage3-train_loss=0.014.ckpt"'
#   'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.06/12.40.48_train_dp_multifreq_cd_50hz_confidence_workspace_dual_switch_v7_dp_multifreq_cd_50hz_confidence/checkpoints/epoch=1120-stage3-train_loss=0.013.ckpt"'

# # DPMultiFreqCD50HzConfidencePolicy stage 3
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_50hz_confidence_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_acp \
#    prediction_type=sample \
#    loss_type=eps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=1 \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.06/12.40.48_train_dp_multifreq_cd_50hz_confidence_workspace_dual_switch_v7_dp_multifreq_cd_50hz_confidence/checkpoints/stage1_final.ckpt

# # FMMultiFreqRDTActionChunkPolicy stage 1
# python eval_real_robot_flexiv.py \
#    --config-name train_fm_multifreq_rdt_action_chunk_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_acp \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=2 \
#    ckpt_path=/home/lifeng/Desktop/frequency_adaptive_policy_flow/2026.06.06/18.26.17_train_fm_multifreq_rdt_action_chunk_workspace_dual_switch_v7_fm_multifreq_rdt_action_chunk/checkpoints/latest.ckpt

# # FMMultiFreqRDTActionChunkPolicy stage 2
# python eval_real_robot_flexiv.py \
#    --config-name train_fm_multifreq_rdt_action_chunk_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_acp \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=2 \
#    ckpt_path=null

# ACT50HzPolicy
# python eval_real_robot_flexiv.py \
#    --config-name train_act_50hz_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_50hz \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    ckpt_path=/home/lifeng/Desktop/frequency_adaptive_policy_flow/2026.06.06/19.22.12_train_act_50hz_workspace_dual_switch_v7_act_50hz/checkpoints/latest.ckpt