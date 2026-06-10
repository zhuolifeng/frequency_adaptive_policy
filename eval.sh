#!/bin/bash

# # ImplicitRDP 
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_dual_switch_image_wrench_offline_10fps  \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.08/14.59.00_train_reactive_diffusion_transformer_image_dual_switch_v8_policy_bigger_acp_round_1/checkpoints/latest.ckpt

# # DPMultiFreqCD30HzConfidencePolicy stage 1
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_acp \
#    prediction_type=sample \
#    loss_type=eps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=1 \
#    policy.inference_head='10hz' \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.08/02.02.25_train_dp_multifreq_cd_30hz_confidence_workspace_dual_switch_v8_dp_multifreq_cd_30hz_confidence/checkpoints/stage1_final.ckpt


   #    /home/lifeng/frequency_adaptive_policy/2026.06.08/15.04.07_train_dp_multifreq_cd_30hz_confidence_workspace_dual_switch_v8_dp_multifreq_cd_30hz_confidence/checkpoints/stage1_final.ckpt    velocity
   #    /home/lifeng/frequency_adaptive_policy/2026.06.08/02.02.25_train_dp_multifreq_cd_30hz_confidence_workspace_dual_switch_v8_dp_multifreq_cd_30hz_confidence/checkpoints/stage1_final.ckpt    eps


# # DPMultiFreqCD30HzConfidencePolicy stage 3
python eval_real_robot_flexiv.py \
   --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
   task=real_dual_switch_image_wrench_offline_10fps_30hz \
   prediction_type=sample \
   loss_type=eps\
   task.env_runner.action_type=right_arm_6DOF \
   +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
   policy.training_stage=3 \
   policy.inference_head='30hz' \
   'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.08/02.02.25_train_dp_multifreq_cd_30hz_confidence_workspace_dual_switch_v8_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1130-stage3-train_loss=0.025.ckpt"'

# 'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.08/15.04.07_train_dp_multifreq_cd_30hz_confidence_workspace_dual_switch_v8_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1130-stage3-train_loss=0.023.ckpt"'  velocity
# 'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.08/02.02.25_train_dp_multifreq_cd_30hz_confidence_workspace_dual_switch_v8_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1130-stage3-train_loss=0.025.ckpt"'  eps


# ACT30HzPolicy
# python eval_real_robot_flexiv.py \
#    --config-name train_act_30hz_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_30hz \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.08/14.58.55_train_act_30hz_workspace_dual_switch_v8_act_30hz/checkpoints/latest.ckpt