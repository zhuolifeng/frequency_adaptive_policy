#!/bin/bash

# # ImplicitRDP dual_switch
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_dual_switch_image_wrench_offline_10fps  \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.08/14.59.00_train_reactive_diffusion_transformer_image_dual_switch_v8_policy_bigger_acp_round_1/checkpoints/latest.ckpt

# # DP dual_switch
# python eval_real_robot_flexiv.py \
#    --config-name train_diffusion_unet_real_image_workspace \
#    task=real_dual_switch_image_dp_10fps \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30 \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.19/13.18.44_train_diffusion_unet_image_real_dual_switch_image_dp_10fps_v2/checkpoints/latest.ckpt

# # RDP dual_switch
# python eval_real_robot_flexiv.py \
#      --config-name train_latent_diffusion_unet_real_image_workspace \
#      task=real_dual_switch_image_wrench_ldp_10fps \
#      at=at_dual_switch \
#      at_load_dir=/home/lifeng/frequency_adaptive_policy/2026.06.19/13.13.56_train_vae_real_dual_switch_image_wrench_at_10fps_0619131345/checkpoints/latest.ckpt \
#      +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30 \
#      ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.19/13.58.24_train_latent_diffusion_unet_image_real_dual_switch_image_wrench_ldp_10fps_0619131345/checkpoints/latest.ckpt

# # DPMultiFreqCD30HzConfidencePolicy stage 1 eps 10hz dual_switch
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

# # DPMultiFreqCD30HzConfidencePolicy stage 1 velocity 10hz dual_switch
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_acp \
#    prediction_type=sample \
#    loss_type=velocity \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=1 \
#    policy.inference_head='10hz' \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.08/15.04.07_train_dp_multifreq_cd_30hz_confidence_workspace_dual_switch_v8_dp_multifreq_cd_30hz_confidence/checkpoints/stage1_final.ckpt


# # DPMultiFreqCD30HzConfidencePolicy stage 3 30hz distilled eps dual_switch
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_30hz \
#    prediction_type=sample \
#    loss_type=eps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=3 \
#    policy.inference_head='30hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.19/19.47.09_train_dp_multifreq_cd_30hz_confidence_workspace_dual_switch_v8_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1770-stage3-train_loss=0.002.ckpt"'

# # DPMultiFreqCD30HzConfidencePolicy stage 3 30hz distilled velocity dual_switch
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_30hz \
#    prediction_type=sample \
#    loss_type=velocity \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=3 \
#    policy.inference_head='30hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.19/12.57.20_train_dp_multifreq_cd_30hz_confidence_workspace_dual_switch_v8_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1640-stage3-train_loss=0.001.ckpt"'

# # DPMultiFreqCD30HzConfidencePolicy stage 3 confidence eps dual_switch
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_dual_hz \
#    prediction_type=sample \
#    loss_type=eps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=3 \
#    policy.inference_head='10hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.08/02.02.25_train_dp_multifreq_cd_30hz_confidence_workspace_dual_switch_v8_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1130-stage3-train_loss=0.025.ckpt"'

# # DPMultiFreqCD30HzConfidencePolicy stage 3 confidence velocity dual_switch
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_dual_hz \
#    prediction_type=sample \
#    loss_type=velocity \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=3 \
#    policy.inference_head='10hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.08/15.04.07_train_dp_multifreq_cd_30hz_confidence_workspace_dual_switch_v8_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1130-stage3-train_loss=0.023.ckpt"'


# ACT30HzPolicy dual_switch
# python eval_real_robot_flexiv.py \
#    --config-name train_act_30hz_workspace \
#    task=real_dual_switch_image_wrench_offline_10fps_30hz \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.08/14.58.55_train_act_30hz_workspace_dual_switch_v8_act_30hz/checkpoints/latest.ckpt

####################################dual_button


# # DP dual_button
# python eval_real_robot_flexiv.py \
#    --config-name train_diffusion_unet_real_image_workspace \
#    task=real_dual_button_image_dp_10fps \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30 \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.19/13.17.25_train_diffusion_unet_image_real_dual_button_image_dp_10fps_v2/checkpoints/latest.ckpt

# # RDP dual_button
# python eval_real_robot_flexiv.py \
#      --config-name train_latent_diffusion_unet_real_image_workspace \
#      task=real_dual_button_image_wrench_ldp_10fps \
#      at=at_dual_button \
#      at_load_dir=/home/lifeng/frequency_adaptive_policy/2026.06.19/13.04.38_train_vae_real_dual_button_image_wrench_at_10fps_0619130432/checkpoints/latest.ckpt \
#      +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30 \
#      ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.19/13.50.08_train_latent_diffusion_unet_image_real_dual_button_image_wrench_ldp_10fps_0619130432/checkpoints/latest.ckpt


# # ImplicitRDP dual_button
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_dual_button_image_wrench_offline_10fps  \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.11/08.45.05_train_reactive_diffusion_transformer_image_dual_button_v1_policy_bigger_acp_round_1/checkpoints/latest.ckpt


# ACT30HzPolicy dual_button
# python eval_real_robot_flexiv.py \
#    --config-name train_act_30hz_workspace \
#    task=real_dual_button_image_wrench_offline_10fps_30hz \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.20/00.14.27_train_act_30hz_workspace_dual_button_v2_act_30hz/checkpoints/latest.ckpt

# # DPMultiFreqCD30HzConfidencePolicy stage 1 eps 10hz dual_button
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_button_image_wrench_offline_10fps_acp \
#    prediction_type=sample \
#    loss_type=eps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=1 \
#    policy.inference_head='10hz' \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.20/16.17.22_train_dp_multifreq_cd_30hz_confidence_workspace_dual_button_v2_dp_multifreq_cd_30hz_confidence/checkpoints/stage1_final.ckpt

# # DPMultiFreqCD30HzConfidencePolicy stage 1 velocity 10hz dual_button
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_button_image_wrench_offline_10fps_acp \
#    prediction_type=sample \
#    loss_type=velocity \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=1 \
#    policy.inference_head='10hz' \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.11/09.23.21_train_dp_multifreq_cd_30hz_confidence_workspace_dual_button_v1_dp_multifreq_cd_30hz_confidence/checkpoints/stage1_final.ckpt


# # DPMultiFreqCD30HzConfidencePolicy stage 3 30hz distilled eps dual_button
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_button_image_wrench_offline_10fps_30hz \
#    prediction_type=sample \
#    loss_type=eps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=3 \
#    policy.inference_head='30hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.19/11.23.38_train_dp_multifreq_cd_30hz_confidence_workspace_dual_button_v1_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1680-stage3-train_loss=0.002.ckpt"'

# # DPMultiFreqCD30HzConfidencePolicy stage 3 30hz distilled velocity dual_button
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_button_image_wrench_offline_10fps_30hz \
#    prediction_type=sample \
#    loss_type=velocity \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=3 \
#    policy.inference_head='30hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.11/09.23.21_train_dp_multifreq_cd_30hz_confidence_workspace_dual_button_v1_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1180-stage3-train_loss=0.024.ckpt"'

# # DPMultiFreqCD30HzConfidencePolicy stage 3 confidence eps dual_button
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_button_image_wrench_offline_10fps_dual_hz \
#    prediction_type=sample \
#    loss_type=eps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=3 \
#    policy.inference_head='10hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.11/08.45.45_train_dp_multifreq_cd_30hz_confidence_workspace_dual_button_v1_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1180-stage3-train_loss=0.023.ckpt"'

# # DPMultiFreqCD30HzConfidencePolicy stage 3 confidence velocity dual_button
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_button_image_wrench_offline_10fps_dual_hz \
#    prediction_type=sample \
#    loss_type=velocity \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=3 \
#    policy.inference_head='10hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.11/09.23.21_train_dp_multifreq_cd_30hz_confidence_workspace_dual_button_v1_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1180-stage3-train_loss=0.024.ckpt"'



####################################dual_box


# # ImplicitRDP dual_box
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_dual_box_image_wrench_offline_10fps  \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.12/13.43.24_train_reactive_diffusion_transformer_image_dual_box_v1_policy_bigger_acp_round_1/checkpoints/latest.ckpt

# ACT30HzPolicy dual_box
# python eval_real_robot_flexiv.py \
#    --config-name train_act_30hz_workspace \
#    task=real_dual_box_image_wrench_offline_10fps_30hz \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.16/13.19.30_train_act_30hz_workspace_dual_box_v5_act_30hz/checkpoints/latest.ckpt

# # DPMultiFreqCD30HzConfidencePolicy stage 1 eps 10hz dual_box
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_box_image_wrench_offline_10fps_acp \
#    prediction_type=sample \
#    loss_type=eps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=1 \
#    policy.inference_head='10hz' \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.19/03.18.40_train_dp_multifreq_cd_30hz_confidence_workspace_dual_box_v5_dp_multifreq_cd_30hz_confidence/checkpoints/stage1_final.ckpt

# DPMultiFreqCD30HzConfidencePolicy stage 1 velocity 10hz dual_box
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_box_image_wrench_offline_10fps_acp \
#    prediction_type=sample \
#    loss_type=velocity \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30 \
#    policy.training_stage=1 \
#    policy.inference_head='10hz' \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.19/03.20.08_train_dp_multifreq_cd_30hz_confidence_workspace_dual_box_v5_dp_multifreq_cd_30hz_confidence/checkpoints/stage1_final.ckpt

# # DPMultiFreqCD30HzConfidencePolicy stage 3 30hz distilled eps dual_box
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_box_image_wrench_offline_10fps_30hz \
#    prediction_type=sample \
#    loss_type=eps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera \
#    policy.training_stage=3 \
#    policy.inference_head='30hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.18/16.12.41_train_dp_multifreq_cd_30hz_confidence_workspace_dual_box_v5_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1270-stage3-train_loss=0.002.ckpt"'

#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.18/16.12.41_train_dp_multifreq_cd_30hz_confidence_workspace_dual_box_v5_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1270-stage3-train_loss=0.002.ckpt"'

# # DPMultiFreqCD30HzConfidencePolicy stage 3 30hz distilled velocity dual_box
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_box_image_wrench_offline_10fps_30hz \
#    prediction_type=sample \
#    loss_type=velocity \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30 \
#    policy.training_stage=3 \
#    policy.inference_head='30hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.19/03.20.08_train_dp_multifreq_cd_30hz_confidence_workspace_dual_box_v5_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1450-stage3-train_loss=0.001.ckpt"'

# # # DPMultiFreqCD30HzConfidencePolicy stage 3 confidence eps dual_box
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_box_image_wrench_offline_10fps_dual_hz \
#    prediction_type=sample \
#    loss_type=eps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=3 \
#    policy.inference_head='10hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.19/03.18.40_train_dp_multifreq_cd_30hz_confidence_workspace_dual_box_v5_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1510-stage3-train_loss=0.001.ckpt"'

# # DPMultiFreqCD30HzConfidencePolicy stage 3 confidence velocity dual_box
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_workspace \
#    task=real_dual_box_image_wrench_offline_10fps_dual_hz \
#    prediction_type=sample \
#    loss_type=velocity \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30 \
#    policy.training_stage=3 \
#    policy.inference_head='10hz' \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.19/03.20.08_train_dp_multifreq_cd_30hz_confidence_workspace_dual_box_v5_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1450-stage3-train_loss=0.001.ckpt"'

# # DP dual_box
# python eval_real_robot_flexiv.py \
#    --config-name train_diffusion_unet_real_image_workspace \
#    task=real_dual_box_image_dp_10fps \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30 \
#    ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.17/00.17.57_train_diffusion_unet_image_real_dual_box_image_dp_10fps_v2/checkpoints/latest.ckpt

# # RDP dual_box
# python eval_real_robot_flexiv.py \
#      --config-name train_latent_diffusion_unet_real_image_workspace \
#      task=real_dual_box_image_wrench_ldp_10fps \
#      at=at_dual_box \
#      at_load_dir=/home/lifeng/frequency_adaptive_policy/2026.06.17/16.33.42_train_vae_real_dual_box_image_wrench_at_10fps_0617163338/checkpoints/latest.ckpt \
#      +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30 \
#      ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.17/17.34.46_train_latent_diffusion_unet_image_real_dual_box_image_wrench_ldp_10fps_0617163338/checkpoints/latest.ckpt

# # DPMultiFreqCD30HzConfidenceDualPolicy stage 3 confidence eps dual_box
# python eval_real_robot_flexiv.py \
#    --config-name train_dp_multifreq_cd_30hz_confidence_dual_workspace \
#    task=real_dual_box_image_wrench_offline_10fps_dual_hz \
#    prediction_type=sample \
#    loss_type=eps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=./data/checkpoints/real_dual_switch_iphone_usb_camera_30_10fps \
#    policy.training_stage=3 \
#    policy.inference_head=30hz \
#    policy.use_confidence_based_head_selection=True \
#    policy.confidence_threshold_for_head_selection=3.5 \
#    policy.stage1_ema_ckpt_path=/home/lifeng/frequency_adaptive_policy/2026.06.19/03.18.40_train_dp_multifreq_cd_30hz_confidence_workspace_dual_box_v5_dp_multifreq_cd_30hz_confidence/checkpoints/stage1_final.ckpt \
#    'ckpt_path="/home/lifeng/frequency_adaptive_policy/2026.06.19/03.18.40_train_dp_multifreq_cd_30hz_confidence_workspace_dual_box_v5_dp_multifreq_cd_30hz_confidence/checkpoints/epoch=1510-stage3-train_loss=0.001.ckpt"'

