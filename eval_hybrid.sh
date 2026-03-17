#train_act_confidence_workspace
#train_hybrid_diffusion_act_workspace
#train_reactive_diffusion_transformer_real_image_workspace

# Hybrid Diffusion + ACT Policy (Flipping) round 1 (diffusion head)
python eval_real_robot_flexiv.py \
   --config-name train_hybrid_diffusion_act_workspace \
   task=real_dual_switch_image_wrench_offline_rl_10fps_acp \
   policy.noise_scheduler.prediction_type=v_prediction \
   task.env_runner.action_type=right_arm_6DOF \
   +task.env_runner.output_dir=data/outputs/2025.12.09/22.21.53_train_hybrid_diffusion_act_workspace_switch_v12_hybrid_policy_round_1 \
   task.env_runner._target_="ImplictRDP.env_runner.real_stable_reactive_runner.RealStableReactiveRunner" \
   ckpt_path=data/outputs/2025.12.19/21.32.27_train_dual_switch_hybrid_diffusion_act_workspace_dual_switch_v2_hybrid_policy_round_1/checkpoints/latest.ckpt \
   +task.env_runner.enable_rerun_visualization=False