# Frequency Adaptive Policy

This repository trains robot manipulation policies from demonstrations with camera images, end-effector poses and force/torque measurements. The `flow` branch includes a policy that uses confidence to select between 10 Hz and 30 Hz action predictions. It also includes fixed-rate 30 Hz iMeanFlow, consistency and ACT policies for comparison. The training code and robot interfaces build on ImplicitRDP. [Original project](https://implicit-rdp.github.io)

## Install

The reference environment uses Ubuntu 22.04, Python 3.10 and an NVIDIA GPU. Run the examples from the repository root. The following commands install the CUDA 12.1 build of PyTorch.

```bash
git clone --single-branch --branch flow https://github.com/zhuolifeng/frequency_adaptive_policy.git
cd frequency_adaptive_policy
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-training.txt
```

Inspect the resolved iMeanFlow configuration before supplying a dataset. This command prints the model, task and training settings and exits before training.

```bash
python train.py --config-name=train_dp_30hz_imeanflow_workspace \
    task=real_dual_switch_image_wrench_offline_10fps_30hz --cfg job --resolve
```

The CPU tests train and validate reduced iMeanFlow and consistency models on two synthetic episodes. They reload both the consistency teacher and student checkpoints and check the loss, action windows, image preprocessing, replay-buffer I/O and mocked robot connections. Run them from the repository root.

```bash
python -m pip install pytest
python -m pytest -q
```

## Prepare demonstrations

The training loader expects `DATASET_PATH` to name a **directory containing** `replay_buffer.zarr`. For example, the dual-switch task uses `data/external_img`, `data/right_wrist_img`, `data/right_robot_tcp_pose`, `data/right_robot_tcp_wrench`, `data/right_robot_tcp_wrench_stack`, `data/action`, `data/action_stack` and `meta/episode_ends` inside the store. The 30 Hz store groups three force samples and three action targets with each 10 Hz observation. Check the task's `shape_meta` when preparing another dataset. [Dual-switch task configuration](ImplicitRDP/config/task/real_dual_switch_image_wrench_offline_10fps_30hz.yaml)

For raw-data preprocessing, install the additional dependencies with `python -m pip install -r requirements.txt`. To convert numbered recording files such as `data/dual_switch_v1_round_1/0.pkl`, use the action representation expected by the task. This example writes `data/dual_switch_v1_round_1_zarr/replay_buffer.zarr`.

```bash
python post_process_data.py --data-root data --tag dual_switch_v1_round_1 \
    --action_type right_arm_6DOF_virtual_target_stiffness
```

Supply recordings or a prepared Zarr store for your experiment. The original ImplicitRDP release provides datasets and checkpoints for its own tasks. Match the observation keys, action representation and sampling rates before reusing them. [Upstream datasets](https://huggingface.co/datasets/WendiChen/ImplicitRDP_dataset) · [Upstream checkpoints](https://huggingface.co/WendiChen/ImplicitRDP_model)

## Train

For a single-GPU run on dual-switch data, set the task and dataset once. Choose one launcher from the table.

```bash
export TASK=dual_switch
export DATASET_PATH="$PWD/data/dual_switch_v1_round_1_zarr"
export GPU_ID=0
```

| Policy | Command | What it trains |
| --- | --- | --- |
| Frequency-adaptive diffusion | `bash train_dp_multifreq_cd_30hz_confidence.sh` | Fits the diffusion backbone, trains the confidence predictor, then distills the 30 Hz head. Set `STAGE2_DATASET_PATH` to use separate calibration data. |
| 30 Hz iMeanFlow | `bash train_dp_30hz_imeanflow.sh` | Learns a flow model with one inference step by default. |
| 30 Hz consistency | `bash train_dp_30hz_consistency.sh` | Trains an EDM teacher and distills a CTM student. |
| 30 Hz ACT | `bash train_act_30hz.sh` | Trains a transformer to predict an action sequence. |

Use one GPU to reproduce these examples. The launcher accepts multiple GPU IDs, but some workspaces compute losses on an unwrapped model and distributed gradient synchronization is unverified.

Hydra writes runs under `data/outputs/<date>/`. Checkpoints live in the run's `checkpoints/` directory. To adjust the batch size, append Hydra overrides such as `dataloader.batch_size=8 val_dataloader.batch_size=8`. [Workspace configurations](ImplicitRDP/config)

The 30 Hz iMeanFlow policy uses its instantaneous-velocity head to compute the training loss and its average-velocity head to generate actions. [iMeanFlow reference implementation](https://github.com/Lyy-iiis/imeanflow/blob/main/imf.py)

The 30 Hz consistency policy uses the trajectory-level pseudo-Huber loss from equation 4. To reproduce the previous loss, append `consistency.pseudo_huber_mode=legacy` to the training command. To start distillation from an existing teacher, set `STAGE1_EPOCHS=0` and `PRETRAINED_CKPT_PATH=/path/to/stage1_final.ckpt`. [Consistency Policy paper](https://www.roboticsproceedings.org/rss20/p071.pdf#page=3)

New consistency checkpoints record completed epochs for each training run and track completed student epochs separately. Resuming an older periodic checkpoint can repeat its last epoch because that format saved the epoch index before incrementing it.

## Evaluate

For the 30 Hz consistency policy, the checkpoint stage selects the sampler. The teacher in `stage1_final.ckpt` uses 39 Heun intervals by default, with two model calls per interval. The distilled stage-2 student uses one model call by default. Use `consistency.teacher_num_inference_steps` to change the teacher's interval count and `consistency.num_inference_steps` to change the student's sampling steps.

To evaluate a trained frequency-adaptive checkpoint on recorded demonstrations, set the path to a checkpoint from the corresponding workspace. This example evaluates the full store specified by `DATASET_PATH`. For held-out results, point it to a separate evaluation store.

```bash
CKPT_PATH=/path/to/checkpoints/latest.ckpt TRAINING_STAGE=auto \
    OUTPUT_DIR=results/dual_switch bash eval_dp_multifreq_cd_30hz_confidence.sh
```

The evaluator scores both frequencies against the corresponding demonstration actions. It writes per-episode mean squared errors for the first six action dimensions to `all_episodes_metrics.json` alongside the generated videos. Use the same task, prediction type and loss type as the training run. [Offline evaluator](eval_dp_multifreq_cd_30hz_confidence.py)

For live evaluation or recording, configure the robot and cameras on the deployment workstation first. The deployment guide covers ROS 2, calibration, service startup and selecting a trained checkpoint. [Deployment guide](docs/customized_deployment_guide.md) · [Data collection tips](docs/data_collection_tips.md)

For live 30 Hz iMeanFlow and consistency runs, each stream's latency plus its action update interval must fit within `min(n_action_steps_30hz, policy_horizon_30hz - action_start_30hz)`. All quantities are in 30 Hz steps. For example, an eight-step window with two steps of latency allows an update interval of up to six steps. The runner checks the TCP and gripper settings independently before starting.

## License and attribution

This code derives from the `flow` branch of Chen-Wendi's frequency-adaptive policy repository and the ImplicitRDP project by Wendi Chen and collaborators. The repository license includes the original copyright and third-party notices. [Source repository](https://github.com/Chen-Wendi/frequency_adaptive_policy) · [License](LICENSE)

ImplicitRDP also builds on [Reactive Diffusion Policy](https://github.com/xiaoxiaoxh/reactive_diffusion_policy), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [VQ-BeT](https://github.com/jayLEE0301/vq_bet_official), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [UMI](https://github.com/real-stanford/universal_manipulation_interface) and [Data Scaling Laws](https://github.com/Fanqi-Lin/Data-Scaling-Laws).

If you use the upstream methods, cite the corresponding papers.

```bibtex
@article{chen2025implicitrdp,
  title     = {ImplicitRDP: An End-to-End Visual-Force Diffusion Policy with Structural Slow-Fast Learning},
  author    = {Chen, Wendi and Xue, Han and Wang, Yi and Zhou, Fangyuan and Lv, Jun and Jin, Yang and Tang, Shirun and Wen, Chuan and Lu, Cewu},
  journal   = {arXiv preprint arXiv:2512.10946},
  year      = {2025}
}

@inproceedings{xue2025reactive,
  title     = {Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation},
  author    = {Xue, Han and Ren, Jieji and Chen, Wendi and Zhang, Gu and Fang, Yuan and Gu, Guoying and Xu, Huazhe and Lu, Cewu},
  booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
  year      = {2025}
}
```
