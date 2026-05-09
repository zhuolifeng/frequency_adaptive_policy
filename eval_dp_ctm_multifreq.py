"""
Offline evaluation for DP-CTM Multi-Frequency Policy.

Loads a checkpoint from TrainDPCTMMultiFreqWorkspace, runs inference on the
validation dataset, computes per-frame MSE for both 10Hz and 50Hz heads,
and generates action chunk comparison videos + MSE visualization videos.

Supports two checkpoint types:
  - EDM (Stage 1):  Both 10Hz and 50Hz use multi-step Karras Heun ODE
  - CTM (Stage 2):  10Hz uses Karras Heun ODE, 50Hz uses CTM
                     (single-step when steps=1, native multi-step when steps>1)

Usage:
    # Evaluate CTM checkpoint (default, Stage 2, single-step 50Hz)
    CKPT_PATH=/path/to/ctm.ckpt python eval_dp_ctm_multifreq.py \
        task.dataset.dataset_path=/path/to/zarr_folder

    # Evaluate EDM checkpoint (Stage 1)
    POLICY_TYPE=edm CKPT_PATH=/path/to/edm.ckpt python eval_dp_ctm_multifreq.py \
        task.dataset.dataset_path=/path/to/zarr_folder

    # CTM with native multi-step (50Hz 2-step)
    CKPT_PATH=/path/to/ctm.ckpt python eval_dp_ctm_multifreq.py \
        task.dataset.dataset_path=/path/to/zarr_folder +num_inference_steps_50hz=2

    Optional env vars:
        CKPT_PATH          (required) Path to checkpoint .ckpt file
        POLICY_TYPE=edm|ctm (default: ctm) EDM uses multi-step ODE for 50Hz
        OUTPUT_DIR          (optional) Override output directory
        EVAL_HEAD=10hz|50hz|both  (default: both) Which head(s) to evaluate
        MAX_EPISODES        (default: all) Maximum number of episodes to evaluate
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib
    # The script lives at the project root (frequency_adaptive_policy/)
    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pathlib
import os
import time
import copy
import torch
import dill
import hydra
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, open_dict
import tqdm
import json

from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ====================================================================
# Visualization helpers (adapted from consistency-policy eval)
# ====================================================================

def denormalize_image(img_tensor):
    """Convert normalized image tensor to BGR uint8 for OpenCV."""
    img = img_tensor.cpu().numpy()
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if img.min() < 0:
        img = (img + 1.0) * 127.5
    else:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def create_simple_plot(data, title, ylabel, max_frames=200, threshold_line=None):
    """Create a simple line plot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(data[:max_frames])), data[:max_frames])
    ax.set_title(title)
    ax.set_xlabel('Frame')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if threshold_line is not None:
        ax.axhline(y=threshold_line, color='r', linestyle='--', linewidth=1.5,
                   label=f'Threshold: {threshold_line}')
        ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    return fig


def create_action_chunk_plot(gt_action, pred_action, title, max_dims=3):
    """Plot GT vs Pred action chunk across first max_dims dimensions (1 row x 3 cols)."""
    if gt_action.ndim != 2 or pred_action.ndim != 2:
        raise ValueError(f"Expected 2D action chunks, got gt={gt_action.shape}, pred={pred_action.shape}")
    chunk_len = min(gt_action.shape[0], pred_action.shape[0])
    action_dim = min(gt_action.shape[1], pred_action.shape[1], max_dims)
    gt_action = gt_action[:chunk_len, :action_dim]
    pred_action = pred_action[:chunk_len, :action_dim]
    x = np.arange(chunk_len)

    dim_labels = ["X", "Y", "Z"]
    n_rows, n_cols = 1, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4), squeeze=False)
    axes_flat = axes.flatten()

    for dim_idx in range(action_dim):
        ax = axes_flat[dim_idx]
        ax.plot(x, gt_action[:, dim_idx], color="black", linestyle="--",
                linewidth=1.5, marker="o", markersize=3, label="GT")
        ax.plot(x, pred_action[:, dim_idx], color="tab:orange",
                linewidth=1.5, marker=".", markersize=4, label="Pred")
        label = dim_labels[dim_idx] if dim_idx < len(dim_labels) else f"Dim {dim_idx}"
        ax.set_title(f"Action {label}")
        ax.set_xlabel("Chunk Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[action_dim:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.995), frameon=True)
    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _get_force_magnitude(force_arr, metric="norm"):
    """Extract force magnitude from 6-DoF wrench data."""
    if force_arr.ndim != 2:
        raise ValueError(f"force_arr must be 2D (n_frames, 6). Got shape {force_arr.shape}")
    if force_arr.shape[1] < 6:
        raise ValueError(f"force_arr must have 6 dims. Got shape {force_arr.shape}")
    metric = metric.lower() if metric else "norm"
    if metric in ("norm", "force"):
        return np.linalg.norm(force_arr[:, :3], axis=1)
    if metric == "torque":
        return np.linalg.norm(force_arr[:, 3:], axis=1)
    if metric in ("x", "y", "z"):
        return force_arr[:, {"x": 0, "y": 1, "z": 2}[metric]]
    if metric in ("rx", "ry", "rz"):
        return force_arr[:, {"rx": 3, "ry": 4, "rz": 5}[metric]]
    return np.linalg.norm(force_arr[:, :3], axis=1)


def create_force_plot(force_arr, metric="norm", title=None, max_frames=200):
    """Create a wrench/force plot with mean/max/min annotations."""
    force_values = _get_force_magnitude(force_arr, metric=metric)
    force_values = force_values[:max_frames] if len(force_values) > max_frames else force_values
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(force_values))
    ax.plot(x, force_values, "b-", linewidth=1.5, alpha=0.8)
    ax.fill_between(x, force_values, alpha=0.3)
    mean_val = float(np.mean(force_values)) if len(force_values) else float("nan")
    max_val = float(np.max(force_values)) if len(force_values) else float("nan")
    min_val = float(np.min(force_values)) if len(force_values) else float("nan")
    ax.axhline(y=mean_val, color="r", linestyle="--", linewidth=1, label=f"Mean: {mean_val:.3f}")
    ax.axhline(y=max_val, color="g", linestyle=":", linewidth=1, label=f"Max: {max_val:.3f}")
    ax.axhline(y=min_val, color="orange", linestyle=":", linewidth=1, label=f"Min: {min_val:.3f}")
    plot_title = title if title is not None else f"Force ({metric})"
    ax.set_title(plot_title)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel(f"Force ({metric})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def fig_to_image(fig):
    """Convert matplotlib figure to BGR numpy image."""
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def compute_mse(gt, pred, n_dims=6):
    """Compute per-sample MSE over first n_dims action dimensions."""
    mse_dims = min(n_dims, gt.shape[-1], pred.shape[-1])
    mse = torch.mean((gt[..., :mse_dims] - pred[..., :mse_dims]) ** 2, dim=-1)
    return mse


# ====================================================================
# Policy loading
# ====================================================================

def load_policy(cfg, ckpt_path):
    """Load DP-CTM MultiFreq policy from checkpoint.

    Uses the same workspace-based loading as training, which ensures
    all sub-modules (model, obs_encoder, RNN, normalizer, schedulers)
    are correctly restored.
    """
    from ImplicitRDP.workspace.train_dp_ctm_multifreq_workspace import (
        TrainDPCTMMultiFreqWorkspace,
    )

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)

    workspace = TrainDPCTMMultiFreqWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer", "scaler"], strict=False)

    # Prefer EMA model if available (Stage 1 EMA)
    if cfg.training.use_ema and workspace.ema_model is not None:
        policy = workspace.ema_model
        print("  Using EMA model for evaluation")
    else:
        policy = workspace.model
        print("  Using base model for evaluation")

    # Drop the frozen teacher to save memory
    if hasattr(policy, 'drop_teacher'):
        policy.drop_teacher()

    return policy


# ====================================================================
# Main evaluation
# ====================================================================

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("ImplicitRDP", "config")),
    config_name="train_dp_ctm_multifreq_workspace",
)
def main(cfg):
    # ---- Parse environment variables ----
    ckpt_path = os.environ.get('CKPT_PATH')
    if not ckpt_path:
        raise ValueError("Set CKPT_PATH environment variable to the checkpoint path")

    policy_type = os.environ.get('POLICY_TYPE', 'ctm').lower()
    assert policy_type in ('edm', 'ctm'), f"POLICY_TYPE must be edm or ctm. Got: {policy_type}"
    use_edm_50hz = (policy_type == 'edm')

    eval_head = os.environ.get('EVAL_HEAD', 'both').lower()
    assert eval_head in ('10hz', '50hz', 'both'), f"EVAL_HEAD must be 10hz, 50hz, or both. Got: {eval_head}"

    max_episodes_env = os.environ.get('MAX_EPISODES')

    # 50Hz inference steps: from hydra cfg (+num_inference_steps_50hz=N)
    num_inference_steps_50hz = int(cfg.get('num_inference_steps_50hz', 1))

    # video_fps and output_dir: prefer hydra cfg, fall back to env var
    video_fps_cfg = cfg.get('video_fps', None)

    output_dir = os.environ.get('OUTPUT_DIR')
    if output_dir:
        output_dir = pathlib.Path(output_dir)
    else:
        steps_suffix = f"_50hz{num_inference_steps_50hz}steps"
        output_dir = pathlib.Path(ckpt_path).parent / f"evaluation_dp_ctm_multifreq_{policy_type}{steps_suffix}"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Determine 50Hz mode description
    if use_edm_50hz:
        mode_50hz_desc = f"EDM multi-step ODE ({num_inference_steps_50hz} steps)"
    elif num_inference_steps_50hz <= 1:
        mode_50hz_desc = "CTM single-step"
    else:
        mode_50hz_desc = f"CTM native multi-step ({num_inference_steps_50hz} steps)"

    print("=" * 70)
    print(" DP-CTM Multi-Frequency Policy Offline Evaluation")
    print("=" * 70)
    print(f"  Checkpoint:       {ckpt_path}")
    print(f"  Policy type:      {policy_type.upper()} ({mode_50hz_desc} for 50Hz)")
    print(f"  Eval head(s):     {eval_head}")
    print(f"  50Hz steps:       {num_inference_steps_50hz}")
    print(f"  Use full dataset: {cfg.get('use_full_dataset', False)}")
    print(f"  Output:           {output_dir}")

    # ---- Load policy ----
    policy = load_policy(cfg, ckpt_path)

    # Override training_stage from hydra config (e.g. +training_stage=2)
    # Default: stage 1 for EDM, stage 2 for CTM
    configured_training_stage = cfg.get('training_stage', None)
    if configured_training_stage is not None and hasattr(policy, 'set_training_stage'):
        policy.set_training_stage(int(configured_training_stage))
        print(f"  Override policy.training_stage -> {int(configured_training_stage)}")
    elif use_edm_50hz:
        policy.set_training_stage(1)
        print(f"  Set policy.training_stage -> 1 (EDM mode)")
    else:
        policy.set_training_stage(2)
        print(f"  Set policy.training_stage -> 2 (CTM mode)")

    # Override num_inference_steps from hydra config (e.g. +num_inference_steps=1)
    configured_inference_steps = cfg.get('num_inference_steps', None)
    if configured_inference_steps is not None:
        # For 10hz head: controls Karras ODE bins
        if hasattr(policy, 'karras_scheduler'):
            policy.karras_scheduler.bins = int(configured_inference_steps)
            print(f"  Override karras_scheduler.bins -> {int(configured_inference_steps)}")

    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    # ---- Load dataset & normalizer ----
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    # Use full dataset or validation-only, controlled by +use_full_dataset=True
    val_dataset = dataset if cfg.get('use_full_dataset', False) else dataset.get_validation_dataset()
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)

    # Move to device AFTER setting normalizer
    policy.eval().to(device)

    replay_buffer = val_dataset.replay_buffer
    episode_ends = replay_buffer.episode_ends[:]
    n_episodes = len(episode_ends)

    # Build mapping: sample_idx -> episode_idx
    sampler = val_dataset.sampler
    sample_to_episode = []
    for i in range(len(sampler)):
        buffer_start = sampler.indices[i][0]
        ep_idx = int(np.searchsorted(episode_ends, buffer_start, side='right'))
        sample_to_episode.append(ep_idx)

    # Organize data by episode
    episode_data = [[] for _ in range(n_episodes)]

    print(f"\nLoading {len(val_dataset)} samples from validation set...")
    for data_idx in tqdm.tqdm(range(len(val_dataset)), desc="Loading data"):
        episode_idx = sample_to_episode[data_idx]
        sample = val_dataset[data_idx]
        sample_batch = dict_apply(
            sample,
            lambda x: x.unsqueeze(0).to(device, non_blocking=True)
            if isinstance(x, torch.Tensor) else x
        )
        if 'obs' in sample and isinstance(sample['obs'], dict):
            sample_batch['obs'] = {
                k: v.unsqueeze(0).to(device, non_blocking=True)
                for k, v in sample['obs'].items() if isinstance(v, torch.Tensor)
            }
        if 'extended_obs' in sample and isinstance(sample['extended_obs'], dict):
            sample_batch['extended_obs'] = {
                k: v.unsqueeze(0).to(device, non_blocking=True)
                for k, v in sample['extended_obs'].items() if isinstance(v, torch.Tensor)
            }
        episode_data[episode_idx].append(sample_batch)

    max_episodes = int(max_episodes_env) if max_episodes_env else n_episodes

    # ---- Extract force/wrench data from zarr ----
    episode_ranges = []
    start = 0
    for end in episode_ends:
        episode_ranges.append((start, int(end)))
        start = int(end)

    force_keys = sorted([k for k in replay_buffer.keys() if 'wrench' in k.lower()])
    print(f"  Found force/wrench keys in zarr: {force_keys}")

    episode_force_data_list = [None] * n_episodes
    if len(force_keys) > 0:
        all_force_data = {}
        for fk in force_keys:
            if fk in replay_buffer:
                all_force_data[fk] = replay_buffer[fk][:]
                print(f"  Loaded force data for {fk}: shape {all_force_data[fk].shape}")
        for ep_idx, (ep_start, ep_end) in enumerate(episode_ranges):
            if len(force_keys) == 1:
                episode_force_data_list[ep_idx] = all_force_data[force_keys[0]][ep_start:ep_end]
            else:
                ep_forces = [all_force_data[fk][ep_start:ep_end] for fk in force_keys]
                episode_force_data_list[ep_idx] = np.stack(ep_forces, axis=1)

    # ---- Determine which heads to evaluate ----
    heads_to_eval = []
    if eval_head in ('10hz', 'both'):
        heads_to_eval.append('10hz')
    if eval_head in ('50hz', 'both'):
        heads_to_eval.append('50hz')

    # GT key mapping: 10hz uses 'action', 50hz uses 'action_stack'
    gt_key_map = {'10hz': 'action', '50hz': 'action_stack'}
    normalizer_key_map = {'10hz': 'action', '50hz': 'action_stack'}
    pred_key_map = {'10hz': 'action_pred_10hz', '50hz': 'action_pred_50hz'}

    # Label for display and filenames
    policy_label = policy_type.upper()  # "EDM" or "CTM"

    print(f"\nProcessing {min(n_episodes, max_episodes)} episodes "
          f"for heads: {heads_to_eval} ({policy_label} mode)...")

    all_metrics = {}
    all_inference_times = []

    for episode_idx in range(min(n_episodes, max_episodes)):
        if len(episode_data[episode_idx]) == 0:
            continue

        n_frames = len(episode_data[episode_idx])
        print(f"\n  Episode {episode_idx} ({n_frames} frames)")

        # Per-head data collection
        head_data = {}
        for head in heads_to_eval:
            head_data[head] = {
                'mse': [],
                'gt_chunks': [],
                'pred_chunks': [],
            }
        inference_times = []

        # ---- Run inference ----
        with torch.no_grad():
            for frame_idx, batch in enumerate(tqdm.tqdm(
                    episode_data[episode_idx],
                    desc=f"Episode {episode_idx} inference")):

                obs_dict = {k: v for k, v in batch['obs'].items()
                            if isinstance(v, torch.Tensor)}
                extended_obs_dict = batch.get('extended_obs', None)

                # Time inference
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t_start = time.time()
                result = policy.predict_action(
                    obs_dict,
                    extended_obs_dict=extended_obs_dict,
                    use_edm_50hz=use_edm_50hz,
                    num_inference_steps=num_inference_steps_50hz,
                )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t_end = time.time()
                inference_times.append(t_end - t_start)

                # Process each head
                for head in heads_to_eval:
                    gt_key = gt_key_map[head]
                    pred_key = pred_key_map[head]

                    gt_action = batch[gt_key][:, 1:]  # skip first step
                    pred_action = result[pred_key][:, 1:]
                    min_len = min(gt_action.shape[1], pred_action.shape[1])
                    gt_action = gt_action[:, :min_len]
                    pred_action = pred_action[:, :min_len]

                    # MSE
                    mse = compute_mse(gt_action, pred_action)
                    head_data[head]['mse'].append(float(mse[0].mean().item()))

                    # Store chunks
                    head_data[head]['gt_chunks'].append(
                        gt_action[0].detach().cpu().numpy()
                    )
                    head_data[head]['pred_chunks'].append(
                        pred_action[0].detach().cpu().numpy()
                    )

        avg_time = np.mean(inference_times)
        all_inference_times.extend(inference_times)

        episode_metrics = {
            'avg_inference_time_ms': float(avg_time * 1000),
        }
        for head in heads_to_eval:
            mse_list = head_data[head]['mse']
            episode_metrics[f'mse_{head}'] = [float(x) for x in mse_list]
            episode_metrics[f'mean_mse_{head}'] = float(np.mean(mse_list))
        all_metrics[str(episode_idx)] = episode_metrics

        # ---- Generate videos for each head ----
        print(f"  Generating videos for episode {episode_idx}...")
        force_data = episode_force_data_list[episode_idx]
        has_force = force_data is not None and len(force_data) > 0

        for head in heads_to_eval:
            freq_hz = 50 if head == '50hz' else 10
            video_fps = int(video_fps_cfg) if video_fps_cfg is not None else freq_hz
            mse_data = head_data[head]['mse']
            gt_chunks = head_data[head]['gt_chunks']
            pred_chunks = head_data[head]['pred_chunks']

            video_writers = {}

            for frame_idx in tqdm.tqdm(
                    range(n_frames),
                    desc=f"Episode {episode_idx} video ({head})"):

                sample = episode_data[episode_idx][frame_idx]

                # Get observation image for display
                obs_img = np.zeros((240, 320, 3), dtype=np.uint8)
                for img_key in ['right_wrist_img', 'external_img']:
                    if img_key in sample['obs']:
                        img_tensor = sample['obs'][img_key][0, -1]  # last obs step
                        obs_img = denormalize_image(img_tensor)
                        obs_img = cv2.resize(obs_img, (640, 480))
                        break

                # Add text overlay
                overlay = obs_img.copy()
                cv2.rectangle(overlay, (5, 5), (obs_img.shape[1] - 5, 50),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, obs_img, 0.4, 0, obs_img)
                cv2.putText(
                    obs_img,
                    f"[DP-{policy_label} {head.upper()}] Frame: {frame_idx + 1}/{n_frames}  "
                    f"MSE: {mse_data[frame_idx]:.5f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

                # --- Video 1: MSE plot ---
                fig_mse = create_simple_plot(
                    mse_data[:frame_idx + 1],
                    f'DP-{policy_label} {head.upper()} Action MSE', 'MSE', n_frames)
                mse_plot = fig_to_image(fig_mse)

                panels_mse = [mse_plot, obs_img]
                target_w = max(p.shape[1] for p in panels_mse)
                resized = []
                for p in panels_mse:
                    if p.shape[1] != target_w:
                        new_h = int(p.shape[0] * target_w / p.shape[1])
                        resized.append(cv2.resize(p, (target_w, new_h)))
                    else:
                        resized.append(p)
                combined_mse = np.vstack(resized)

                vname_mse = f'mse_{head}'
                if vname_mse not in video_writers:
                    h, w = combined_mse.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vpath = output_dir / f"episode_{episode_idx:03d}_dp_{policy_type}_mse_{head}.mp4"
                    video_writers[vname_mse] = cv2.VideoWriter(
                        str(vpath), fourcc, video_fps, (w, h))
                    print(f"    Creating: {vpath}")
                video_writers[vname_mse].write(combined_mse)

                # --- Video 2: Action chunk (top) + Wrench (middle) + Obs (bottom) ---
                fig_chunk = create_action_chunk_plot(
                    gt_chunks[frame_idx], pred_chunks[frame_idx],
                    title=f'DP-{policy_label} {head.upper()} Action Chunk GT vs Pred '
                          f'(Frame {frame_idx + 1})')
                chunk_plot = fig_to_image(fig_chunk)

                # Middle: wrench/force plot
                if has_force:
                    force_arr = np.array(force_data[:frame_idx + 1])
                    if force_arr.ndim == 3:
                        force_arr = force_arr.mean(axis=1)
                    fig_force = create_force_plot(
                        force_arr, metric="norm",
                        title="Wrench (norm)", max_frames=n_frames)
                    force_plot = fig_to_image(fig_force)
                else:
                    fig_placeholder, ax_ph = plt.subplots(figsize=(12, 4))
                    ax_ph.text(0.5, 0.5, "No wrench data available",
                               ha='center', va='center', fontsize=16,
                               color='gray')
                    ax_ph.set_xlim(0, 1)
                    ax_ph.set_ylim(0, 1)
                    ax_ph.axis('off')
                    plt.tight_layout()
                    force_plot = fig_to_image(fig_placeholder)

                # Stack: action chunk (top) + wrench (middle) + obs (bottom)
                panels_chunk = [chunk_plot, force_plot, obs_img]
                target_w_c = max(p.shape[1] for p in panels_chunk)
                resized_chunk = []
                for p in panels_chunk:
                    if p.shape[1] != target_w_c:
                        new_h = int(p.shape[0] * target_w_c / p.shape[1])
                        resized_chunk.append(cv2.resize(p, (target_w_c, new_h)))
                    else:
                        resized_chunk.append(p)
                combined_chunk = np.vstack(resized_chunk)

                vname_chunk = f'action_chunk_{head}'
                if vname_chunk not in video_writers:
                    h, w = combined_chunk.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vpath = output_dir / f"episode_{episode_idx:03d}_dp_{policy_type}_action_chunk_{head}.mp4"
                    video_writers[vname_chunk] = cv2.VideoWriter(
                        str(vpath), fourcc, video_fps, (w, h))
                    print(f"    Creating: {vpath}")
                video_writers[vname_chunk].write(combined_chunk)

            for writer in video_writers.values():
                writer.release()

        # Print episode summary
        summary_parts = []
        for head in heads_to_eval:
            summary_parts.append(
                f"Mean MSE ({head}): {episode_metrics[f'mean_mse_{head}']:.6f}"
            )
        print(f"  Episode {episode_idx} done. {', '.join(summary_parts)}, "
              f"Avg inference: {avg_time * 1000:.1f}ms")

    # ---- Save metrics ----
    overall_metrics = {}
    for head in heads_to_eval:
        vals = [m[f'mean_mse_{head}'] for m in all_metrics.values()
                if f'mean_mse_{head}' in m]
        overall_metrics[f'overall_mean_mse_{head}'] = float(np.mean(vals)) if vals else 0
    overall_time = np.mean(all_inference_times) * 1000 if all_inference_times else 0

    summary = {
        'policy_type': f'dp_ctm_multifreq_{policy_type}',
        'checkpoint_stage': 'edm' if use_edm_50hz else 'ctm',
        'eval_heads': heads_to_eval,
        'num_inference_steps_50hz': num_inference_steps_50hz,
        'ckpt_path': str(ckpt_path),
        'overall_avg_inference_time_ms': float(overall_time),
        **overall_metrics,
        'episodes': all_metrics,
    }

    if all_metrics:
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print(" DP-CTM Multi-Frequency Evaluation Summary")
    print("=" * 70)
    for ep_idx, metrics in all_metrics.items():
        parts = []
        for head in heads_to_eval:
            if f'mean_mse_{head}' in metrics:
                parts.append(f"MSE({head})={metrics[f'mean_mse_{head}']:.6f}")
        print(f"  Episode {ep_idx}: {', '.join(parts)}, "
              f"Avg Time = {metrics['avg_inference_time_ms']:.1f}ms")

    print(f"\n  Evaluated heads:        {heads_to_eval}")
    print(f"  50Hz steps:             {num_inference_steps_50hz}")
    for head in heads_to_eval:
        key = f'overall_mean_mse_{head}'
        print(f"  Overall Mean MSE ({head}): {overall_metrics.get(key, 0):.6f}")
    print(f"  Overall Avg Inference:  {overall_time:.1f}ms")
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
