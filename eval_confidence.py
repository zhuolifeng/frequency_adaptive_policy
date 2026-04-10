import pathlib
import os
import torch
import dill
import hydra
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import OmegaConf
import tqdm
import json

from ImplicitRDP.workspace.base_workspace import BaseWorkspace
from ImplicitRDP.policy.diffusion_transformer_image_policy import DiffusionTransformerImagePolicy
from ImplicitRDP.policy.hybrid_diffusion_act_policy import HybridDiffusionACTPolicy
from ImplicitRDP.policy.hybrid_diffusion_act_stagewise_policy import HybridDiffusionACTStagewisePolicy
from ImplicitRDP.policy.act_confidence_policy import ACTConfidencePolicy
from ImplicitRDP.policy.act_confidence_dual_encoder_policy import ACTConfidenceDualEncoderPolicy
from ImplicitRDP.policy.act_confidence_dual_encoder_stagewise_policy import ACTConfidenceDualEncoderStagewisePolicy
from ImplicitRDP.policy.act_confidence_stagewise_policy import ACTConfidenceStagewisePolicy
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.common.pytorch_util import dict_apply

plt.switch_backend('Agg')
OmegaConf.register_new_resolver("eval", eval, replace=True)


def denormalize_image(img_tensor):
    img = img_tensor.cpu().numpy()
    
    # Process channel dimension
    if img.shape[0] == 3:  # (C, H, W)
        img = img.transpose(1, 2, 0)  # -> (H, W, C)
    
    # Unnormalize from [-1, 1] or [0, 1] to [0, 255]
    if img.min() < 0:  # Assume input was normalized to [-1, 1]
        img = (img + 1.0) * 127.5
    else:  # Assume [0, 1] normalization
        img = img * 255.0
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Convert RGB to BGR (OpenCV format)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

def align_gt_pred(gt_action, pred_action, policy):
    """Align GT and prediction actions - remove 1 step offset (hack for training alignment)"""
    if gt_action.shape[1] > pred_action.shape[1]:
        gt_action = gt_action[:, 1:1+pred_action.shape[1]]

    # Ensure same length
    min_steps = min(gt_action.shape[1], pred_action.shape[1])
    gt_action = gt_action[:, :min_steps]
    pred_action = pred_action[:, :min_steps]

    return gt_action, pred_action

def compute_errors(gt_action, pred_action, policy):
    # Align actions first
    gt_action, pred_action = align_gt_pred(gt_action, pred_action, policy)

    action_dim = gt_action.shape[-1]

    # XYZ error (first 3 dimensions)
    gt_xyz = gt_action[:, :3].cpu().numpy()
    pred_xyz = pred_action[:, :3].cpu().numpy()
    error_xyz = gt_xyz - pred_xyz

    # RPY error (dimensions 3-5 if available)
    if action_dim >= 6:
        gt_rpy = gt_action[:, 3:6].cpu().numpy()
        pred_rpy = pred_action[:, 3:6].cpu().numpy()
        error_rpy = gt_rpy - pred_rpy
    else:
        error_rpy = np.zeros_like(error_xyz)

    # MSE (first 6 dimensions or all available)
    mse_dims = min(6, action_dim)
    mse = torch.mean((gt_action[..., :mse_dims] - pred_action[..., :mse_dims]) ** 2, dim=-1).cpu().numpy()

    return mse, error_xyz, error_rpy


def create_simple_plot(data, title, ylabel, max_frames=200, threshold_line=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(data[:max_frames])), data[:max_frames])
    ax.set_title(title)
    ax.set_xlabel('Frame')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    
    # Add threshold line if provided
    if threshold_line is not None:
        ax.axhline(y=threshold_line, color='r', linestyle='--', linewidth=1.5, label=f'Threshold: {threshold_line}')
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_error_plot(error_data, title, max_frames=200):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ['X', 'Y', 'Z'] if 'Z' in title else ['Roll', 'Pitch', 'Yaw']

    for i in range(3):
        if len(error_data) > 0:
            axis_data = [frame[i] for frame in error_data[:max_frames]]
            axes[i].plot(range(len(axis_data)), axis_data)
        axes[i].set_title(f'{labels[i]} Error')
        axes[i].set_xlabel('Frame')
        axes[i].set_ylabel('Error')
        axes[i].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    return fig

def fig_to_image(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # Remove alpha channel
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def _get_force_magnitude(force_arr: np.ndarray, metric: str = "norm") -> np.ndarray:
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
    # default to force norm
    return np.linalg.norm(force_arr[:, :3], axis=1)

def create_force_plot(force_arr: np.ndarray, metric: str = "norm", title: str | None = None, max_frames: int = 200):
    force_values = _get_force_magnitude(force_arr, metric=metric)
    force_values = force_values[:max_frames] if len(force_values) > max_frames else force_values

    fig, ax = plt.subplots(figsize=(12, 6))
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

def create_action_chunk_plot(
    gt_action: np.ndarray,
    pred_action: np.ndarray,
    title: str,
    max_dims: int = 6,
):
    if gt_action.ndim != 2 or pred_action.ndim != 2:
        raise ValueError(
            f"Expected 2D action chunks, got gt={gt_action.shape}, pred={pred_action.shape}"
        )

    chunk_len = min(gt_action.shape[0], pred_action.shape[0])
    action_dim = min(gt_action.shape[1], pred_action.shape[1], max_dims)

    gt_action = gt_action[:chunk_len, :action_dim]
    pred_action = pred_action[:chunk_len, :action_dim]
    x = np.arange(chunk_len)

    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8), squeeze=False)
    axes_flat = axes.flatten()

    for dim_idx in range(action_dim):
        ax = axes_flat[dim_idx]
        ax.plot(x, gt_action[:, dim_idx], color="black", linestyle="--", linewidth=1.5, marker="o", markersize=3, label="GT")
        ax.plot(x, pred_action[:, dim_idx], color="tab:orange", linewidth=1.5, marker=".", markersize=4, label="Pred")
        ax.set_title(f"Action Dim {dim_idx}")
        ax.set_xlabel("Chunk Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[action_dim:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995), frameon=True)
    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig

def process_single_episode(policy, episode_data_list, episode_force_data, has_confidence, device, output_dir, episode_idx, video_fps=10):
    logger.info(f"Processing Episode {episode_idx} ({len(episode_data_list)} frames)")

    # Reset policy state if needed
    if hasattr(policy, 'reset'):
        policy.reset()

    # Collect all frame data
    mse_data = []
    xyz_error_data = []
    rpy_error_data = []
    log_sigma_data = [] if has_confidence else None
    gt_action_chunk_data = []
    pred_action_chunk_data = []

    # Force data for force curve - use the extracted zarr data (same as force.py)
    force_data = episode_force_data if episode_force_data is not None else []
    has_force = len(force_data) > 0
    logger.info(f"Force data available: {has_force}, shape: {force_data.shape if has_force else 'N/A'}")

    with torch.no_grad():
        for frame_idx, sample in enumerate(tqdm.tqdm(episode_data_list, desc=f"Episode {episode_idx}")):
            # Get observation data
            obs_dict = {k: v for k, v in sample['obs'].items() if isinstance(v, torch.Tensor)}
            gt_action = sample['action'][:,1:16]
            extended_obs_dict = None
            if 'extended_obs' in sample and sample['extended_obs'] is not None:
                extended_obs_dict = {k: v for k, v in sample['extended_obs'].items() if isinstance(v, torch.Tensor)}

            # Policy prediction
            pred_result = policy.predict_action(obs_dict=obs_dict, extended_obs_dict=extended_obs_dict)
            pred_action = pred_result['action']

            # Compute errors
            mse, error_xyz, error_rpy = compute_errors(gt_action, pred_action, policy)
            aligned_gt_action, aligned_pred_action = align_gt_pred(gt_action, pred_action, policy)

            # Store data - take average MSE over the entire chunk
            mse_data.append(float(mse[0].mean()))
            xyz_error_data.append(error_xyz[0].tolist())
            rpy_error_data.append(error_rpy[0].tolist())
            gt_action_chunk_data.append(aligned_gt_action[0].detach().cpu().numpy())
            pred_action_chunk_data.append(aligned_pred_action[0].detach().cpu().numpy())

            # Compute confidence if available
            if has_confidence:
                try:
                    if hasattr(policy, 'act_encoder') and hasattr(policy, 'confidence_encoder'):
                        # Dual encoder policy - use confidence encoder
                        obs_tokens, _ = policy._predict_obs_tokens(
                            obs_dict, extended_obs_dict,
                            encoder_type='confidence',
                            return_temporal_cond=True
                        )
                    else:
                        # other policies
                        obs_tokens, _ = policy._predict_obs_tokens(
                            obs_dict, extended_obs_dict, 
                            return_temporal_cond=True
                        )
                    log_sigma = policy._compute_confidence_logits(obs_tokens)
                    log_sigma_val = float(log_sigma.mean().item()) if log_sigma is not None else float('nan')
                    log_sigma_data.append(log_sigma_val)
                except Exception as e:
                    logger.warning(f"Failed to compute confidence: {e}")
                    log_sigma_data.append(float('nan'))

    # Generate videos
    logger.info("Generating videos...")
    video_writers = {}

    for frame_idx in tqdm.tqdm(range(len(episode_data_list)), desc=f"Episode {episode_idx} Video"):
        # Create plots
        plots = {}

        # MSE plot
        fig_mse = create_simple_plot(mse_data[:frame_idx+1], 'Action MSE', 'MSE', len(episode_data_list))
        plots['action_mse'] = fig_to_image(fig_mse)

        # XYZ error plot
        fig_xyz = create_error_plot(xyz_error_data[:frame_idx+1], 'XYZ Position Error', len(episode_data_list))
        plots['xyz_error'] = fig_to_image(fig_xyz)

        # RPY error plot
        fig_rpy = create_error_plot(rpy_error_data[:frame_idx+1], 'RPY Rotation Error', len(episode_data_list))
        plots['rpy_error'] = fig_to_image(fig_rpy)

        # Current frame action chunk plot
        fig_action_chunk = create_action_chunk_plot(
            gt_action_chunk_data[frame_idx],
            pred_action_chunk_data[frame_idx],
            title=f'Action Chunk GT vs Pred (Frame {frame_idx+1})',
        )
        plots['action_chunk'] = fig_to_image(fig_action_chunk)

        # Get observation image
        sample = episode_data_list[frame_idx]
        obs_img = np.zeros((240, 320, 3), dtype=np.uint8)
        if 'right_wrist_img' in sample['obs']:
            img_tensor = sample['obs']['right_wrist_img'][0, -1]
            obs_img = denormalize_image(img_tensor)
            obs_img = cv2.resize(obs_img, (640, 480))

        # Add text overlay
        overlay = obs_img.copy()
        cv2.rectangle(overlay, (5, 5), (obs_img.shape[1]-5, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, obs_img, 0.4, 0, obs_img)

        cv2.putText(obs_img, f"Frame: {frame_idx+1}/{len(episode_data_list)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if has_confidence and frame_idx < len(log_sigma_data):
            sigma_val = log_sigma_data[frame_idx]
            if not math.isnan(sigma_val):
                cv2.putText(obs_img, f"log σ: {sigma_val:.4f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Create special 3-panel confidence video (upper: confidence, middle: force, lower: frame)
        if has_confidence and log_sigma_data and has_force and len(force_data) > 0:
            # Create confidence plot (smaller for 3-panel layout) with threshold line
            fig_sigma = create_simple_plot(log_sigma_data[:frame_idx+1], 'Confidence (log σ)', 'log σ', len(episode_data_list), threshold_line=2.5)
            confidence_plot = fig_to_image(fig_sigma)
            
            # Create force plot - pass the zarr force array directly
            force_arr = np.array(force_data[:frame_idx+1])  # Shape: (n_frames, 6) or (n_frames, n_keys, 6)
            if force_arr.ndim == 3:
                # Multiple force keys - average across keys to get a single curve
                force_arr = force_arr.mean(axis=1)
            fig_force = create_force_plot(force_arr, metric="norm", title="Episode Force (norm)", max_frames=len(episode_data_list))
            force_plot = fig_to_image(fig_force)
            
            # Resize all to same width
            target_w = max(confidence_plot.shape[1], force_plot.shape[1], obs_img.shape[1])
            
            confidence_plot_resized = cv2.resize(confidence_plot, (target_w, int(confidence_plot.shape[0] * target_w / confidence_plot.shape[1])))
            force_plot_resized = cv2.resize(force_plot, (target_w, int(force_plot.shape[0] * target_w / force_plot.shape[1])))
            obs_img_resized = cv2.resize(obs_img, (target_w, int(obs_img.shape[0] * target_w / obs_img.shape[1])))
            
            # Stack vertically: confidence -> force -> frame
            combined_3panel = np.vstack([confidence_plot_resized, force_plot_resized, obs_img_resized])
            
            # Initialize video writer for 3-panel confidence video
            video_name_3panel = 'confidence_with_force'
            if video_name_3panel not in video_writers:
                h, w = combined_3panel.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = output_dir / f"episode_{episode_idx:03d}_{video_name_3panel}.mp4"
                video_writers[video_name_3panel] = cv2.VideoWriter(str(video_path), fourcc, video_fps, (w, h))
                logger.info(f"Creating 3-panel video: {video_path}")
            
            video_writers[video_name_3panel].write(combined_3panel)

        # Also create 2-panel confidence video if force not available
        if has_confidence and log_sigma_data:
            fig_sigma = create_simple_plot(log_sigma_data[:frame_idx+1], 'Confidence (log σ)', 'log σ', len(episode_data_list), threshold_line=2.5)
            plots['log_sigma'] = fig_to_image(fig_sigma)

        # Combine plot and image for each video
        for video_name, plot_img in plots.items():
            # Resize images to match
            plot_h, plot_w = plot_img.shape[:2]
            obs_h, obs_w = obs_img.shape[:2]
            target_w = max(plot_w, obs_w)

            if plot_w != target_w:
                plot_img = cv2.resize(plot_img, (target_w, plot_h))
            if obs_w != target_w:
                obs_img_resized = cv2.resize(obs_img, (target_w, obs_h))
            else:
                obs_img_resized = obs_img

            # Stack vertically
            combined = np.vstack([plot_img, obs_img_resized])

            # Initialize video writer if needed
            if video_name not in video_writers:
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = output_dir / f"episode_{episode_idx:03d}_{video_name}.mp4"
                video_writers[video_name] = cv2.VideoWriter(str(video_path), fourcc, video_fps, (w, h))
                logger.info(f"Creating video: {video_path}")

            # Write frame
            video_writers[video_name].write(combined)

    # Close all video writers
    for writer in video_writers.values():
        writer.release()

    logger.info(f"✓ Episode {episode_idx} completed")
    return {'mse_data': mse_data, 'xyz_error_data': xyz_error_data, 'rpy_error_data': rpy_error_data,
            'log_sigma_data': log_sigma_data if has_confidence else None,
            'force_data': force_data if has_force else None}


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('ImplicitRDP', 'config')),
    config_name="train_act_confidence_workspace"
)
def main(cfg):
    logger.info("Starting policy evaluation...")

    # Load checkpoint
    ckpt_path = os.environ.get('CKPT_PATH')
    if not ckpt_path:
        raise ValueError("Set CKPT_PATH environment variable")

    # Get output directory from environment variable or use default
    output_dir = os.environ.get('OUTPUT_DIR')
    if output_dir:
        output_dir = pathlib.Path(output_dir)
    else:
        output_dir = pathlib.Path(ckpt_path).parent / "evaluation_videos"
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    logger.info(f"Output directory: {output_dir}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)

    # Create workspace and load policy
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer", "scaler"], strict=False)

    # Get policy
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if not isinstance(policy, (ACTConfidencePolicy, ACTConfidenceDualEncoderPolicy, ACTConfidenceDualEncoderStagewisePolicy, ACTConfidenceStagewisePolicy, HybridDiffusionACTPolicy, HybridDiffusionACTStagewisePolicy, DiffusionTransformerImagePolicy)):
        raise ValueError(f"Unsupported policy type: {type(policy)}")

    # Determine evaluation parameters
    if isinstance(policy, (ACTConfidencePolicy, ACTConfidenceDualEncoderPolicy, ACTConfidenceDualEncoderStagewisePolicy, ACTConfidenceStagewisePolicy)):
        has_confidence = policy.confidence_head is not None
    elif isinstance(policy, (HybridDiffusionACTPolicy, HybridDiffusionACTStagewisePolicy)):
        has_confidence = hasattr(policy, 'confidence_head') and policy.confidence_head is not None
    else:  # DiffusionTransformerImagePolicy
        has_confidence = False

    logger.info(f"Policy type: {type(policy).__name__}")
    logger.info(f"Confidence head: {'Enabled' if has_confidence else 'Disabled'}")

    # Setup device and policy
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    policy.eval().to(device)

    # Load dataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset if cfg.get('use_full_dataset', False) else dataset.get_validation_dataset()

    # Get episode data
    replay_buffer = val_dataset.replay_buffer
    episode_ends = replay_buffer.episode_ends[:]
    n_episodes = len(episode_ends)

    # Organize data by episodes
    episode_data = [[] for _ in range(n_episodes)]
    sampler = val_dataset.sampler

    for data_idx in tqdm.tqdm(range(len(val_dataset)), desc="Loading data"):
        episode_idx = sampler.indices_to_episode_idx[data_idx]
        sample = val_dataset[data_idx]
        sample_batch = dict_apply(sample, lambda x: x.unsqueeze(0).to(device, non_blocking=True)
                                 if isinstance(x, torch.Tensor) else x)
        if 'obs' in sample and isinstance(sample['obs'], dict):
            sample_batch['obs'] = {k: v.unsqueeze(0).to(device, non_blocking=True)
                                   for k, v in sample['obs'].items() if isinstance(v, torch.Tensor)}
        if 'extended_obs' in sample and isinstance(sample['extended_obs'], dict):
            sample_batch['extended_obs'] = {k: v.unsqueeze(0).to(device, non_blocking=True)
                                            for k, v in sample['extended_obs'].items() if isinstance(v, torch.Tensor)}
        episode_data[episode_idx].append(sample_batch)

    # Create output directory (already created above)
    video_fps = cfg.get('video_fps', 10)

    # Extract force data from zarr (same approach as force.py)
    episode_ranges = []
    start = 0
    for end in episode_ends:
        episode_ranges.append((start, int(end)))
        start = int(end)

    # Find force keys in zarr
    force_keys = []
    for k in replay_buffer.keys():
        if 'wrench' in k.lower():
            force_keys.append(k)
    force_keys = sorted(force_keys)
    logger.info(f"Found force keys in zarr: {force_keys}")

    # Load force data from zarr
    episode_force_data_list = [None] * n_episodes
    if len(force_keys) > 0:
        # Load all force data
        all_force_data = {}
        for fk in force_keys:
            if fk in replay_buffer:
                all_force_data[fk] = replay_buffer[fk][:]
                logger.info(f"Loaded force data for {fk}: shape {all_force_data[fk].shape}")

        # Extract force for each episode
        for ep_idx, (ep_start, ep_end) in enumerate(episode_ranges):
            if len(force_keys) == 1:
                # Single force key - use directly
                episode_force_data_list[ep_idx] = all_force_data[force_keys[0]][ep_start:ep_end]
            else:
                # Multiple force keys - stack them
                ep_forces = [all_force_data[fk][ep_start:ep_end] for fk in force_keys]
                episode_force_data_list[ep_idx] = np.stack(ep_forces, axis=1)  # (n_frames, n_keys, 6)

    logger.info(f"Processing {n_episodes} episodes...")
    logger.info(f"Output directory: {output_dir}")

    # Collect all confidence data and save to JSON
    all_confidence_data = {}
    for episode_idx in range(min(n_episodes, cfg.get('max_episodes', n_episodes))):
        if len(episode_data[episode_idx]) == 0:
            continue
        episode_result = process_single_episode(
            policy=policy,
            episode_data_list=episode_data[episode_idx],
            episode_force_data=episode_force_data_list[episode_idx],
            has_confidence=has_confidence,
            device=device,
            output_dir=output_dir,
            episode_idx=episode_idx,
            video_fps=video_fps
        )
        
        if has_confidence and episode_result['log_sigma_data'] is not None:
            all_confidence_data[episode_idx] = episode_result['log_sigma_data']
    
    # Save all confidence data to a single JSON file
    if all_confidence_data:
        all_confidence_data_json = {str(k): [float(x) if not np.isnan(x) else None for x in v] for k, v in all_confidence_data.items()}
        all_confidence_json_path = output_dir / "all_episodes_confidence.json"
        with open(all_confidence_json_path, 'w') as f:
            json.dump(all_confidence_data_json, f, indent=2)
        logger.info(f"Saved confidence data to {all_confidence_json_path}")

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
