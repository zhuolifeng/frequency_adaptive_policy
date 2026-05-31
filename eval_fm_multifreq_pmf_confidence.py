"""
Evaluation script for PMF multi-frequency confidence policy.

Generates per-episode:
  - MSE 10hz video (MSE plot + force plot + obs image)
  - MSE 50hz video (MSE plot + force plot + obs image)
  - Action chunk 10hz video (GT vs Pred per-dim plot + obs image)
  - Action chunk 50hz video (GT vs Pred per-dim plot + obs image)
  - Confidence video (confidence curve + force plot + obs image)
"""

import pathlib
import os
import torch
import dill
import hydra
import numpy as np
import cv2
import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import OmegaConf
import tqdm
import json

from ImplicitRDP.workspace.base_workspace import BaseWorkspace
from ImplicitRDP.policy.fm_multifreq_pmf_confidence_policy import (
    FMMultiFreqPMFConfidencePolicy,
)
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.common.pytorch_util import dict_apply

plt.switch_backend('Agg')
OmegaConf.register_new_resolver("eval", eval, replace=True)


def denormalize_image(img_tensor):
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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(data[:max_frames])), data[:max_frames])
    ax.set_title(title)
    ax.set_xlabel('Frame')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if threshold_line is not None:
        ax.axhline(y=threshold_line, color='r', linestyle='--', linewidth=1.5, label=f'Threshold: {threshold_line}')
        ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    return fig


def fig_to_image(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]
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


def create_action_chunk_plot(gt_action: np.ndarray, pred_action: np.ndarray, title: str, max_dims: int = 6):
    if gt_action.ndim != 2 or pred_action.ndim != 2:
        raise ValueError(f"Expected 2D action chunks, got gt={gt_action.shape}, pred={pred_action.shape}")
    chunk_len = min(gt_action.shape[0], pred_action.shape[0])
    action_dim = min(gt_action.shape[1], pred_action.shape[1], max_dims)
    gt_action = gt_action[:chunk_len, :action_dim]
    pred_action = pred_action[:chunk_len, :action_dim]
    x = np.arange(chunk_len)
    n_rows, n_cols = 2, 3
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


def compute_mse(gt, pred, n_dims=6):
    """Compute per-sample MSE over first n_dims action dimensions."""
    mse_dims = min(n_dims, gt.shape[-1], pred.shape[-1])
    mse = torch.mean((gt[..., :mse_dims] - pred[..., :mse_dims]) ** 2, dim=-1)
    return mse


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('ImplicitRDP', 'config')),
    config_name="train_fm_multifreq_pmf_confidence_workspace"
)
def main(cfg):
    logger.info("Starting PMF MultiFreq Confidence policy evaluation...")

    ckpt_path = os.environ.get('CKPT_PATH')
    if not ckpt_path:
        raise ValueError("Set CKPT_PATH environment variable")

    output_dir = os.environ.get('OUTPUT_DIR')
    if output_dir:
        output_dir = pathlib.Path(output_dir)
    else:
        output_dir = pathlib.Path(ckpt_path).parent / "evaluation_fm_multifreq_pmf_confidence"
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    logger.info(f"Output directory: {output_dir}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer", "scaler"], strict=False)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if not isinstance(policy, FMMultiFreqPMFConfidencePolicy):
        raise ValueError(
            f"Expected FMMultiFreqPMFConfidencePolicy, got {type(policy)}"
        )

    configured_inference_steps = cfg.get('num_inference_steps', None)
    if configured_inference_steps is not None:
        original_inference_steps = getattr(policy, 'num_inference_steps', None)
        policy.num_inference_steps = int(configured_inference_steps)
        logger.info(f"Override policy.num_inference_steps: {original_inference_steps} -> {policy.num_inference_steps}")

    logger.info(f"Policy type: {type(policy).__name__}")
    logger.info(f"Inference steps: {policy.num_inference_steps}")
    logger.info(f"Inference head: {policy.inference_head}")
    logger.info(
        "Confidence head: "
        f"{'enabled' if policy.confidence_head is not None else 'disabled'}"
    )

    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    policy.eval().to(device)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset if cfg.get('use_full_dataset', False) else dataset.get_validation_dataset()

    replay_buffer = val_dataset.replay_buffer
    episode_ends = replay_buffer.episode_ends[:]
    n_episodes = len(episode_ends)

    # Organize data by episode
    episode_data = [[] for _ in range(n_episodes)]
    sampler = val_dataset.sampler

    for data_idx in tqdm.tqdm(range(len(val_dataset)), desc="Loading data"):
        episode_idx = sampler.indices_to_episode_idx[data_idx]
        sample = val_dataset[data_idx]
        sample_batch = dict_apply(sample, lambda x: x.unsqueeze(0).to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x)
        if 'obs' in sample and isinstance(sample['obs'], dict):
            sample_batch['obs'] = {k: v.unsqueeze(0).to(device, non_blocking=True) for k, v in sample['obs'].items() if isinstance(v, torch.Tensor)}
        if 'extended_obs' in sample and isinstance(sample['extended_obs'], dict):
            sample_batch['extended_obs'] = {k: v.unsqueeze(0).to(device, non_blocking=True) for k, v in sample['extended_obs'].items() if isinstance(v, torch.Tensor)}
        episode_data[episode_idx].append(sample_batch)

    video_fps = cfg.get('video_fps', 10)
    dataset_obs_temporal_downsample_ratio = cfg.task.dataset.get('obs_temporal_downsample_ratio', 1)

    # Extract force data from zarr
    episode_ranges = []
    start = 0
    for end in episode_ends:
        episode_ranges.append((start, int(end)))
        start = int(end)

    force_keys = sorted([k for k in replay_buffer.keys() if 'wrench' in k.lower()])
    logger.info(f"Found force keys in zarr: {force_keys}")

    episode_force_data_list = [None] * n_episodes
    if len(force_keys) > 0:
        all_force_data = {}
        for fk in force_keys:
            if fk in replay_buffer:
                all_force_data[fk] = replay_buffer[fk][:]
                logger.info(f"Loaded force data for {fk}: shape {all_force_data[fk].shape}")
        for ep_idx, (ep_start, ep_end) in enumerate(episode_ranges):
            if len(force_keys) == 1:
                episode_force_data_list[ep_idx] = all_force_data[force_keys[0]][ep_start:ep_end]
            else:
                ep_forces = [all_force_data[fk][ep_start:ep_end] for fk in force_keys]
                episode_force_data_list[ep_idx] = np.stack(ep_forces, axis=1)

    logger.info(f"Processing {n_episodes} episodes...")

    all_metrics = {}

    for episode_idx in range(min(n_episodes, cfg.get('max_episodes', n_episodes))):
        if len(episode_data[episode_idx]) == 0:
            continue

        logger.info(f"Processing Episode {episode_idx} ({len(episode_data[episode_idx])} frames)")

        mse_10hz_data = []
        mse_50hz_data = []
        gt_chunk_10hz_data = []
        pred_chunk_10hz_data = []
        gt_chunk_50hz_data = []
        pred_chunk_50hz_data = []
        confidence_data = []
        selected_freq_data = []

        force_data = episode_force_data_list[episode_idx]
        has_force = force_data is not None and len(force_data) > 0

        with torch.no_grad():
            for frame_idx, batch in enumerate(tqdm.tqdm(episode_data[episode_idx], desc=f"Episode {episode_idx}")):
                obs_dict = {k: v for k, v in batch['obs'].items() if isinstance(v, torch.Tensor)}
                extended_obs_dict = None
                if 'extended_obs' in batch and batch['extended_obs'] is not None:
                    extended_obs_dict = {k: v for k, v in batch['extended_obs'].items() if isinstance(v, torch.Tensor)}

                result = policy.predict_action(
                    obs_dict=obs_dict,
                    dataset_obs_temporal_downsample_ratio=dataset_obs_temporal_downsample_ratio,
                    extended_obs_dict=extended_obs_dict,
                )

                # 10hz MSE: gt from batch['action'][:, 1:16], pred from result
                gt_10hz = batch['action'][:, 1:16]
                pred_10hz = result['action_pred_10hz'][:, 1:16]
                min_len_10 = min(gt_10hz.shape[1], pred_10hz.shape[1])
                gt_10hz = gt_10hz[:, :min_len_10]
                pred_10hz = pred_10hz[:, :min_len_10]

                mse_10 = compute_mse(gt_10hz, pred_10hz)
                mse_10hz_data.append(float(mse_10[0].mean().item()))
                gt_chunk_10hz_data.append(gt_10hz[0].detach().cpu().numpy())
                pred_chunk_10hz_data.append(pred_10hz[0].detach().cpu().numpy())

                # 50hz MSE: gt from batch['action_stack'][:, 13:], pred from result
                gt_50hz = batch['action_stack'][:, 13:]
                pred_50hz = result['action_pred_50hz'][:, 13:]
                min_len_50 = min(gt_50hz.shape[1], pred_50hz.shape[1])
                gt_50hz = gt_50hz[:, :min_len_50]
                pred_50hz = pred_50hz[:, :min_len_50]

                mse_50 = compute_mse(gt_50hz, pred_50hz)
                mse_50hz_data.append(float(mse_50[0].mean().item()))
                gt_chunk_50hz_data.append(gt_50hz[0].detach().cpu().numpy())
                pred_chunk_50hz_data.append(pred_50hz[0].detach().cpu().numpy())
                confidence_data.append(float(result.get("confidence", np.nan)))
                selected_freq_data.append(str(result.get("selected_freq", "unknown")))

        all_metrics[str(episode_idx)] = {
            'mse_10hz': [float(x) for x in mse_10hz_data],
            'mse_50hz': [float(x) for x in mse_50hz_data],
            'confidence': [
                None if np.isnan(x) else float(x) for x in confidence_data
            ],
            'selected_freq': selected_freq_data,
        }

        # Generate videos
        logger.info(f"Generating videos for episode {episode_idx}...")
        n_frames = len(episode_data[episode_idx])
        video_writers = {}

        for frame_idx in tqdm.tqdm(range(n_frames), desc=f"Episode {episode_idx} Video"):
            sample = episode_data[episode_idx][frame_idx]

            # Get observation image
            obs_img = np.zeros((240, 320, 3), dtype=np.uint8)
            if 'right_wrist_img' in sample['obs']:
                img_tensor = sample['obs']['right_wrist_img'][0, -1]
                obs_img = denormalize_image(img_tensor)
                obs_img = cv2.resize(obs_img, (640, 480))

            # Add text overlay
            overlay = obs_img.copy()
            cv2.rectangle(overlay, (5, 5), (obs_img.shape[1]-5, 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, obs_img, 0.4, 0, obs_img)
            cv2.putText(obs_img, f"Frame: {frame_idx+1}/{n_frames}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Force plot (shared)
            force_plot = None
            if has_force:
                force_arr = np.array(force_data[:frame_idx+1])
                if force_arr.ndim == 3:
                    force_arr = force_arr.mean(axis=1)
                fig_force = create_force_plot(force_arr, metric="norm", title="Wrench (norm)", max_frames=n_frames)
                force_plot = fig_to_image(fig_force)

            # --- Video 1: MSE 10hz ---
            fig_mse_10 = create_simple_plot(mse_10hz_data[:frame_idx+1], '10Hz Action MSE (PMF)', 'MSE', n_frames)
            mse_10_plot = fig_to_image(fig_mse_10)

            panels_10 = [mse_10_plot]
            if force_plot is not None:
                panels_10.append(force_plot)
            panels_10.append(obs_img)

            target_w = max(p.shape[1] for p in panels_10)
            resized_10 = []
            for p in panels_10:
                if p.shape[1] != target_w:
                    new_h = int(p.shape[0] * target_w / p.shape[1])
                    resized_10.append(cv2.resize(p, (target_w, new_h)))
                else:
                    resized_10.append(p)
            combined_10 = np.vstack(resized_10)

            vname_10 = 'mse_10hz'
            if vname_10 not in video_writers:
                h, w = combined_10.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vpath = output_dir / f"episode_{episode_idx:03d}_mse_10hz.mp4"
                video_writers[vname_10] = cv2.VideoWriter(str(vpath), fourcc, video_fps, (w, h))
                logger.info(f"Creating video: {vpath}")
            video_writers[vname_10].write(combined_10)

            # --- Video 2: MSE 50hz ---
            fig_mse_50 = create_simple_plot(mse_50hz_data[:frame_idx+1], '50Hz Action MSE (PMF)', 'MSE', n_frames)
            mse_50_plot = fig_to_image(fig_mse_50)

            panels_50 = [mse_50_plot]
            if force_plot is not None:
                panels_50.append(force_plot)
            panels_50.append(obs_img)

            target_w = max(p.shape[1] for p in panels_50)
            resized_50 = []
            for p in panels_50:
                if p.shape[1] != target_w:
                    new_h = int(p.shape[0] * target_w / p.shape[1])
                    resized_50.append(cv2.resize(p, (target_w, new_h)))
                else:
                    resized_50.append(p)
            combined_50 = np.vstack(resized_50)

            vname_50 = 'mse_50hz'
            if vname_50 not in video_writers:
                h, w = combined_50.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vpath = output_dir / f"episode_{episode_idx:03d}_mse_50hz.mp4"
                video_writers[vname_50] = cv2.VideoWriter(str(vpath), fourcc, video_fps, (w, h))
                logger.info(f"Creating video: {vpath}")
            video_writers[vname_50].write(combined_50)

            # --- Video 3: Action chunk 10hz ---
            fig_chunk_10 = create_action_chunk_plot(
                gt_chunk_10hz_data[frame_idx], pred_chunk_10hz_data[frame_idx],
                title=f'10Hz Action Chunk GT vs Pred - PMF (Frame {frame_idx+1})')
            chunk_10_plot = fig_to_image(fig_chunk_10)

            target_w_c10 = max(chunk_10_plot.shape[1], obs_img.shape[1])
            if chunk_10_plot.shape[1] != target_w_c10:
                new_h = int(chunk_10_plot.shape[0] * target_w_c10 / chunk_10_plot.shape[1])
                chunk_10_plot = cv2.resize(chunk_10_plot, (target_w_c10, new_h))
            obs_img_c10 = obs_img
            if obs_img.shape[1] != target_w_c10:
                new_h = int(obs_img.shape[0] * target_w_c10 / obs_img.shape[1])
                obs_img_c10 = cv2.resize(obs_img, (target_w_c10, new_h))
            combined_c10 = np.vstack([chunk_10_plot, obs_img_c10])

            vname_c10 = 'action_chunk_10hz'
            if vname_c10 not in video_writers:
                h, w = combined_c10.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vpath = output_dir / f"episode_{episode_idx:03d}_action_chunk_10hz.mp4"
                video_writers[vname_c10] = cv2.VideoWriter(str(vpath), fourcc, video_fps, (w, h))
                logger.info(f"Creating video: {vpath}")
            video_writers[vname_c10].write(combined_c10)

            # --- Video 4: Action chunk 50hz ---
            fig_chunk_50 = create_action_chunk_plot(
                gt_chunk_50hz_data[frame_idx], pred_chunk_50hz_data[frame_idx],
                title=f'50Hz Action Chunk GT vs Pred - PMF (Frame {frame_idx+1})')
            chunk_50_plot = fig_to_image(fig_chunk_50)

            target_w_c50 = max(chunk_50_plot.shape[1], obs_img.shape[1])
            if chunk_50_plot.shape[1] != target_w_c50:
                new_h = int(chunk_50_plot.shape[0] * target_w_c50 / chunk_50_plot.shape[1])
                chunk_50_plot = cv2.resize(chunk_50_plot, (target_w_c50, new_h))
            obs_img_c50 = obs_img
            if obs_img.shape[1] != target_w_c50:
                new_h = int(obs_img.shape[0] * target_w_c50 / obs_img.shape[1])
                obs_img_c50 = cv2.resize(obs_img, (target_w_c50, new_h))
            combined_c50 = np.vstack([chunk_50_plot, obs_img_c50])

            vname_c50 = 'action_chunk_50hz'
            if vname_c50 not in video_writers:
                h, w = combined_c50.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vpath = output_dir / f"episode_{episode_idx:03d}_action_chunk_50hz.mp4"
                video_writers[vname_c50] = cv2.VideoWriter(str(vpath), fourcc, video_fps, (w, h))
                logger.info(f"Creating video: {vpath}")
            video_writers[vname_c50].write(combined_c50)

            # --- Video 5: Confidence ---
            threshold = getattr(policy, "confidence_threshold_for_head_selection", None)
            fig_conf = create_simple_plot(
                confidence_data[:frame_idx+1],
                "Confidence",
                "confidence",
                n_frames,
                threshold_line=threshold,
            )
            confidence_plot = fig_to_image(fig_conf)

            panels_conf = [confidence_plot]
            if force_plot is not None:
                panels_conf.append(force_plot)
            else:
                fig_force_placeholder = create_simple_plot(
                    [], "Wrench (no data)", "Force", n_frames
                )
                panels_conf.append(fig_to_image(fig_force_placeholder))
            panels_conf.append(obs_img)
            target_w_conf = max(p.shape[1] for p in panels_conf)
            resized_conf = []
            for p in panels_conf:
                if p.shape[1] != target_w_conf:
                    new_h = int(p.shape[0] * target_w_conf / p.shape[1])
                    resized_conf.append(cv2.resize(p, (target_w_conf, new_h)))
                else:
                    resized_conf.append(p)
            combined_conf = np.vstack(resized_conf)

            vname_conf = 'confidence'
            if vname_conf not in video_writers:
                h, w = combined_conf.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vpath = output_dir / f"episode_{episode_idx:03d}_confidence.mp4"
                video_writers[vname_conf] = cv2.VideoWriter(str(vpath), fourcc, video_fps, (w, h))
                logger.info(f"Creating video: {vpath}")
            video_writers[vname_conf].write(combined_conf)

        for writer in video_writers.values():
            writer.release()
        logger.info(f"Episode {episode_idx} completed")

    # Save metrics
    if all_metrics:
        metrics_path = output_dir / "all_episodes_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

    logger.info("PMF confidence evaluation completed!")


if __name__ == "__main__":
    main()
