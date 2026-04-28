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
from ImplicitRDP.policy.fm_multifreq_confidence_policy import FMMultiFreqConfidencePolicy
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


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('ImplicitRDP', 'config')),
    config_name="train_fm_multifreq_confidence_workspace"
)
def main(cfg):
    logger.info("Starting FM MultiFreq Confidence policy evaluation...")

    ckpt_path = os.environ.get('CKPT_PATH')
    if not ckpt_path:
        raise ValueError("Set CKPT_PATH environment variable")

    output_dir = os.environ.get('OUTPUT_DIR')
    if output_dir:
        output_dir = pathlib.Path(output_dir)
    else:
        output_dir = pathlib.Path(ckpt_path).parent / "evaluation_fm_multifreq_confidence"
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    logger.info(f"Output directory: {output_dir}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer", "scaler"], strict=False)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if not isinstance(policy, FMMultiFreqConfidencePolicy):
        raise ValueError(f"Expected FMMultiFreqConfidencePolicy, got {type(policy)}")

    configured_inference_steps = cfg.get('num_inference_steps', None)
    if configured_inference_steps is not None and hasattr(policy, 'num_inference_steps'):
        original_inference_steps = getattr(policy, 'num_inference_steps', None)
        policy.num_inference_steps = int(configured_inference_steps)
        logger.info(f"Override policy.num_inference_steps: {original_inference_steps} -> {policy.num_inference_steps}")

    has_confidence = policy.confidence_head is not None
    logger.info(f"Policy type: {type(policy).__name__}")
    logger.info(f"Confidence head: {'Enabled' if has_confidence else 'Disabled'}")

    if not has_confidence:
        raise ValueError("FMMultiFreqConfidencePolicy has no confidence head - nothing to evaluate")

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

    # Get confidence threshold from policy
    confidence_threshold = getattr(policy, 'confidence_threshold', 2.5)
    logger.info(f"Confidence threshold: {confidence_threshold}")

    logger.info(f"Processing {n_episodes} episodes...")

    all_confidence_data = {}

    for episode_idx in range(min(n_episodes, cfg.get('max_episodes', n_episodes))):
        if len(episode_data[episode_idx]) == 0:
            continue

        logger.info(f"Processing Episode {episode_idx} ({len(episode_data[episode_idx])} frames)")

        if hasattr(policy, 'reset'):
            policy.reset()

        log_sigma_data = []
        force_data = episode_force_data_list[episode_idx]
        has_force = force_data is not None and len(force_data) > 0

        with torch.no_grad():
            for frame_idx, batch in enumerate(tqdm.tqdm(episode_data[episode_idx], desc=f"Episode {episode_idx}")):
                obs_dict = {k: v for k, v in batch['obs'].items() if isinstance(v, torch.Tensor)}

                # Compute confidence
                try:
                    obs_tokens = policy._encode_obs(obs_dict)
                    log_sigma = policy._compute_confidence_logits(obs_tokens)
                    if log_sigma is not None:
                        log_sigma_val = float(log_sigma.mean().item())
                    else:
                        log_sigma_val = float('nan')
                except Exception as e:
                    logger.warning(f"Failed to compute confidence: {e}")
                    log_sigma_val = float('nan')

                log_sigma_data.append(log_sigma_val)

        all_confidence_data[str(episode_idx)] = [
            float(x) if not math.isnan(x) else None for x in log_sigma_data
        ]

        # Generate confidence video: 3-panel (confidence, force, image)
        logger.info(f"Generating confidence video for episode {episode_idx}...")
        n_frames = len(episode_data[episode_idx])
        video_writer = None

        for frame_idx in tqdm.tqdm(range(n_frames), desc=f"Episode {episode_idx} Video"):
            sample = episode_data[episode_idx][frame_idx]

            # Observation image
            obs_img = np.zeros((240, 320, 3), dtype=np.uint8)
            if 'right_wrist_img' in sample['obs']:
                img_tensor = sample['obs']['right_wrist_img'][0, -1]
                obs_img = denormalize_image(img_tensor)
                obs_img = cv2.resize(obs_img, (640, 480))

            # Text overlay
            overlay = obs_img.copy()
            cv2.rectangle(overlay, (5, 5), (obs_img.shape[1]-5, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, obs_img, 0.4, 0, obs_img)
            cv2.putText(obs_img, f"Frame: {frame_idx+1}/{n_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            sigma_val = log_sigma_data[frame_idx]
            if not math.isnan(sigma_val):
                cv2.putText(obs_img, f"log sigma: {sigma_val:.4f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Confidence plot
            fig_sigma = create_simple_plot(
                log_sigma_data[:frame_idx+1], 'Confidence (log sigma)', 'log sigma',
                n_frames, threshold_line=confidence_threshold)
            confidence_plot = fig_to_image(fig_sigma)

            # Force plot
            if has_force:
                force_arr = np.array(force_data[:frame_idx+1])
                if force_arr.ndim == 3:
                    force_arr = force_arr.mean(axis=1)
                fig_force = create_force_plot(force_arr, metric="norm", title="Wrench (norm)", max_frames=n_frames)
                force_plot = fig_to_image(fig_force)
            else:
                # Placeholder
                fig_placeholder = create_simple_plot([], 'Wrench (no data)', 'Force', n_frames)
                force_plot = fig_to_image(fig_placeholder)

            # Stack: confidence -> force -> image
            panels = [confidence_plot, force_plot, obs_img]
            target_w = max(p.shape[1] for p in panels)
            resized = []
            for p in panels:
                if p.shape[1] != target_w:
                    new_h = int(p.shape[0] * target_w / p.shape[1])
                    resized.append(cv2.resize(p, (target_w, new_h)))
                else:
                    resized.append(p)
            combined = np.vstack(resized)

            if video_writer is None:
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vpath = output_dir / f"episode_{episode_idx:03d}_confidence.mp4"
                video_writer = cv2.VideoWriter(str(vpath), fourcc, video_fps, (w, h))
                logger.info(f"Creating video: {vpath}")

            video_writer.write(combined)

        if video_writer is not None:
            video_writer.release()
        logger.info(f"Episode {episode_idx} completed")

    # Save confidence data
    if all_confidence_data:
        confidence_path = output_dir / "all_episodes_confidence.json"
        with open(confidence_path, 'w') as f:
            json.dump(all_confidence_data, f, indent=2)
        logger.info(f"Saved confidence data to {confidence_path}")

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
