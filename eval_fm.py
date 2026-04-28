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

from ImplicitRDP.workspace.base_workspace import BaseWorkspace
from ImplicitRDP.policy.fm_policy import FlowMatchingTransformerImagePolicy
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.common.pytorch_util import dict_apply

plt.switch_backend('Agg')
OmegaConf.register_new_resolver("eval", eval, replace=True)


def denormalize_image(img_tensor):
    img = img_tensor.cpu().numpy()
    if img.shape[0] == 3:  # (C, H, W)
        img = img.transpose(1, 2, 0)  # -> (H, W, C)
    if img.min() < 0:
        img = (img + 1.0) * 127.5
    else:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def create_simple_plot(data, title, ylabel, max_frames=200):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(len(data[:max_frames])), data[:max_frames], linewidth=1.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Frame')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_force_plot(force_arr, max_frames=200):
    force_norm = np.linalg.norm(force_arr[:max_frames, :3], axis=1)
    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(len(force_norm))
    ax.plot(x, force_norm, 'b-', linewidth=1.5, alpha=0.8)
    ax.fill_between(x, force_norm, alpha=0.2)
    mean_val = float(np.mean(force_norm))
    ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.3f}')
    ax.set_title('Force (norm)', fontsize=12)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Force Norm')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def fig_to_image(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def create_action_chunk_plot(gt_action, pred_action, title, max_dims=6):
    """Plot GT vs Pred action chunk for each dimension in a 2x3 grid."""
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
        ax.plot(x, gt_action[:, dim_idx], color="black", linestyle="--", linewidth=1.5,
                marker="o", markersize=3, label="GT")
        ax.plot(x, pred_action[:, dim_idx], color="tab:orange", linewidth=1.5,
                marker=".", markersize=4, label="Pred")
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


def compute_mse(gt_action, pred_action):
    """Compute per-frame MSE between gt and pred action chunks (front 6 dims only)."""
    gt_action, pred_action = gt_action.cpu(), pred_action.cpu()
    # Align lengths
    min_steps = min(gt_action.shape[1], pred_action.shape[1])
    gt_action = gt_action[:, :min_steps, :6]
    pred_action = pred_action[:, :min_steps, :6]
    mse = torch.mean((gt_action - pred_action) ** 2, dim=-1)  # (B, T)
    return float(mse.mean().item())


def process_episode(policy, episode_data_list, episode_force_data, device, output_dir,
                    episode_idx, video_fps=10):
    logger.info(f"Processing Episode {episode_idx} ({len(episode_data_list)} frames)")

    if hasattr(policy, 'reset'):
        policy.reset()

    mse_data = []
    gt_chunk_data = []
    pred_chunk_data = []
    n_frames = len(episode_data_list)
    has_force = episode_force_data is not None and len(episode_force_data) > 0

    with torch.no_grad():
        for frame_idx, sample in enumerate(tqdm.tqdm(episode_data_list, desc=f"Episode {episode_idx}")):
            obs_dict = {k: v for k, v in sample['obs'].items() if isinstance(v, torch.Tensor)}
            gt_action = sample['action'][:, 1:16]
            extended_obs_dict = None
            if 'extended_obs' in sample and sample['extended_obs'] is not None:
                extended_obs_dict = {k: v for k, v in sample['extended_obs'].items()
                                     if isinstance(v, torch.Tensor)}

            pred_result = policy.predict_action(obs_dict=obs_dict, extended_obs_dict=extended_obs_dict)
            pred_action = pred_result['action']

            mse = compute_mse(gt_action, pred_action)
            mse_data.append(mse)

            # Store action chunks for action chunk video
            min_steps = min(gt_action.shape[1], pred_action.shape[1])
            gt_chunk_data.append(gt_action[0, :min_steps].detach().cpu().numpy())
            pred_chunk_data.append(pred_action[0, :min_steps].detach().cpu().numpy())

    # Generate videos
    logger.info(f"Generating videos for episode {episode_idx}...")

    video_writer_eval = None
    video_writer_chunk = None
    for frame_idx in tqdm.tqdm(range(n_frames), desc=f"Episode {episode_idx} Video"):
        # ===== Observation image (shared by both videos) =====
        sample = episode_data_list[frame_idx]
        obs_img = np.zeros((240, 320, 3), dtype=np.uint8)
        # Try to find an image key in obs
        for img_key in ['right_wrist_img', 'external_img']:
            if img_key in sample['obs']:
                img_tensor = sample['obs'][img_key][0, -1]  # last obs step
                obs_img = denormalize_image(img_tensor)
                obs_img = cv2.resize(obs_img, (640, 480))
                break

        # Add frame counter overlay
        overlay = obs_img.copy()
        cv2.rectangle(overlay, (5, 5), (250, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, obs_img, 0.4, 0, obs_img)
        cv2.putText(obs_img, f"Frame: {frame_idx + 1}/{n_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ===== Video 1: 3-panel eval video (MSE + Force + Image) =====
        # Upper panel: MSE curve
        fig_mse = create_simple_plot(mse_data[:frame_idx + 1], 'Action MSE Error', 'MSE', n_frames)
        mse_plot = fig_to_image(fig_mse)

        # Middle panel: wrench curve
        if has_force:
            force_arr = np.array(episode_force_data[:frame_idx + 1])
            if force_arr.ndim == 3:
                force_arr = force_arr.mean(axis=1)
            fig_force = create_force_plot(force_arr, n_frames)
        else:
            fig_force = create_simple_plot([], 'Force (N/A)', 'Force', n_frames)
        force_plot = fig_to_image(fig_force)

        # Resize all panels to same width
        target_w = max(mse_plot.shape[1], force_plot.shape[1], obs_img.shape[1])

        mse_resized = cv2.resize(mse_plot, (target_w, int(mse_plot.shape[0] * target_w / mse_plot.shape[1])))
        force_resized = cv2.resize(force_plot, (target_w, int(force_plot.shape[0] * target_w / force_plot.shape[1])))
        obs_resized = cv2.resize(obs_img, (target_w, int(obs_img.shape[0] * target_w / obs_img.shape[1])))

        # Stack vertically: MSE -> Force -> Image
        combined = np.vstack([mse_resized, force_resized, obs_resized])

        # Initialize eval video writer
        if video_writer_eval is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = output_dir / f"episode_{episode_idx:03d}_eval.mp4"
            video_writer_eval = cv2.VideoWriter(str(video_path), fourcc, video_fps, (w, h))
            logger.info(f"Creating video: {video_path}")

        video_writer_eval.write(combined)

        # ===== Video 2: Action chunk video (chunk plot + Image) =====
        fig_chunk = create_action_chunk_plot(
            gt_chunk_data[frame_idx], pred_chunk_data[frame_idx],
            title=f'Action Chunk GT vs Pred (Frame {frame_idx + 1})')
        chunk_plot = fig_to_image(fig_chunk)

        target_w_chunk = max(chunk_plot.shape[1], obs_img.shape[1])
        if chunk_plot.shape[1] != target_w_chunk:
            new_h = int(chunk_plot.shape[0] * target_w_chunk / chunk_plot.shape[1])
            chunk_plot = cv2.resize(chunk_plot, (target_w_chunk, new_h))
        obs_img_chunk = obs_img
        if obs_img.shape[1] != target_w_chunk:
            new_h = int(obs_img.shape[0] * target_w_chunk / obs_img.shape[1])
            obs_img_chunk = cv2.resize(obs_img, (target_w_chunk, new_h))
        combined_chunk = np.vstack([chunk_plot, obs_img_chunk])

        # Initialize action chunk video writer
        if video_writer_chunk is None:
            h, w = combined_chunk.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path_chunk = output_dir / f"episode_{episode_idx:03d}_action_chunk.mp4"
            video_writer_chunk = cv2.VideoWriter(str(video_path_chunk), fourcc, video_fps, (w, h))
            logger.info(f"Creating video: {video_path_chunk}")

        video_writer_chunk.write(combined_chunk)

    if video_writer_eval is not None:
        video_writer_eval.release()
    if video_writer_chunk is not None:
        video_writer_chunk.release()

    logger.info(f"Episode {episode_idx} completed. Mean MSE: {np.mean(mse_data):.6f}")
    return {'mse_data': mse_data}


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('ImplicitRDP', 'config')),
    config_name="train_fm_workspace"
)
def main(cfg):
    logger.info("Starting Flow Matching Transformer policy evaluation...")

    # Load checkpoint
    ckpt_path = os.environ.get('CKPT_PATH')
    if not ckpt_path:
        raise ValueError("Set CKPT_PATH environment variable")

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
    assert isinstance(policy, FlowMatchingTransformerImagePolicy), f"Unsupported policy type: {type(policy)}"

    logger.info(f"Policy type: {type(policy).__name__}")

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

        for ep_idx, (ep_start, ep_end) in enumerate(episode_ranges):
            if len(force_keys) == 1:
                episode_force_data_list[ep_idx] = all_force_data[force_keys[0]][ep_start:ep_end]
            else:
                ep_forces = [all_force_data[fk][ep_start:ep_end] for fk in force_keys]
                episode_force_data_list[ep_idx] = np.stack(ep_forces, axis=1)

    logger.info(f"Processing {n_episodes} episodes...")
    max_episodes = cfg.get('max_episodes', n_episodes)

    all_mse = []
    for episode_idx in range(min(n_episodes, max_episodes)):
        if len(episode_data[episode_idx]) == 0:
            continue
        result = process_episode(
            policy=policy,
            episode_data_list=episode_data[episode_idx],
            episode_force_data=episode_force_data_list[episode_idx],
            device=device,
            output_dir=output_dir,
            episode_idx=episode_idx,
            video_fps=video_fps
        )
        all_mse.extend(result['mse_data'])

    logger.info(f"Overall Mean MSE: {np.mean(all_mse):.6f}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
