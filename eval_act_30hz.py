import json
import os
import pathlib

import cv2
import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from loguru import logger
from omegaconf import OmegaConf

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.policy.act_30hz_policy import ACT30HzPolicy

plt.switch_backend("Agg")
OmegaConf.register_new_resolver("eval", eval, replace=True)


def denormalize_image(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if img.min() < 0:
        img = (img + 1.0) * 127.5
    else:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def fig_to_image(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def create_curve_plot(data, title, ylabel, max_frames):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(data)), data)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(max_frames - 1, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_action_chunk_plot(gt_action, pred_action, title, max_dims=6):
    chunk_len = min(gt_action.shape[0], pred_action.shape[0])
    action_dim = min(gt_action.shape[1], pred_action.shape[1], max_dims)
    gt_action = gt_action[:chunk_len, :action_dim]
    pred_action = pred_action[:chunk_len, :action_dim]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)
    axes_flat = axes.flatten()
    x = np.arange(chunk_len)
    for dim_idx in range(action_dim):
        ax = axes_flat[dim_idx]
        ax.plot(x, gt_action[:, dim_idx], color="black", linestyle="--", linewidth=1.5, label="GT")
        ax.plot(x, pred_action[:, dim_idx], color="tab:orange", linewidth=1.5, label="Pred")
        ax.set_title(f"Action Dim {dim_idx}")
        ax.set_xlabel("Chunk Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[action_dim:]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def compute_mse(gt_action, pred_action, n_dims=6):
    min_len = min(gt_action.shape[1], pred_action.shape[1])
    dims = min(n_dims, gt_action.shape[-1], pred_action.shape[-1])
    gt_action = gt_action[:, :min_len, :dims]
    pred_action = pred_action[:, :min_len, :dims]
    return torch.mean((gt_action - pred_action) ** 2, dim=-1)


def resize_and_stack(panels):
    target_w = max(panel.shape[1] for panel in panels)
    resized = []
    for panel in panels:
        if panel.shape[1] == target_w:
            resized.append(panel)
        else:
            new_h = int(panel.shape[0] * target_w / panel.shape[1])
            resized.append(cv2.resize(panel, (target_w, new_h)))
    return np.vstack(resized)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("ImplicitRDP", "config")),
    config_name="train_act_30hz_workspace",
)
def main(cfg):
    logger.info("Starting ACT 30Hz evaluation...")

    ckpt_path = os.environ.get("CKPT_PATH")
    if not ckpt_path:
        raise ValueError("Set CKPT_PATH environment variable")

    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir:
        output_dir = pathlib.Path(output_dir)
    else:
        output_dir = pathlib.Path(ckpt_path).parent / "evaluation_act_30hz"
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer", "scaler"], strict=False)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if not isinstance(policy, ACT30HzPolicy):
        raise ValueError(f"Expected ACT30HzPolicy, got {type(policy)}")

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    policy.eval().to(device)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset if cfg.get("use_full_dataset", False) else dataset.get_validation_dataset()

    replay_buffer = val_dataset.replay_buffer
    episode_ends = replay_buffer.episode_ends[:]
    n_episodes = len(episode_ends)
    episode_data = [[] for _ in range(n_episodes)]
    sampler = val_dataset.sampler

    for data_idx in tqdm.tqdm(range(len(val_dataset)), desc="Loading data"):
        episode_idx = sampler.indices_to_episode_idx[data_idx]
        sample = val_dataset[data_idx]
        sample_batch = dict_apply(
            sample,
            lambda x: x.unsqueeze(0).to(device, non_blocking=True)
            if isinstance(x, torch.Tensor)
            else x,
        )
        if "obs" in sample and isinstance(sample["obs"], dict):
            sample_batch["obs"] = {
                k: v.unsqueeze(0).to(device, non_blocking=True)
                for k, v in sample["obs"].items()
                if isinstance(v, torch.Tensor)
            }
        if "extended_obs" in sample and isinstance(sample["extended_obs"], dict):
            sample_batch["extended_obs"] = {
                k: v.unsqueeze(0).to(device, non_blocking=True)
                for k, v in sample["extended_obs"].items()
                if isinstance(v, torch.Tensor)
            }
        episode_data[episode_idx].append(sample_batch)

    action_start = int(getattr(policy, "action_start_30hz", cfg.get("action_start_30hz", 7)))
    video_fps = int(cfg.get("video_fps", 1))
    max_episodes = int(cfg.get("max_episodes", n_episodes))

    all_metrics = {}
    logger.info(f"Processing {min(n_episodes, max_episodes)} episodes...")

    for episode_idx in range(min(n_episodes, max_episodes)):
        if len(episode_data[episode_idx]) == 0:
            continue

        if hasattr(policy, "reset"):
            policy.reset()

        full_mse_data = []
        suffix_mse_data = []
        gt_suffix_chunks = []
        pred_suffix_chunks = []

        with torch.no_grad():
            for batch in tqdm.tqdm(episode_data[episode_idx], desc=f"Episode {episode_idx}"):
                obs_dict = {k: v for k, v in batch["obs"].items() if isinstance(v, torch.Tensor)}
                extended_obs_dict = None
                if "extended_obs" in batch and batch["extended_obs"] is not None:
                    extended_obs_dict = {
                        k: v for k, v in batch["extended_obs"].items() if isinstance(v, torch.Tensor)
                    }

                result = policy.predict_action(obs_dict=obs_dict, extended_obs_dict=extended_obs_dict)
                gt_full = batch["action_stack"]
                pred_full = result["action_pred"]

                full_mse = compute_mse(gt_full, pred_full)
                full_mse_data.append(float(full_mse[0].mean().item()))

                gt_suffix = gt_full[:, action_start:]
                pred_suffix = result["action"]
                suffix_mse = compute_mse(gt_suffix, pred_suffix)
                suffix_mse_data.append(float(suffix_mse[0].mean().item()))

                min_len = min(gt_suffix.shape[1], pred_suffix.shape[1])
                gt_suffix_chunks.append(gt_suffix[0, :min_len].detach().cpu().numpy())
                pred_suffix_chunks.append(pred_suffix[0, :min_len].detach().cpu().numpy())

        all_metrics[str(episode_idx)] = {
            "mse_full_48": full_mse_data,
            "mse_suffix_from_action_start": suffix_mse_data,
            "action_start_30hz": action_start,
        }

        logger.info(f"Generating videos for episode {episode_idx}...")
        video_writers = {}
        n_frames = len(episode_data[episode_idx])

        for frame_idx in tqdm.tqdm(range(n_frames), desc=f"Episode {episode_idx} Video"):
            sample = episode_data[episode_idx][frame_idx]
            obs_img = np.zeros((480, 640, 3), dtype=np.uint8)
            if "right_wrist_img" in sample["obs"]:
                obs_img = denormalize_image(sample["obs"]["right_wrist_img"][0, -1])
                obs_img = cv2.resize(obs_img, (640, 480))

            overlay = obs_img.copy()
            cv2.rectangle(overlay, (5, 5), (obs_img.shape[1] - 5, 55), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, obs_img, 0.4, 0, obs_img)
            cv2.putText(
                obs_img,
                f"Frame: {frame_idx + 1}/{n_frames}  action_start={action_start}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            fig_full = create_curve_plot(
                full_mse_data[: frame_idx + 1],
                "Full 48-step Action MSE",
                "MSE",
                n_frames,
            )
            full_plot = fig_to_image(fig_full)
            full_panel = resize_and_stack([full_plot, obs_img])
            if "mse_full_48" not in video_writers:
                h, w = full_panel.shape[:2]
                vpath = output_dir / f"episode_{episode_idx:03d}_mse_full_48.mp4"
                video_writers["mse_full_48"] = cv2.VideoWriter(
                    str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h)
                )
                logger.info(f"Creating video: {vpath}")
            video_writers["mse_full_48"].write(full_panel)

            fig_suffix = create_curve_plot(
                suffix_mse_data[: frame_idx + 1],
                "Executable Suffix Action MSE",
                "MSE",
                n_frames,
            )
            suffix_plot = fig_to_image(fig_suffix)
            suffix_panel = resize_and_stack([suffix_plot, obs_img])
            if "mse_suffix" not in video_writers:
                h, w = suffix_panel.shape[:2]
                vpath = output_dir / f"episode_{episode_idx:03d}_mse_suffix.mp4"
                video_writers["mse_suffix"] = cv2.VideoWriter(
                    str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h)
                )
                logger.info(f"Creating video: {vpath}")
            video_writers["mse_suffix"].write(suffix_panel)

            fig_chunk = create_action_chunk_plot(
                gt_suffix_chunks[frame_idx],
                pred_suffix_chunks[frame_idx],
                title=f"30Hz Suffix GT vs Pred (Frame {frame_idx + 1})",
            )
            chunk_plot = fig_to_image(fig_chunk)
            chunk_panel = resize_and_stack([chunk_plot, obs_img])
            if "action_chunk_suffix" not in video_writers:
                h, w = chunk_panel.shape[:2]
                vpath = output_dir / f"episode_{episode_idx:03d}_action_chunk_suffix.mp4"
                video_writers["action_chunk_suffix"] = cv2.VideoWriter(
                    str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h)
                )
                logger.info(f"Creating video: {vpath}")
            video_writers["action_chunk_suffix"].write(chunk_panel)

        for writer in video_writers.values():
            writer.release()

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    logger.info("ACT 30Hz evaluation completed.")


if __name__ == "__main__":
    main()
