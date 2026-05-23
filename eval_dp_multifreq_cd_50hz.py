import json
import os
import pathlib

import cv2
import dill
import hydra
import numpy as np
import torch
import tqdm
from loguru import logger
from omegaconf import OmegaConf

from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.policy.dp_multifreq_cd_50hz_policy import DPMultiFreqCD50HzPolicy
from eval_dp_multifreq import (
    compute_mse,
    create_action_chunk_plot,
    create_force_plot,
    create_simple_plot,
    denormalize_image,
    fig_to_image,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("ImplicitRDP", "config")),
    config_name="train_dp_multifreq_cd_50hz_workspace",
)
def main(cfg):
    logger.info("Starting DP MultiFreq 50Hz CD policy evaluation...")

    ckpt_path = os.environ.get("CKPT_PATH")
    if not ckpt_path:
        raise ValueError("Set CKPT_PATH environment variable")

    output_dir = os.environ.get("OUTPUT_DIR")
    output_dir = (
        pathlib.Path(output_dir)
        if output_dir
        else pathlib.Path(ckpt_path).parent / "evaluation_dp_multifreq_cd_50hz"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(
        payload,
        exclude_keys=["teacher_model", "optimizer", "scaler"],
        strict=False,
    )

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if not isinstance(policy, DPMultiFreqCD50HzPolicy):
        raise ValueError(f"Expected DPMultiFreqCD50HzPolicy, got {type(policy)}")
    policy.set_training_stage(2)

    configured_inference_steps = cfg.get("num_inference_steps", None)
    if configured_inference_steps is not None:
        policy.num_inference_steps = int(configured_inference_steps)

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

    episode_ranges = []
    start = 0
    for end in episode_ends:
        episode_ranges.append((start, int(end)))
        start = int(end)

    force_keys = sorted([k for k in replay_buffer.keys() if "wrench" in k.lower()])
    episode_force_data_list = [None] * n_episodes
    if len(force_keys) > 0:
        all_force_data = {fk: replay_buffer[fk][:] for fk in force_keys if fk in replay_buffer}
        for ep_idx, (ep_start, ep_end) in enumerate(episode_ranges):
            if len(force_keys) == 1:
                episode_force_data_list[ep_idx] = all_force_data[force_keys[0]][ep_start:ep_end]
            else:
                ep_forces = [all_force_data[fk][ep_start:ep_end] for fk in force_keys]
                episode_force_data_list[ep_idx] = np.stack(ep_forces, axis=1)

    video_fps = cfg.get("video_fps", 10)
    dataset_obs_temporal_downsample_ratio = cfg.task.dataset.get("obs_temporal_downsample_ratio", 1)
    all_metrics = {}
    max_episodes = min(n_episodes, cfg.get("max_episodes", n_episodes))

    for episode_idx in range(max_episodes):
        if len(episode_data[episode_idx]) == 0:
            continue

        logger.info(f"Processing Episode {episode_idx} ({len(episode_data[episode_idx])} frames)")
        mse_50hz_data = []
        gt_chunk_50hz_data = []
        pred_chunk_50hz_data = []
        force_data = episode_force_data_list[episode_idx]
        has_force = force_data is not None and len(force_data) > 0

        with torch.no_grad():
            for batch in tqdm.tqdm(episode_data[episode_idx], desc=f"Episode {episode_idx}"):
                obs_dict = {k: v for k, v in batch["obs"].items() if isinstance(v, torch.Tensor)}
                extended_obs_dict = {
                    k: v
                    for k, v in batch["extended_obs"].items()
                    if isinstance(v, torch.Tensor)
                }
                result = policy.predict_action(
                    obs_dict=obs_dict,
                    dataset_obs_temporal_downsample_ratio=dataset_obs_temporal_downsample_ratio,
                    extended_obs_dict=extended_obs_dict,
                )

                gt_50hz = batch["action_stack"][:, 13:]
                pred_50hz = result["action_pred_50hz"][:, 13:]
                min_len_50 = min(gt_50hz.shape[1], pred_50hz.shape[1])
                gt_50hz = gt_50hz[:, :min_len_50]
                pred_50hz = pred_50hz[:, :min_len_50]
                mse_50 = compute_mse(gt_50hz, pred_50hz)
                mse_50hz_data.append(float(mse_50[0].mean().item()))
                gt_chunk_50hz_data.append(gt_50hz[0].detach().cpu().numpy())
                pred_chunk_50hz_data.append(pred_50hz[0].detach().cpu().numpy())

        all_metrics[str(episode_idx)] = {"mse_50hz": [float(x) for x in mse_50hz_data]}

        video_writer = None
        for frame_idx in tqdm.tqdm(
            range(len(episode_data[episode_idx])),
            desc=f"Episode {episode_idx} Video",
        ):
            sample = episode_data[episode_idx][frame_idx]
            obs_img = np.zeros((240, 320, 3), dtype=np.uint8)
            if "right_wrist_img" in sample["obs"]:
                img_tensor = sample["obs"]["right_wrist_img"][0, -1]
                obs_img = cv2.resize(denormalize_image(img_tensor), (640, 480))

            panels = [
                fig_to_image(
                    create_simple_plot(
                        mse_50hz_data[:frame_idx + 1],
                        "50Hz Action MSE",
                        "MSE",
                        len(episode_data[episode_idx]),
                    )
                ),
                fig_to_image(
                    create_action_chunk_plot(
                        gt_chunk_50hz_data[frame_idx],
                        pred_chunk_50hz_data[frame_idx],
                        f"50Hz Action Chunk GT vs Pred (Frame {frame_idx + 1})",
                    )
                ),
            ]
            if has_force:
                force_arr = np.array(force_data[:frame_idx + 1])
                if force_arr.ndim == 3:
                    force_arr = force_arr.mean(axis=1)
                panels.append(
                    fig_to_image(
                        create_force_plot(
                            force_arr,
                            metric="norm",
                            title="Wrench (norm)",
                            max_frames=len(episode_data[episode_idx]),
                        )
                    )
                )
            panels.append(obs_img)

            target_w = max(p.shape[1] for p in panels)
            resized = []
            for panel in panels:
                if panel.shape[1] != target_w:
                    new_h = int(panel.shape[0] * target_w / panel.shape[1])
                    resized.append(cv2.resize(panel, (target_w, new_h)))
                else:
                    resized.append(panel)
            combined = np.vstack(resized)
            if video_writer is None:
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vpath = output_dir / f"episode_{episode_idx:03d}_50hz.mp4"
                video_writer = cv2.VideoWriter(str(vpath), fourcc, video_fps, (w, h))
                logger.info(f"Creating video: {vpath}")
            video_writer.write(combined)
        if video_writer is not None:
            video_writer.release()

    if all_metrics:
        metrics_path = output_dir / "all_episodes_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

    logger.info("50Hz CD evaluation completed")


if __name__ == "__main__":
    main()
