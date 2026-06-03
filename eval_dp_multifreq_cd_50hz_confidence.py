import json
import math
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
from ImplicitRDP.policy.dp_multifreq_cd_50hz_confidence_policy import (
    DPMultiFreqCD50HzConfidencePolicy,
)
from eval_dp_multifreq import (
    compute_mse,
    create_action_chunk_plot,
    create_force_plot,
    create_simple_plot,
    denormalize_image,
    fig_to_image,
)

plt.switch_backend("Agg")
OmegaConf.register_new_resolver("eval", eval, replace=True)


def create_confidence_plot(data, max_frames, threshold=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(data[:max_frames])), data[:max_frames])
    ax.set_title("Confidence (log sigma)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("log sigma")
    ax.grid(True)
    if threshold is not None:
        ax.axhline(y=threshold, color="r", linestyle="--", linewidth=1.5)
    plt.tight_layout()
    return fig


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("ImplicitRDP", "config")),
    config_name="train_dp_multifreq_cd_50hz_confidence_workspace",
)
def main(cfg):
    ckpt_path = os.environ.get("CKPT_PATH")
    if not ckpt_path:
        raise ValueError("Set CKPT_PATH environment variable")

    output_dir = os.environ.get("OUTPUT_DIR")
    output_dir = (
        pathlib.Path(output_dir)
        if output_dir
        else pathlib.Path(ckpt_path).parent / "evaluation_dp_multifreq_cd_50hz_confidence"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(
        payload,
        exclude_keys=["optimizer", "scaler"],
        strict=False,
    )

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if not isinstance(policy, DPMultiFreqCD50HzConfidencePolicy):
        raise ValueError(
            f"Expected DPMultiFreqCD50HzConfidencePolicy, got {type(policy)}"
        )

    configured_stage = cfg.get("training_stage", "auto")
    if configured_stage is None or str(configured_stage).lower() == "auto":
        checkpoint_stage = getattr(workspace, "current_stage", None)
        eval_stage = checkpoint_stage if checkpoint_stage is not None else policy.training_stage
    else:
        eval_stage = int(configured_stage)
    policy.set_training_stage(int(eval_stage))
    logger.info(f"Evaluation training_stage={policy.training_stage}")
    logger.info(f"Evaluation loss_type={getattr(policy, 'loss_type', 'unknown')}")

    num_inference_steps_10hz = int(cfg.get("num_inference_steps_10hz", 5))
    num_inference_steps_50hz = int(cfg.get("num_inference_steps_50hz", 1))
    logger.info(
        f"Inference steps: 10Hz={num_inference_steps_10hz}, "
        f"50Hz={num_inference_steps_50hz}"
    )

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
    dataset_obs_temporal_downsample_ratio = cfg.task.dataset.get(
        "obs_temporal_downsample_ratio", 1
    )
    use_confidence_curve = bool(
        getattr(policy, "use_confidence_based_head_selection", False)
    )
    if policy.training_stage == 1 and use_confidence_curve:
        logger.warning(
            "Stage 1 confidence head is untrained; set CONFIDENCE_SELECTION=False "
            "when comparing DP backbone behavior."
        )
    confidence_threshold = (
        getattr(policy, "confidence_threshold_for_head_selection", None)
        if use_confidence_curve
        else None
    )
    logger.info(f"Confidence curve video: {use_confidence_curve}")

    all_metrics = {}
    all_confidence = {}
    all_metrics["_config"] = {
        "loss_type": getattr(policy, "loss_type", None),
        "prediction_type": getattr(policy.noise_scheduler.config, "prediction_type", None),
        "training_stage": int(policy.training_stage),
        "num_inference_steps_10hz": num_inference_steps_10hz,
        "num_inference_steps_50hz": num_inference_steps_50hz,
    }
    max_episodes = min(n_episodes, cfg.get("max_episodes", n_episodes))
    for episode_idx in range(max_episodes):
        if len(episode_data[episode_idx]) == 0:
            continue

        logger.info(f"Processing Episode {episode_idx}")
        mse_10hz_data = []
        mse_50hz_data = []
        confidence_data = []
        gt_chunk_10hz_data = []
        pred_chunk_10hz_data = []
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
                    num_inference_steps_10hz=num_inference_steps_10hz,
                    num_inference_steps_50hz=num_inference_steps_50hz,
                )

                gt_10hz = batch["action"][:, 1:16]
                pred_10hz = result["action_pred_10hz"][:, 1:16]
                min_len_10 = min(gt_10hz.shape[1], pred_10hz.shape[1])
                gt_10hz = gt_10hz[:, :min_len_10]
                pred_10hz = pred_10hz[:, :min_len_10]
                mse_10 = compute_mse(gt_10hz, pred_10hz)
                mse_10hz_data.append(float(mse_10[0].mean().item()))
                gt_chunk_10hz_data.append(gt_10hz[0].detach().cpu().numpy())
                pred_chunk_10hz_data.append(pred_10hz[0].detach().cpu().numpy())

                gt_50hz = batch["action_stack"][:, 13:]
                pred_50hz = result["action_pred_50hz"][:, 13:]
                min_len_50 = min(gt_50hz.shape[1], pred_50hz.shape[1])
                gt_50hz = gt_50hz[:, :min_len_50]
                pred_50hz = pred_50hz[:, :min_len_50]
                mse_50 = compute_mse(gt_50hz, pred_50hz)
                mse_50hz_data.append(float(mse_50[0].mean().item()))
                gt_chunk_50hz_data.append(gt_50hz[0].detach().cpu().numpy())
                pred_chunk_50hz_data.append(pred_50hz[0].detach().cpu().numpy())

                if use_confidence_curve:
                    if "confidence" in result:
                        confidence_data.append(float(result["confidence"]))
                    else:
                        confidence_data.append(float("nan"))

        all_metrics[str(episode_idx)] = {
            "mse_10hz": [float(x) for x in mse_10hz_data],
            "mse_50hz": [float(x) for x in mse_50hz_data],
        }
        if use_confidence_curve:
            all_confidence[str(episode_idx)] = [
                None if math.isnan(x) else float(x) for x in confidence_data
            ]

        n_frames = len(episode_data[episode_idx])
        video_writers = {}
        for frame_idx in tqdm.tqdm(range(n_frames), desc=f"Episode {episode_idx} Video"):
            sample = episode_data[episode_idx][frame_idx]
            obs_img = np.zeros((240, 320, 3), dtype=np.uint8)
            if "right_wrist_img" in sample["obs"]:
                img_tensor = sample["obs"]["right_wrist_img"][0, -1]
                obs_img = cv2.resize(denormalize_image(img_tensor), (640, 480))

            overlay = obs_img.copy()
            cv2.rectangle(overlay, (5, 5), (obs_img.shape[1] - 5, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, obs_img, 0.4, 0, obs_img)
            cv2.putText(
                obs_img,
                f"Frame: {frame_idx + 1}/{n_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            conf_val = (
                confidence_data[frame_idx]
                if use_confidence_curve and frame_idx < len(confidence_data)
                else float("nan")
            )
            if use_confidence_curve and not math.isnan(conf_val):
                cv2.putText(
                    obs_img,
                    f"log sigma: {conf_val:.4f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            force_plot = None
            if has_force:
                force_arr = np.array(force_data[:frame_idx + 1])
                if force_arr.ndim == 3:
                    force_arr = force_arr.mean(axis=1)
                force_plot = fig_to_image(
                    create_force_plot(
                        force_arr,
                        metric="norm",
                        title="Wrench (norm)",
                        max_frames=n_frames,
                    )
                )

            videos = [
                (
                    "mse_10hz",
                    fig_to_image(
                        create_simple_plot(
                            mse_10hz_data[:frame_idx + 1],
                            "10Hz Action MSE",
                            "MSE",
                            n_frames,
                        )
                    ),
                ),
                (
                    "mse_50hz",
                    fig_to_image(
                        create_simple_plot(
                            mse_50hz_data[:frame_idx + 1],
                            "50Hz Action MSE",
                            "MSE",
                            n_frames,
                        )
                    ),
                ),
                (
                    "action_chunk_10hz",
                    fig_to_image(
                        create_action_chunk_plot(
                            gt_chunk_10hz_data[frame_idx],
                            pred_chunk_10hz_data[frame_idx],
                            f"10Hz Action Chunk GT vs Pred (Frame {frame_idx + 1})",
                        )
                    ),
                ),
                (
                    "action_chunk_50hz",
                    fig_to_image(
                        create_action_chunk_plot(
                            gt_chunk_50hz_data[frame_idx],
                            pred_chunk_50hz_data[frame_idx],
                            f"50Hz Action Chunk GT vs Pred (Frame {frame_idx + 1})",
                        )
                    ),
                ),
            ]
            if use_confidence_curve:
                videos.append(
                    (
                        "confidence",
                        fig_to_image(
                            create_confidence_plot(
                                confidence_data[:frame_idx + 1],
                                n_frames,
                                threshold=confidence_threshold,
                            )
                        ),
                    )
                )

            for vname, plot_img in videos:
                panels = [plot_img]
                if force_plot is not None and vname == "confidence":
                    panels.append(force_plot)
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
                if vname not in video_writers:
                    h, w = combined.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vpath = output_dir / f"episode_{episode_idx:03d}_{vname}.mp4"
                    video_writers[vname] = cv2.VideoWriter(
                        str(vpath), fourcc, video_fps, (w, h)
                    )
                    logger.info(f"Creating video: {vpath}")
                video_writers[vname].write(combined)

        for writer in video_writers.values():
            writer.release()

    if all_metrics:
        with open(output_dir / "all_episodes_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
    if all_confidence:
        with open(output_dir / "all_episodes_confidence.json", "w") as f:
            json.dump(all_confidence, f, indent=2)

    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
