"""Exercise the 30 Hz training recipes on two synthetic replay episodes."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from ImplicitRDP.common.replay_buffer import ReplayBuffer
from ImplicitRDP.workspace.train_dp_30hz_consistency_workspace import TrainDP30HzConsistencyWorkspace


def run_training(tmp_path, recipe, overrides):
    buffer = ReplayBuffer.create_empty_numpy()
    for direction in (-1, 1):
        pose = np.zeros((4, 9), dtype=np.float32)
        pose[:, 0] = direction * np.arange(4) * 0.01
        pose[:, 3:9] = [1, 0, 0, 0, 1, 0]
        action = np.column_stack([pose, pose, np.ones(4, dtype=np.float32)])
        image = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        wrench = np.zeros((4, 6), dtype=np.float32)
        buffer.add_episode({
            "action": action,
            "action_stack": np.repeat(action[:, None], 3, axis=1),
            "external_img": image,
            "right_wrist_img": image,
            "right_robot_tcp_pose": pose,
            "right_robot_tcp_wrench": wrench,
            "right_robot_tcp_wrench_stack": np.repeat(wrench[:, None], 3, axis=1),
        })
    buffer.save_to_path(str(tmp_path / "replay_buffer.zarr"))

    # Exercise real data loading, normalization and optimization on one CPU batch per split.
    output = tmp_path / "run"
    subprocess.run([
        sys.executable, "train.py", f"--config-name={recipe}",
        "task=real_dual_switch_image_wrench_offline_10fps_30hz",
        f"task.dataset_path={tmp_path}", f"hydra.run.dir={output}",
        "task.image_shape=[3,32,32]", "n_emb=16", "policy.model.n_layer=1",
        "policy.obs_encoder.model_name=resnet18", "policy.obs_encoder.transforms=null",
        "policy.obs_encoder.share_rgb_model=true", "policy.rnn_obs_encoder_hidden_dim=16",
        "training.device=cpu", "checkpoint.topk.k=0",
        "logging.backend=none", "dataloader.batch_size=4", "val_dataloader.batch_size=4",
        "dataloader.num_workers=0", "val_dataloader.num_workers=0",
        "dataloader.persistent_workers=false", "val_dataloader.persistent_workers=false",
        "dataloader.pin_memory=false", "val_dataloader.pin_memory=false",
        *overrides,
    ], cwd=Path(__file__).resolve().parents[1], check=True, timeout=180, env={
        **os.environ, "ACCELERATE_USE_CPU": "true", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    return output


def test_imeanflow_training(tmp_path):
    output = run_training(tmp_path, "train_dp_30hz_imeanflow_workspace", [
        "training.num_epochs=1", "training.use_ema=false",
    ])
    metrics = json.loads((output / "logs.json.txt").read_text().splitlines()[-1])
    assert math.isfinite(metrics["train_loss"])
    assert math.isfinite(metrics["val_loss"])
    assert (output / "checkpoints" / "latest.ckpt").stat().st_size > 0


def test_consistency_training_and_checkpoint_stages(tmp_path):
    output = run_training(tmp_path, "train_dp_30hz_consistency_workspace", [
        "training.num_epochs=2", "training.stage1_epochs=1", "training.stage2_epochs=1",
        "policy.obs_encoder.downsample_ratio=16",
    ])
    epochs = [json.loads(line) for line in (output / "logs.json.txt").read_text().splitlines()]
    epochs = [metrics for metrics in epochs if "val_loss" in metrics]
    assert [epoch["training_stage"] for epoch in epochs] == [1, 2]
    for metrics in epochs:
        assert math.isfinite(metrics["train_loss"])
        assert math.isfinite(metrics["val_loss"])

    for filename, stage in [("stage1_final.ckpt", 1), ("latest.ckpt", 2)]:
        workspace = TrainDP30HzConsistencyWorkspace.create_from_checkpoint(output / "checkpoints" / filename)
        assert workspace.current_stage == stage
        assert workspace.model.training_stage == stage
        assert workspace.ema_model.training_stage == stage
