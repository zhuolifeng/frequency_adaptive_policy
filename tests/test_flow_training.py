"""Exercise the released iMeanFlow recipe on two synthetic replay episodes."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from ImplicitRDP.common.replay_buffer import ReplayBuffer


def test_imeanflow_training(tmp_path):
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

    # Keep the data loading, normalization, JVP loss and training loop in the test.
    # Use a smaller backbone and one CPU batch per split.
    output = tmp_path / "run"
    subprocess.run([
        sys.executable, "train.py", "--config-name=train_dp_30hz_imeanflow_workspace",
        "task=real_dual_switch_image_wrench_offline_10fps_30hz",
        f"task.dataset_path={tmp_path}", f"hydra.run.dir={output}",
        "task.image_shape=[3,32,32]", "n_emb=16", "policy.model.n_layer=1",
        "policy.obs_encoder.model_name=resnet18", "policy.obs_encoder.transforms=null",
        "policy.obs_encoder.share_rgb_model=true", "policy.rnn_obs_encoder_hidden_dim=16",
        "training.num_epochs=1", "training.use_ema=false", "training.device=cpu",
        "logging.backend=none", "dataloader.batch_size=4", "val_dataloader.batch_size=4",
        "dataloader.num_workers=0", "val_dataloader.num_workers=0",
        "dataloader.persistent_workers=false", "val_dataloader.persistent_workers=false",
        "dataloader.pin_memory=false", "val_dataloader.pin_memory=false",
    ], cwd=Path(__file__).resolve().parents[1], check=True, timeout=180, env={
        **os.environ, "ACCELERATE_USE_CPU": "true", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    metrics = json.loads((output / "logs.json.txt").read_text().splitlines()[-1])
    assert math.isfinite(metrics["train_loss"])
    assert math.isfinite(metrics["val_loss"])
    assert (output / "checkpoints" / "latest.ckpt").stat().st_size > 0
