"""
FM shortcut multi-frequency inference latency benchmark.

Measures the sampler-only latency for FMMultiFreqShortcutPolicy:
  1. 50Hz head, 1 shortcut step
  2. 50Hz head, 2 shortcut steps
  3. 10Hz head, 5 Euler steps
  4. 10Hz head, 6 Euler steps

The script precomputes obs_tokens and temporal conditions once, then times only
policy.conditional_sample(...) for the requested head/step count.

Usage:
    CKPT_PATH=/path/to/latest.ckpt DATASET_PATH=/path/to/zarr \
        DEVICE=cuda:0 python benchmark_inference_time.py

Optional environment variables:
    CKPT_PATH          required checkpoint path
    DATASET_PATH       override task dataset path
    TASK               default: dual_switch
    DATASET_VERSION    default: v6.1
    ROUND              default: 1
    DEVICE             default: cuda:0
    WARMUP_ITERS       default: 50
    BENCHMARK_ITERS    default: 200
    USE_AMP            default: 0
    OUTPUT_DIR         default: <checkpoint_parent>/benchmark_results
"""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import json
import os
import pathlib
import time
from typing import Callable, Dict, List

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import autocast

from ImplicitRDP.policy.fm_multifreq_shortcut_policy import FMMultiFreqShortcutPolicy
from ImplicitRDP.workspace.train_fm_multifreq_shortcut_workspace import (
    TrainFMMultiFreqShortcutWorkspace,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy(cfg, ckpt_path: str) -> FMMultiFreqShortcutPolicy:
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    workspace = TrainFMMultiFreqShortcutWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer", "scaler"], strict=False)

    if cfg.training.use_ema and workspace.ema_model is not None:
        policy = workspace.ema_model
        print("  [INFO] Using EMA model")
    else:
        policy = workspace.model
        print("  [INFO] Using base model")

    if not isinstance(policy, FMMultiFreqShortcutPolicy):
        raise TypeError(f"Expected FMMultiFreqShortcutPolicy, got {type(policy)}")
    return policy


def get_real_batch(cfg, dataset, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    val_dataset = dataset.get_validation_dataset()
    source_dataset = val_dataset if len(val_dataset) > 0 else dataset
    sample = source_dataset[0]

    batch = {}
    if "obs" in sample and isinstance(sample["obs"], dict):
        batch["obs"] = {
            k: v.unsqueeze(0).to(device, non_blocking=True)
            for k, v in sample["obs"].items()
            if isinstance(v, torch.Tensor)
        }
    if "extended_obs" in sample and isinstance(sample["extended_obs"], dict):
        batch["extended_obs"] = {
            k: v.unsqueeze(0).to(device, non_blocking=True)
            for k, v in sample["extended_obs"].items()
            if isinstance(v, torch.Tensor)
        }
    return batch


class CUDATimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda = device.type == "cuda"

    def measure(self, fn: Callable[[], None], warmup: int, repeat: int) -> List[float]:
        for _ in range(warmup):
            fn()
            if self.use_cuda:
                torch.cuda.synchronize(self.device)

        latencies = []
        for _ in range(repeat):
            if self.use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(self.device)
                start_event.record(torch.cuda.current_stream(self.device))
                fn()
                end_event.record(torch.cuda.current_stream(self.device))
                torch.cuda.synchronize(self.device)
                latencies.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                fn()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
        return latencies


def summarize(latencies: List[float]) -> Dict[str, float]:
    arr = np.array(latencies, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def print_stats(name: str, latencies: List[float]):
    row = summarize(latencies)
    print(f"\n{'=' * 64}")
    print(f"  {name}")
    print(f"{'=' * 64}")
    print(f"  samples: {len(latencies)}")
    print(f"  mean:    {row['mean_ms']:.3f} ms")
    print(f"  std:     {row['std_ms']:.3f} ms")
    print(f"  min:     {row['min_ms']:.3f} ms")
    print(f"  max:     {row['max_ms']:.3f} ms")
    print(f"  p50:     {row['p50_ms']:.3f} ms")
    print(f"  p95:     {row['p95_ms']:.3f} ms")
    print(f"  p99:     {row['p99_ms']:.3f} ms")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("ImplicitRDP", "config")),
    config_name="train_fm_multifreq_shortcut_workspace",
)
def main(cfg):
    ckpt_path = os.environ.get("CKPT_PATH")
    if not ckpt_path:
        raise ValueError("Set CKPT_PATH to point to an FM shortcut checkpoint")

    task = os.environ.get("TASK", "dual_switch")
    dataset_version = os.environ.get("DATASET_VERSION", "v6.1")
    round_id = os.environ.get("ROUND", "1")
    dataset_path = os.environ.get(
        "DATASET_PATH",
        f"/mnt/workspace/users/lifeng/{task}_{dataset_version}_round_{round_id}_zarr",
    )

    warmup_iters = int(os.environ.get("WARMUP_ITERS", "50"))
    benchmark_iters = int(os.environ.get("BENCHMARK_ITERS", "200"))
    device_str = os.environ.get("DEVICE", "cuda:0")
    use_amp = os.environ.get("USE_AMP", "0") == "1"

    cfg.task.dataset_path = dataset_path
    cfg.task.dataset.val_ratio = 0.0

    device = torch.device(device_str)
    amp_dtype = (
        torch.bfloat16
        if use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("=" * 76)
    print("  FM Shortcut Multi-Frequency Inference Latency Benchmark")
    print("=" * 76)
    print(f"  Checkpoint:     {ckpt_path}")
    print(f"  Dataset:        {dataset_path}")
    print(f"  Device:         {device}")
    print(f"  AMP:            {'ON (' + str(amp_dtype) + ')' if use_amp else 'OFF'}")
    print(f"  Warmup iters:   {warmup_iters}")
    print(f"  Benchmark iters:{benchmark_iters}")
    print("  Cases:          50Hz 1/2 steps, 10Hz 5/6 steps")
    print("=" * 76)

    print("\n[1/4] Loading dataset...")
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    print("\n[2/4] Loading policy...")
    policy = load_policy(cfg, ckpt_path)
    policy.eval().to(device)

    print("\n[3/4] Preparing one real sample...")
    batch = get_real_batch(cfg, dataset, device)
    obs_dict = batch["obs"]
    extended_obs_dict = batch.get("extended_obs", None)
    if extended_obs_dict is None:
        raise ValueError("FMMultiFreqShortcutPolicy requires extended_obs")

    with torch.no_grad():
        with autocast(enabled=use_amp, dtype=amp_dtype):
            obs_tokens = policy.predict_obs_tokens(obs_dict)
            batch_size = obs_tokens.shape[0]
            obs_tokens = obs_tokens.reshape(batch_size, -1, policy.n_emb)
            temporal_cond_10hz = policy.get_temporal_cond(extended_obs_dict, head="10hz")
            temporal_cond_50hz = policy.get_temporal_cond(extended_obs_dict, head="50hz")

    timer = CUDATimer(device)
    cases = [
        ("shortcut_50hz_1step", "50hz", temporal_cond_50hz, 1),
        ("shortcut_50hz_2step", "50hz", temporal_cond_50hz, 2),
        ("fm_10hz_5step", "10hz", temporal_cond_10hz, 5),
        ("fm_10hz_6step", "10hz", temporal_cond_10hz, 6),
    ]

    print("\n[4/4] Running benchmark...")
    all_results = {}
    for name, head, temporal_cond, steps in cases:
        print(f"\n>>> {name}: head={head}, steps={steps}")

        def bench_sample(head=head, temporal_cond=temporal_cond, steps=steps):
            with torch.no_grad():
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    policy.conditional_sample(
                        cond=obs_tokens,
                        temporal_cond=temporal_cond,
                        freq_mode=head,
                        num_inference_steps=steps,
                    )

        latencies = timer.measure(
            bench_sample, warmup=warmup_iters, repeat=benchmark_iters
        )
        all_results[name] = latencies
        print_stats(name, latencies)

    summary = {name: summarize(lats) for name, lats in all_results.items()}

    print("\n")
    print("=" * 86)
    print("  Latency Comparison")
    print("=" * 86)
    print(f"{'mode':<28} {'mean(ms)':>10} {'std(ms)':>10} {'p50(ms)':>10} {'p95(ms)':>10} {'p99(ms)':>10}")
    print("-" * 86)
    for name, row in summary.items():
        print(
            f"{name:<28} {row['mean_ms']:>10.3f} {row['std_ms']:>10.3f} "
            f"{row['p50_ms']:>10.3f} {row['p95_ms']:>10.3f} {row['p99_ms']:>10.3f}"
        )
    print("=" * 86)

    output_dir_env = os.environ.get("OUTPUT_DIR")
    output_dir = (
        pathlib.Path(output_dir_env)
        if output_dir_env
        else pathlib.Path(ckpt_path).parent / "benchmark_results"
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    result_path = output_dir / "fm_shortcut_inference_latency.json"

    result_json = {
        "config": {
            "ckpt_path": str(ckpt_path),
            "dataset_path": str(dataset_path),
            "device": device_str,
            "use_amp": use_amp,
            "warmup_iters": warmup_iters,
            "benchmark_iters": benchmark_iters,
            "horizon_10hz": policy.horizon_10hz,
            "horizon_50hz": policy.horizon_50hz,
            "action_dim": policy.action_dim,
            "shortcut_denoise_steps": policy.shortcut_denoise_steps,
            "shortcut_dt_flow": policy.shortcut_dt_flow,
        },
        "summary": summary,
        "latencies_ms": all_results,
    }

    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=2)

    print(f"\nSaved benchmark results to: {result_path}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
