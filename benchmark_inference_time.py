"""Benchmark the 10 Hz and 30 Hz samplers of a multi-frequency CD checkpoint.

Set CKPT_PATH and DATASET_PATH, then run python benchmark_inference_time.py.
Only sampling is timed. Observation encoding and wrench conditioning are
computed once with the same preprocessing as policy.predict_action().
"""
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

from ImplicitRDP.policy.dp_multifreq_cd_30hz_confidence_policy import DPMultiFreqCD30HzConfidencePolicy
from ImplicitRDP.workspace.train_dp_multifreq_cd_30hz_confidence_workspace import (
    TrainDPMultiFreqCD30HzConfidenceWorkspace,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy(cfg, ckpt_path: str) -> DPMultiFreqCD30HzConfidencePolicy:
    payload = torch.load(ckpt_path, map_location="cpu", pickle_module=dill)
    workspace = TrainDPMultiFreqCD30HzConfidenceWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer", "scaler", "teacher_model"], strict=True)

    if cfg.training.use_ema and workspace.ema_model is not None:
        policy = workspace.ema_model
        print("  [INFO] Using EMA model")
    else:
        policy = workspace.model
        print("  [INFO] Using base model")

    if not isinstance(policy, DPMultiFreqCD30HzConfidencePolicy):
        raise TypeError(f"Expected DPMultiFreqCD30HzConfidencePolicy, got {type(policy)}")
    policy.set_training_stage(workspace.current_stage)
    return policy


def get_real_batch(dataset, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
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
    config_name="train_dp_multifreq_cd_30hz_confidence_workspace",
)
def main(cfg):
    ckpt_path = os.environ.get("CKPT_PATH")
    if not ckpt_path:
        raise ValueError("Set CKPT_PATH to point to a multi-frequency CD checkpoint")

    task = os.environ.get("TASK", "dual_switch")
    dataset_version = os.environ.get("DATASET_VERSION", "v6.1")
    round_id = os.environ.get("ROUND", "1")
    dataset_path = os.environ.get(
        "DATASET_PATH",
        f"data/{task}_{dataset_version}_round_{round_id}_zarr",
    )

    warmup_iters = int(os.environ.get("WARMUP_ITERS", "50"))
    benchmark_iters = int(os.environ.get("BENCHMARK_ITERS", "200"))
    device_str = os.environ.get("DEVICE", "cuda:0")
    use_amp = os.environ.get("USE_AMP", "0") == "1"

    if benchmark_iters <= 0 or warmup_iters < 0:
        raise ValueError("BENCHMARK_ITERS must be positive and WARMUP_ITERS must be nonnegative")
    if use_amp and torch.device(device_str).type != "cuda":
        raise ValueError("USE_AMP=1 requires a CUDA device")

    cfg.task.dataset_path = dataset_path
    cfg.task.dataset.val_ratio = 0.0

    device = torch.device(device_str)
    amp_dtype = (
        torch.bfloat16
        if use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("=" * 76)
    print("  Multi-frequency CD Sampler Latency Benchmark")
    print("=" * 76)
    print(f"  Checkpoint:     {ckpt_path}")
    print(f"  Dataset:        {dataset_path}")
    print(f"  Device:         {device}")
    print(f"  AMP:            {'ON (' + str(amp_dtype) + ')' if use_amp else 'OFF'}")
    print(f"  Warmup iters:   {warmup_iters}")
    print(f"  Benchmark iters:{benchmark_iters}")
    print("  Cases:          30Hz 1/2 steps, 10Hz 5/6 steps")
    print("=" * 76)

    print("\n[1/4] Loading dataset...")
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    print("\n[2/4] Loading policy...")
    policy = load_policy(cfg, ckpt_path)
    policy.eval().to(device)

    print("\n[3/4] Preparing one real sample...")
    batch = get_real_batch(dataset, device)
    obs_dict = batch["obs"]
    extended_obs_dict = batch.get("extended_obs", None)
    if extended_obs_dict is None:
        raise ValueError("DPMultiFreqCD30HzConfidencePolicy requires extended_obs")

    with torch.no_grad():
        with autocast(enabled=use_amp, dtype=amp_dtype):
            obs_tokens = policy.predict_obs_tokens(obs_dict)
            batch_size = obs_tokens.shape[0]
            obs_tokens = obs_tokens.reshape(batch_size, -1, policy.n_emb)
            temporal_cond_10hz = policy.get_temporal_cond(
                extended_obs_dict, head="10hz", allow_partial=True, post_process=True
            )
            temporal_cond_30hz = policy.get_temporal_cond(
                extended_obs_dict, head="30hz", allow_partial=True, post_process=True
            )

    timer = CUDATimer(device)
    cases = [
        ("dp_30hz_1step", "30hz", temporal_cond_30hz, 1),
        ("dp_30hz_2step", "30hz", temporal_cond_30hz, 2),
        ("dp_10hz_5step", "10hz", temporal_cond_10hz, 5),
        ("dp_10hz_6step", "10hz", temporal_cond_10hz, 6),
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
    result_path = output_dir / "multifreq_cd_inference_latency.json"

    result_json = {
        "config": {
            "ckpt_path": str(ckpt_path),
            "dataset_path": str(dataset_path),
            "device": device_str,
            "use_amp": use_amp,
            "warmup_iters": warmup_iters,
            "benchmark_iters": benchmark_iters,
            "horizon_10hz": policy.horizon_10hz,
            "horizon_30hz": policy.horizon_30hz,
            "action_dim": policy.action_dim,
            "training_stage": policy.training_stage,
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
