"""
推理时间基准测试脚本 (Inference Latency Benchmark)

分别测量三种推理模式的单步耗时 (ms)：
  1. CTM 50Hz 单步推理  (consistency_sample_50hz, 无 chaining)
  2. CTM 50Hz chaining 推理  (consistency_sample_50hz, 有 chaining)
  3. 10Hz 5步 Karras Heun ODE 推理  (conditional_sample_10hz, bins=5)

用法：
    CKPT_PATH=/path/to/ctm.ckpt python benchmark_inference_time.py \
        task.dataset.dataset_path=/path/to/zarr_folder

    可选环境变量：
        CKPT_PATH          (必需) checkpoint 路径
        WARMUP_ITERS        (默认: 50)  GPU 预热次数
        BENCHMARK_ITERS     (默认: 200) 正式测量次数
        DEVICE              (默认: cuda:0) 设备
        USE_AMP             (默认: 0)  是否使用混合精度推理 (0/1)
        KARRAS_10HZ_BINS    (默认: 5)  10Hz Karras ODE 步数
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib
import time
import copy
import json
from collections import defaultdict

import torch
import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
from torch.cuda.amp import autocast

from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ====================================================================
# 加载模型 (与 eval_dp_ctm_multifreq.py 保持一致)
# ====================================================================
def load_policy(cfg, ckpt_path):
    """从 checkpoint 加载 DP-CTM MultiFreq policy。"""
    from ImplicitRDP.workspace.train_dp_ctm_multifreq_workspace import (
        TrainDPCTMMultiFreqWorkspace,
    )

    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    workspace = TrainDPCTMMultiFreqWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer", "scaler"], strict=False)

    # 优先使用 EMA model
    if cfg.training.use_ema and workspace.ema_model is not None:
        policy = workspace.ema_model
        print("  [INFO] 使用 EMA model")
    else:
        policy = workspace.model
        print("  [INFO] 使用 base model")

    # 释放 frozen teacher 节省显存
    if hasattr(policy, "drop_teacher"):
        policy.drop_teacher()

    return policy


# ====================================================================
# 准备一个真实数据 batch (用于推理)
# ====================================================================
def get_real_batch(cfg, dataset, device):
    """从验证集取一个 batch 送到 device 上。"""
    val_dataset = dataset.get_validation_dataset()
    sample = val_dataset[0]

    batch = {}
    # obs
    if "obs" in sample and isinstance(sample["obs"], dict):
        batch["obs"] = {
            k: v.unsqueeze(0).to(device, non_blocking=True)
            for k, v in sample["obs"].items()
            if isinstance(v, torch.Tensor)
        }
    # extended_obs
    if "extended_obs" in sample and isinstance(sample["extended_obs"], dict):
        batch["extended_obs"] = {
            k: v.unsqueeze(0).to(device, non_blocking=True)
            for k, v in sample["extended_obs"].items()
            if isinstance(v, torch.Tensor)
        }

    return batch


# ====================================================================
# 计时工具
# ====================================================================
class CUDATimer:
    """使用 CUDA events 精确计时 (ms)。"""

    def __init__(self, device):
        self.device = device
        self.use_cuda = device.type == "cuda"

    def measure(self, fn, warmup=50, repeat=200):
        """
        预热 + 正式计时。

        返回: list of float (每次调用的耗时, 单位 ms)
        """
        # 预热
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
                latencies.append(start_event.elapsed_time(end_event))  # ms
            else:
                t0 = time.perf_counter()
                fn()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

        return latencies


def print_stats(name: str, latencies: list):
    """打印统计信息。"""
    arr = np.array(latencies)
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  样本数:      {len(arr)}")
    print(f"  平均 (mean):  {arr.mean():.3f} ms")
    print(f"  标准差 (std): {arr.std():.3f} ms")
    print(f"  最小 (min):   {arr.min():.3f} ms")
    print(f"  最大 (max):   {arr.max():.3f} ms")
    print(f"  中位数 (p50): {np.percentile(arr, 50):.3f} ms")
    print(f"  p95:          {np.percentile(arr, 95):.3f} ms")
    print(f"  p99:          {np.percentile(arr, 99):.3f} ms")
    print(f"{'=' * 60}")


# ====================================================================
# 主函数
# ====================================================================
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("ImplicitRDP", "config")),
    config_name="train_dp_ctm_multifreq_workspace",
)
def main(cfg):
    # ---- 读取环境变量 ----
    ckpt_path = os.environ.get("CKPT_PATH")
    if not ckpt_path:
        raise ValueError("请设置 CKPT_PATH 环境变量指向 checkpoint 路径")

    warmup_iters = int(os.environ.get("WARMUP_ITERS", "50"))
    benchmark_iters = int(os.environ.get("BENCHMARK_ITERS", "200"))
    device_str = os.environ.get("DEVICE", "cuda:0")
    use_amp = os.environ.get("USE_AMP", "0") == "1"
    karras_10hz_bins = int(os.environ.get("KARRAS_10HZ_BINS", "5"))

    device = torch.device(device_str)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    print("=" * 70)
    print("  DP-CTM Multi-Frequency 推理时间基准测试")
    print("=" * 70)
    print(f"  Checkpoint:       {ckpt_path}")
    print(f"  Device:           {device}")
    print(f"  AMP:              {'ON (' + str(amp_dtype) + ')' if use_amp else 'OFF'}")
    print(f"  预热次数:         {warmup_iters}")
    print(f"  测量次数:         {benchmark_iters}")
    print(f"  10Hz ODE 步数:    {karras_10hz_bins}")
    print("=" * 70)

    # ---- 加载数据集 & normalizer ----
    print("\n[1/4] 加载数据集...")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()

    # ---- 加载模型 ----
    print("\n[2/4] 加载模型...")
    policy = load_policy(cfg, ckpt_path)
    policy.set_normalizer(normalizer)
    policy.set_training_stage(2)  # CTM 模式
    policy.eval().to(device)

    # ---- 准备数据 ----
    print("\n[3/4] 准备数据...")
    batch = get_real_batch(cfg, dataset, device)
    obs_dict = batch["obs"]
    extended_obs_dict = batch.get("extended_obs", None)

    # 预计算 obs_tokens 和 temporal_cond (这部分是共享的，不计入单独头的时间)
    with torch.no_grad():
        obs_tokens = policy.predict_obs_tokens(obs_dict)
        B = obs_tokens.shape[0]
        obs_tokens = obs_tokens.reshape(B, -1, policy.n_emb)

        temporal_cond_10hz = policy.get_temporal_cond(extended_obs_dict, head="10hz")
        temporal_cond_50hz = policy.get_temporal_cond(extended_obs_dict, head="50hz")

    timer = CUDATimer(device)

    # ---- Benchmark 1: CTM 50Hz 单步 (无 chaining) ----
    print("\n[4/4] 开始计时测量...\n")
    print(">>> 测试 1: CTM 50Hz 单步推理 (无 chaining)")

    policy.disable_chaining()

    def bench_ctm_50hz_single():
        with torch.no_grad():
            with autocast(enabled=use_amp, dtype=amp_dtype):
                policy.consistency_sample_50hz(obs_tokens, temporal_cond_50hz)

    latencies_ctm_single = timer.measure(
        bench_ctm_50hz_single, warmup=warmup_iters, repeat=benchmark_iters
    )
    print_stats("CTM 50Hz 单步推理 (无 chaining)", latencies_ctm_single)

    # ---- Benchmark 2: CTM 50Hz chaining ----
    print("\n>>> 测试 2: CTM 50Hz chaining 推理")

    policy.enable_chaining()

    def bench_ctm_50hz_chain():
        with torch.no_grad():
            with autocast(enabled=use_amp, dtype=amp_dtype):
                policy.consistency_sample_50hz(obs_tokens, temporal_cond_50hz)

    latencies_ctm_chain = timer.measure(
        bench_ctm_50hz_chain, warmup=warmup_iters, repeat=benchmark_iters
    )
    print_stats("CTM 50Hz chaining 推理", latencies_ctm_chain)
    policy.disable_chaining()

    # ---- Benchmark 3: 10Hz 5步 Karras Heun ODE ----
    print(f"\n>>> 测试 3: 10Hz Karras Heun ODE ({karras_10hz_bins} 步)")

    # 临时修改 bins
    original_bins = policy.karras_scheduler.bins
    policy.karras_scheduler.bins = karras_10hz_bins

    def bench_10hz_ode():
        with torch.no_grad():
            with autocast(enabled=use_amp, dtype=amp_dtype):
                policy.conditional_sample_10hz(obs_tokens, temporal_cond_10hz)

    latencies_10hz = timer.measure(
        bench_10hz_ode, warmup=warmup_iters, repeat=benchmark_iters
    )
    print_stats(f"10Hz Karras Heun ODE ({karras_10hz_bins} 步)", latencies_10hz)

    # 恢复原始 bins
    policy.karras_scheduler.bins = original_bins

    # ---- Bonus: 完整 predict_action (两个头同时推理) ----
    print("\n>>> 测试 4 (额外): 完整 predict_action (10Hz + 50Hz CTM)")

    policy.disable_chaining()
    policy.karras_scheduler.bins = karras_10hz_bins

    def bench_predict_action():
        with torch.no_grad():
            with autocast(enabled=use_amp, dtype=amp_dtype):
                policy._predict_action(
                    obs_dict,
                    extended_obs_dict=extended_obs_dict,
                    use_edm_50hz=False,  # CTM 单步
                )

    latencies_full = timer.measure(
        bench_predict_action, warmup=warmup_iters, repeat=benchmark_iters
    )
    print_stats(
        f"完整 predict_action (10Hz {karras_10hz_bins}步 + 50Hz CTM 单步)",
        latencies_full,
    )

    # 恢复
    policy.karras_scheduler.bins = original_bins

    # ---- 汇总对比表 ----
    all_results = {
        "ctm_50hz_single": latencies_ctm_single,
        "ctm_50hz_chain": latencies_ctm_chain,
        f"karras_10hz_{karras_10hz_bins}step": latencies_10hz,
        f"full_predict_action": latencies_full,
    }

    print("\n\n")
    print("=" * 80)
    print("  汇总对比表 (Latency Comparison)")
    print("=" * 80)
    header = f"{'模式':<45} {'mean(ms)':>10} {'std(ms)':>10} {'p50(ms)':>10} {'p95(ms)':>10}"
    print(header)
    print("-" * 80)

    summary = {}
    for name, lats in all_results.items():
        arr = np.array(lats)
        row = {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
        }
        summary[name] = row
        print(
            f"  {name:<43} {row['mean_ms']:>10.3f} {row['std_ms']:>10.3f} "
            f"{row['p50_ms']:>10.3f} {row['p95_ms']:>10.3f}"
        )

    print("=" * 80)

    # ---- 保存结果到 JSON ----
    output_dir = pathlib.Path(ckpt_path).parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True, parents=True)
    result_path = output_dir / "inference_latency.json"

    result_json = {
        "config": {
            "ckpt_path": str(ckpt_path),
            "device": device_str,
            "use_amp": use_amp,
            "warmup_iters": warmup_iters,
            "benchmark_iters": benchmark_iters,
            "karras_10hz_bins": karras_10hz_bins,
            "horizon_10hz": policy.horizon_10hz,
            "horizon_50hz": policy.horizon_50hz,
            "action_dim": policy.action_dim,
            "chaining_times": str(policy.chaining_times),
        },
        "summary": summary,
    }

    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {result_path}")
    print("测试完成!")


if __name__ == "__main__":
    main()
