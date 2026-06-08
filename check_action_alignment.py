import argparse

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf


OmegaConf.register_new_resolver("eval", eval, replace=True)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    parser = argparse.ArgumentParser(
        description="Check alignment between high-frequency action_stack and low-frequency action."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train_fm_50hz_workspace",
        help="Hydra train config name under ImplicitRDP/config, without .yaml.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional override for task.dataset_path, e.g. data/switch_30hz_zarr.",
    )
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--action_stack_index", type=int, default=7)
    parser.add_argument("--action_index", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--max_values", type=int, default=19)
    args = parser.parse_args()

    overrides = []
    if args.dataset_path is not None:
        overrides.append(f"task.dataset_path={args.dataset_path}")

    with initialize(config_path="ImplicitRDP/config", version_base="1.3"):
        cfg = compose(config_name=args.config, overrides=overrides)

    print("config:", args.config)
    print("task:", cfg.task.name)
    print("dataset_path:", cfg.task.dataset_path)
    print("dataset horizon:", cfg.task.dataset.horizon)
    print("dataset n_obs_steps:", cfg.task.dataset.n_obs_steps)
    print("dataset pad_before:", cfg.task.dataset.pad_before)
    print("dataset pad_after:", cfg.task.dataset.pad_after)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if hasattr(dataset, "sampler") and hasattr(dataset.sampler, "indices"):
        print("sampler index tuple:", dataset.sampler.indices[args.sample_idx])
        print("sampler tuple format: buffer_start, buffer_end, sample_start, sample_end")
    sample = dataset[args.sample_idx]

    action = to_numpy(sample["action"])
    action_stack = to_numpy(sample["action_stack"])

    print("sample_idx:", args.sample_idx)
    print("action shape:", action.shape)
    print("action_stack shape:", action_stack.shape)

    if args.action_index >= action.shape[0]:
        raise IndexError(
            f"action_index {args.action_index} out of range for action length {action.shape[0]}"
        )
    if args.action_stack_index >= action_stack.shape[0]:
        raise IndexError(
            f"action_stack_index {args.action_stack_index} out of range for action_stack length {action_stack.shape[0]}"
        )

    a = action[args.action_index]
    hs = action_stack[args.action_stack_index]
    diff = hs - a
    max_abs_diff = float(np.max(np.abs(diff)))

    n = min(args.max_values, a.shape[-1])
    print(f"action[{args.action_index}][: {n}]:")
    print(a[:n])
    print(f"action_stack[{args.action_stack_index}][: {n}]:")
    print(hs[:n])
    print(f"diff[: {n}]:")
    print(diff[:n])
    print("max_abs_diff:", max_abs_diff)
    print("allclose:", bool(np.allclose(hs, a, atol=args.atol, rtol=0.0)))

    if hasattr(dataset, "sensor_stacks"):
        print("sensor_stacks:", OmegaConf.to_yaml(dataset.sensor_stacks) if not isinstance(dataset.sensor_stacks, dict) else dataset.sensor_stacks)


if __name__ == "__main__":
    main()
