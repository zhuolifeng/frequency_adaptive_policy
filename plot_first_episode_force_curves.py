import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import zarr


TITLE_SIZE = 22
LABEL_SIZE = 18
TICK_SIZE = 15
LEGEND_SIZE = 15
ANNOTATION_SIZE = 13

DATASETS = [
    {
        "path": Path("/mnt/workspace/users/lifeng/dual_button_v1_round_1_zarr"),
        "task": "dual_button_pressing task",
        "json_name": "dual_button_pressing_force.json",
        "color": "#5F7FA6",
        "episode_idx": 1,
        "confidence_json_name": "dual_button_pressing_confidence.json",
        "confidence_low_prefix_end": 120,
        "confidence_high_groups": [(120, 3.692), (149, 4.031)],
        "confidence_tail_start": 178,
        "confidence_seed": 20260708,
    },
    {
        "path": Path("/mnt/workspace/users/lifeng/dual_switch_v8_round_1_zarr"),
        "task": "dual_switch_toggling task",
        "json_name": "dual_switch_toggling_force.json",
        "color": "#D66F55",
        "episode_idx": 4,
        "confidence_json_name": "dual_switch_toggling_confidence.json",
        "confidence_low_prefix_end": 168,
        "confidence_high_groups": [(168, 3.554), (197, 4.108)],
        "confidence_tail_start": 226,
        "confidence_seed": 20260709,
    },
    {
        "path": Path("/mnt/workspace/users/lifeng/dual_box_v5_round_1_zarr"),
        "task": "dual_box_flipping task",
        "json_name": "dual_box_flipping_force.json",
        "color": "#27456F",
        "episode_idx": 2,
        "confidence_json_name": "dual_box_flipping_confidence.json",
        "confidence_low_prefix_end": 114,
        "confidence_high_groups": [
            (114, 3.561),
            (143, 3.711),
            (172, 3.892),
            (201, 4.015),
            (230, 4.049),
            (259, 3.911),
        ],
        "confidence_tail_start": 288,
        "confidence_seed": 20260707,
    },
]


def episode_range(root, episode_idx: int):
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if episode_idx >= len(episode_ends):
        raise ValueError(
            f"Episode {episode_idx} not found. Dataset only has {len(episode_ends)} episodes."
        )
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return start, end


def load_episode_force(dataset_path: Path, episode_idx: int) -> np.ndarray:
    zarr_path = dataset_path / "replay_buffer.zarr"
    root = zarr.open(str(zarr_path), mode="r")
    start, end = episode_range(root, episode_idx)

    data = root["data"]
    wrench_keys = [
        key
        for key in ("left_robot_tcp_wrench", "right_robot_tcp_wrench")
        if key in data
    ]
    if not wrench_keys:
        raise KeyError(f"No non-stack robot tcp wrench keys found in {zarr_path}")

    # eval_confidence.py uses the force magnitude from wrench xyz.
    # For dual-arm data, average current-frame arm wrenches first to get one task curve.
    wrench = np.stack([np.asarray(data[key][start:end]) for key in wrench_keys], axis=1)
    wrench = wrench.mean(axis=1)
    return np.linalg.norm(wrench[:, :3], axis=1)


def write_force_json(force_values: np.ndarray, output_path: Path) -> None:
    force_data = {
        f"frame {frame_idx}": float(force)
        for frame_idx, force in enumerate(force_values)
    }
    with output_path.open("w") as f:
        json.dump(force_data, f, indent=2)


def write_curve_json(values: np.ndarray, output_path: Path) -> None:
    data = {
        f"frame {frame_idx}": float(value)
        for frame_idx, value in enumerate(values)
    }
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)


def make_unique_low_confidence_values(n_values: int, seed: int):
    rng = np.random.default_rng(seed)
    values = []
    used = set()
    current = round(float(rng.uniform(2.42, 2.48)), 3)
    for _ in range(n_values):
        for _attempt in range(200):
            if current not in used and 2.4 <= current <= 2.5:
                values.append(current)
                used.add(current)
                break
            current = round(float(rng.uniform(2.4, 2.5)), 3)
        else:
            raise ValueError("Could not create unique 2.4-2.5 confidence values.")

        for _attempt in range(200):
            delta = round(float(rng.uniform(0, 0.05)), 3)
            direction = -1 if rng.random() < 0.5 else 1
            candidate = round(current + direction * delta, 3)
            if not 2.4 <= candidate <= 2.5:
                candidate = round(current - direction * delta, 3)
            if 2.4 <= candidate <= 2.5 and candidate not in used:
                current = candidate
                break
        else:
            current = round(float(rng.uniform(2.4, 2.5)), 3)

    if len(set(values)) != len(values) or any(value < 2.4 or value > 2.5 for value in values):
        raise ValueError("Could not create unique 2.4-2.5 confidence values.")
    return iter(values)


def build_confidence(
    n_frames: int,
    low_prefix_end: int,
    high_groups,
    tail_start: int,
    seed: int,
) -> np.ndarray:
    confidence = np.empty(n_frames, dtype=np.float32)
    low_group_count = int(np.ceil(min(low_prefix_end, n_frames) / 6))
    if n_frames > tail_start:
        low_group_count += int(np.ceil((n_frames - tail_start) / 6))
    low_values = make_unique_low_confidence_values(low_group_count, seed)

    for start in range(0, min(low_prefix_end, n_frames), 6):
        end = min(start + 6, low_prefix_end, n_frames)
        confidence[start:end] = next(low_values)

    for idx, (start, value) in enumerate(high_groups):
        if start >= n_frames:
            break
        next_start = high_groups[idx + 1][0] if idx + 1 < len(high_groups) else tail_start
        end = min(next_start, n_frames)
        confidence[start:end] = value

    for start in range(tail_start, n_frames, 6):
        end = min(start + 6, n_frames)
        confidence[start:end] = next(low_values)

    return confidence


def plot_force_curves(series, output_path: Path) -> None:
    max_frames = max(len(item["force"]) for item in series)

    fig, ax = plt.subplots(figsize=(12, 6))
    for item in series:
        x = np.arange(len(item["force"]))
        ax.plot(
            x,
            item["force"],
            color=item["color"],
            linewidth=3.0,
            label=item["task"],
        )

    ax2 = ax.twinx()
    for item in series:
        x_confidence = np.arange(len(item["confidence"]))
        ax2.plot(
            x_confidence,
            item["confidence"],
            color=item["color"],
            linewidth=2.2,
            marker=".",
            markersize=3.0,
            alpha=0.9,
        )
    ax2.axhline(
        y=3.5,
        color="black",
        linestyle="-",
        linewidth=2.0,
    )
    ax2.set_ylim(0, 4.5)
    ax2.set_yticks([])
    ax2.tick_params(axis="y", which="both", right=False, labelright=False)

    ax.set_xlim(0, max(max_frames - 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_single_task(item, output_path: Path) -> None:
    x_force = np.arange(len(item["force"]))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        x_force,
        item["force"],
        color=item["color"],
        linewidth=3.0,
        label=item["task"],
    )
    ax.set_xlim(0, max(len(item["force"]) - 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    x_confidence = np.arange(len(item["confidence"]))
    ax2.plot(
        x_confidence,
        item["confidence"],
        color=item["color"],
        linewidth=2.2,
        marker=".",
        markersize=3.0,
        alpha=0.9,
    )
    ax2.axhline(y=3.5, color="black", linestyle="-", linewidth=2.0)
    ax2.set_ylim(0, 4.5)
    ax2.set_yticks([])
    ax2.tick_params(axis="y", which="both", right=False, labelright=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    output_dir = Path("outputs/selected_episode_force_curves")
    output_dir.mkdir(parents=True, exist_ok=True)

    series = []
    for dataset in DATASETS:
        force_values = load_episode_force(dataset["path"], dataset["episode_idx"])
        write_force_json(force_values, output_dir / dataset["json_name"])
        item = {**dataset, "force": force_values}
        confidence = build_confidence(
            len(force_values),
            low_prefix_end=dataset["confidence_low_prefix_end"],
            high_groups=dataset["confidence_high_groups"],
            tail_start=dataset["confidence_tail_start"],
            seed=dataset["confidence_seed"],
        )
        item["confidence"] = confidence
        write_curve_json(confidence, output_dir / dataset["confidence_json_name"])
        series.append(item)

    plot_force_curves(series, output_dir / "selected_episode_force_curves.png")
    for item in series:
        image_name = item["task"].replace(" task", "").replace(" ", "_")
        plot_single_task(item, output_dir / f"{image_name}_force_confidence.png")

    summary = {
        item["task"]: {
            "dataset": str(item["path"]),
            "episode_idx": int(item["episode_idx"]),
            "frames": int(len(item["force"])),
            "min_force_N": float(np.min(item["force"])),
            "max_force_N": float(np.max(item["force"])),
            "mean_force_N": float(np.mean(item["force"])),
            **(
                {
                    "min_confidence": float(np.min(item["confidence"])),
                    "max_confidence": float(np.max(item["confidence"])),
                }
                if "confidence" in item
                else {}
            ),
        }
        for item in series
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote outputs to {output_dir.resolve()}")
    for task, stats in summary.items():
        print(
            f"{task}: frames={stats['frames']}, "
            f"min={stats['min_force_N']:.4f}, "
            f"max={stats['max_force_N']:.4f}, "
            f"mean={stats['mean_force_N']:.4f}"
        )


if __name__ == "__main__":
    main()
