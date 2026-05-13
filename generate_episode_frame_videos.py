import argparse
from pathlib import Path

import numpy as np
import tqdm
import zarr


RIGHT_WRIST_ALIASES = ("right_wrist_img", "right_wrist_image")
EXTERNAL_ALIASES = ("external_img", "external_image")
WRENCH_ALIASES = (
    "right_robot_tcp_wrench",
    "left_robot_tcp_wrench",
    "tcp_wrench",
    "wrench",
)
cv2 = None


def require_cv2():
    global cv2
    if cv2 is not None:
        return cv2
    try:
        import cv2 as cv2_module
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required to generate videos. Install opencv-python or run in an environment "
            "where `import cv2` works."
        ) from exc
    cv2 = cv2_module
    return cv2


def resolve_zarr_path(dataset_path: str) -> Path:
    path = Path(dataset_path).expanduser()
    if path.name == "replay_buffer.zarr":
        return path
    return path / "replay_buffer.zarr"


def find_key(data_group, aliases, label):
    for key in aliases:
        if key in data_group:
            return key
    available = ", ".join(sorted(data_group.keys()))
    raise KeyError(
        f"Cannot find {label} image key. Tried {aliases}. "
        f"Available data keys: {available}"
    )


def find_wrench_key(data_group):
    for key in WRENCH_ALIASES:
        if key in data_group:
            return key
    wrench_keys = sorted(
        key for key in data_group.keys()
        if "wrench" in key.lower() and "stack" not in key.lower()
    )
    if wrench_keys:
        return wrench_keys[0]
    available = ", ".join(sorted(data_group.keys()))
    raise KeyError(
        "Cannot find wrench key. Use --wrench-key to specify it. "
        f"Available data keys: {available}"
    )


def image_to_bgr(image: np.ndarray) -> np.ndarray:
    cv2_module = require_cv2()
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")

    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.min() < 0:
            image = (image + 1.0) * 127.5
        elif image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[-1] == 1:
        return cv2_module.cvtColor(image, cv2_module.COLOR_GRAY2BGR)
    if image.shape[-1] == 3:
        return cv2_module.cvtColor(image, cv2_module.COLOR_RGB2BGR)

    raise ValueError(f"Expected 1 or 3 channels, got shape {image.shape}")


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    cv2_module = require_cv2()
    height, width = image.shape[:2]
    if height == target_height:
        return image
    target_width = int(round(width * target_height / height))
    return cv2_module.resize(image, (target_width, target_height), interpolation=cv2_module.INTER_AREA)


def draw_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    cv2_module = require_cv2()
    result = image.copy()
    cv2_module.rectangle(result, (0, 0), (result.shape[1], 34), (0, 0, 0), -1)
    cv2_module.putText(
        result,
        label,
        (10, 24),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2_module.LINE_AA,
    )
    return result


def draw_frame_overlay(frame: np.ndarray, episode_idx: int, frame_idx: int, n_frames: int) -> np.ndarray:
    cv2_module = require_cv2()
    result = frame.copy()
    overlay = result.copy()
    top = result.shape[0] - 57
    bottom = result.shape[0] - 10
    cv2_module.rectangle(overlay, (5, top), (result.shape[1] - 5, bottom), (0, 0, 0), -1)
    cv2_module.addWeighted(overlay, 0.6, result, 0.4, 0, result)
    cv2_module.putText(
        result,
        f"Episode: {episode_idx:03d}    Frame: {frame_idx + 1}/{n_frames}",
        (14, result.shape[0] - 26),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2_module.LINE_AA,
    )
    return result


def get_wrench_values(wrench_arr: np.ndarray, metric: str) -> np.ndarray:
    wrench_arr = np.asarray(wrench_arr)
    if wrench_arr.ndim == 3:
        wrench_arr = wrench_arr.mean(axis=1)
    if wrench_arr.ndim != 2:
        raise ValueError(f"Expected wrench array with shape (T, D), got {wrench_arr.shape}")
    if wrench_arr.shape[1] < 6:
        raise ValueError(f"Expected wrench array with at least 6 dims, got {wrench_arr.shape}")

    metric = metric.lower()
    if metric in ("norm", "force", "force_norm"):
        return np.linalg.norm(wrench_arr[:, :3], axis=1)
    if metric in ("torque", "torque_norm"):
        return np.linalg.norm(wrench_arr[:, 3:6], axis=1)
    dim_map = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}
    if metric in dim_map:
        return wrench_arr[:, dim_map[metric]]
    raise ValueError(f"Unsupported wrench metric: {metric}")


def make_wrench_plot(
    episode_wrench_values: np.ndarray,
    current_frame_idx: int,
    width: int,
    height: int,
    title: str,
) -> np.ndarray:
    cv2_module = require_cv2()
    values = np.asarray(episode_wrench_values, dtype=np.float32)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    margin_left = 72
    margin_right = 24
    margin_top = 40
    margin_bottom = 44
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    if len(values) == 0 or plot_right <= plot_left or plot_bottom <= plot_top:
        return canvas

    y_min = float(np.min(values))
    y_max = float(np.max(values))
    if np.isclose(y_min, y_max):
        pad = max(abs(y_max) * 0.1, 1.0)
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    cv2_module.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (245, 245, 245), -1)
    cv2_module.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (160, 160, 160), 1)

    for ratio in (0.25, 0.5, 0.75):
        y = int(round(plot_top + ratio * (plot_bottom - plot_top)))
        cv2_module.line(canvas, (plot_left, y), (plot_right, y), (220, 220, 220), 1)

    x_positions = np.linspace(plot_left, plot_right, num=len(values))
    y_positions = plot_bottom - (values - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
    points = np.stack([x_positions, y_positions], axis=1).round().astype(np.int32)

    if len(points) > 1:
        cv2_module.polylines(canvas, [points], False, (210, 210, 210), 2, cv2_module.LINE_AA)
        current_points = points[: current_frame_idx + 1]
        if len(current_points) > 1:
            cv2_module.polylines(canvas, [current_points], False, (0, 120, 255), 2, cv2_module.LINE_AA)

    current_frame_idx = min(current_frame_idx, len(values) - 1)
    current_point = tuple(points[current_frame_idx])
    cv2_module.circle(canvas, current_point, 5, (0, 0, 255), -1, cv2_module.LINE_AA)
    cv2_module.line(canvas, (current_point[0], plot_top), (current_point[0], plot_bottom), (180, 180, 180), 1)

    cv2_module.putText(
        canvas,
        title,
        (16, 26),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        canvas,
        f"{y_max:.2f}",
        (8, plot_top + 6),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.45,
        (70, 70, 70),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        canvas,
        f"{y_min:.2f}",
        (8, plot_bottom + 4),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.45,
        (70, 70, 70),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        canvas,
        f"current: {values[current_frame_idx]:.3f}",
        (plot_left + 12, height - 14),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2_module.LINE_AA,
    )
    return canvas


def make_dual_camera_frame(
    right_wrist_image: np.ndarray,
    external_image: np.ndarray,
    episode_wrench_values: np.ndarray,
    wrench_title: str,
    right_label: str,
    external_label: str,
    episode_idx: int,
    frame_idx: int,
    n_frames: int,
    height: int | None,
    plot_height: int,
) -> np.ndarray:
    right_bgr = image_to_bgr(right_wrist_image)
    external_bgr = image_to_bgr(external_image)

    if height is None:
        target_height = max(right_bgr.shape[0], external_bgr.shape[0])
    else:
        target_height = height

    right_bgr = resize_to_height(right_bgr, target_height)
    external_bgr = resize_to_height(external_bgr, target_height)

    right_bgr = draw_panel_label(right_bgr, right_label)
    external_bgr = draw_panel_label(external_bgr, external_label)
    image_row = np.hstack([right_bgr, external_bgr])
    wrench_plot = make_wrench_plot(
        episode_wrench_values=episode_wrench_values,
        current_frame_idx=frame_idx,
        width=image_row.shape[1],
        height=plot_height,
        title=wrench_title,
    )
    combined = np.vstack([wrench_plot, image_row])
    return draw_frame_overlay(combined, episode_idx, frame_idx, n_frames)


def parse_episode_limit(max_episodes: int | None, n_episodes: int) -> int:
    if max_episodes is None or max_episodes < 0:
        return n_episodes
    return min(max_episodes, n_episodes)


def main():
    parser = argparse.ArgumentParser(
        description="Generate one dual-camera frame video per episode from a replay_buffer.zarr dataset."
    )
    parser.add_argument("dataset_path", help="Dataset directory or path to replay_buffer.zarr.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <dataset_path>/episode_frame_videos.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS.")
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Panel height before horizontal concatenation. Use -1 to keep original max height.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes.")
    parser.add_argument(
        "--right-key",
        default=None,
        help="Override right wrist image key. Defaults to right_wrist_img/right_wrist_image.",
    )
    parser.add_argument(
        "--external-key",
        default=None,
        help="Override external image key. Defaults to external_img/external_image.",
    )
    parser.add_argument(
        "--wrench-key",
        default=None,
        help="Override wrench key. Defaults to right_robot_tcp_wrench or first non-stack *wrench* key.",
    )
    parser.add_argument(
        "--wrench-metric",
        default="norm",
        choices=("norm", "force", "force_norm", "torque", "torque_norm", "x", "y", "z", "rx", "ry", "rz"),
        help="Wrench value plotted on top of the images.",
    )
    parser.add_argument("--plot-height", type=int, default=240, help="Height of the wrench plot panel.")
    args = parser.parse_args()

    zarr_path = resolve_zarr_path(args.dataset_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Cannot find replay buffer zarr: {zarr_path}")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else zarr_path.parent / "episode_frame_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(zarr_path), mode="r")
    data_group = root["data"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if len(episode_ends) == 0:
        raise ValueError(f"No episodes found in {zarr_path}")

    right_key = args.right_key or find_key(data_group, RIGHT_WRIST_ALIASES, "right wrist")
    external_key = args.external_key or find_key(data_group, EXTERNAL_ALIASES, "external")
    wrench_key = args.wrench_key or find_wrench_key(data_group)
    if right_key not in data_group:
        raise KeyError(f"right image key does not exist: {right_key}")
    if external_key not in data_group:
        raise KeyError(f"external image key does not exist: {external_key}")
    if wrench_key not in data_group:
        raise KeyError(f"wrench key does not exist: {wrench_key}")

    right_images = data_group[right_key]
    external_images = data_group[external_key]
    wrench_data = data_group[wrench_key]
    if right_images.shape[0] != external_images.shape[0]:
        raise ValueError(
            f"Image arrays have different lengths: {right_key}={right_images.shape[0]}, "
            f"{external_key}={external_images.shape[0]}"
        )
    if wrench_data.shape[0] != right_images.shape[0]:
        raise ValueError(
            f"Wrench and image arrays have different lengths: {wrench_key}={wrench_data.shape[0]}, "
            f"{right_key}={right_images.shape[0]}"
        )

    n_episodes = parse_episode_limit(args.max_episodes, len(episode_ends))
    panel_height = None if args.height is None or args.height < 0 else args.height
    cv2_module = require_cv2()
    fourcc = cv2_module.VideoWriter_fourcc(*"mp4v")

    start = 0
    for episode_idx in tqdm.tqdm(range(n_episodes), desc="Episodes"):
        end = int(episode_ends[episode_idx])
        n_frames = end - start
        if n_frames <= 0:
            start = end
            continue
        episode_wrench_values = get_wrench_values(wrench_data[start:end], metric=args.wrench_metric)
        wrench_title = f"{wrench_key} ({args.wrench_metric})"

        video_path = output_dir / f"episode_{episode_idx:03d}_dual_camera.mp4"
        writer = None
        try:
            for local_frame_idx, global_frame_idx in enumerate(
                tqdm.tqdm(range(start, end), desc=f"Episode {episode_idx:03d}", leave=False)
            ):
                frame = make_dual_camera_frame(
                    right_wrist_image=right_images[global_frame_idx],
                    external_image=external_images[global_frame_idx],
                    episode_wrench_values=episode_wrench_values,
                    wrench_title=wrench_title,
                    right_label=right_key,
                    external_label=external_key,
                    episode_idx=episode_idx,
                    frame_idx=local_frame_idx,
                    n_frames=n_frames,
                    height=panel_height,
                    plot_height=args.plot_height,
                )

                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2_module.VideoWriter(str(video_path), fourcc, args.fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer: {video_path}")
                writer.write(frame)
        finally:
            if writer is not None:
                writer.release()

        start = end

    print(f"Saved {n_episodes} episode videos to {output_dir}")


if __name__ == "__main__":
    main()
