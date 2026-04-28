#!/usr/bin/env python3
"""Randomly split copied PKL files into 70/10 dataset directories."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def numeric_pkl_key(path: Path) -> tuple[int, str]:
    return (pkl_number(path), path.name)


def pkl_number(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        raise ValueError(f"PKL filename must be numeric, got: {path.name}") from None


def ten_file_stratified_split(files: list[Path], seed: int) -> tuple[list[Path], list[Path]]:
    """Choose 10 held-out files while covering every numeric tens-prefix group."""
    rng = random.Random(seed)
    blocks: dict[int, list[Path]] = {}

    for path in files:
        number = pkl_number(path)
        block = number // 10
        blocks.setdefault(block, []).append(path)

    test_files = []
    for block in sorted(blocks):
        test_files.append(rng.choice(blocks[block]))

    remaining = [path for path in files if path not in set(test_files)]
    extra_count = 10 - len(test_files)
    if extra_count < 0:
        raise ValueError(
            f"Cannot make a 10-file split because there are {len(test_files)} "
            "numeric tens-prefix groups to cover."
        )
    test_files.extend(rng.sample(remaining, extra_count))

    test_set = set(test_files)
    train_files = [path for path in files if path not in test_set]
    return sorted(train_files, key=numeric_pkl_key), sorted(test_files, key=numeric_pkl_key)


def block_label(path: Path) -> str:
    number = pkl_number(path)
    block = number // 10
    if block == 0:
        return "1-9"
    start = block * 10
    return f"{start}-{start + 9}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly split 80 PKL files from dual_switch_v5_round_1 into "
            "70 copied files in dual_switch_v5.1_round_1 and 10 copied files "
            "in dual_switch_v5.2_round_1."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dual_switch_v5_round_1"),
        help="Source directory containing the original 80 .pkl files.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Output directory for the 70-file split. Defaults to source sibling.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Output directory for the 10-file split. Defaults to source sibling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=24,
        help="Random seed for a reproducible split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting copied files in output directories.",
    )
    return parser.parse_args()


def copy_split(files: list[Path], output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {src.name for src in files}
    stale_files = [
        path for path in output_dir.glob("*.pkl") if path.name not in expected_names
    ]
    if stale_files and not overwrite:
        stale_list = ", ".join(path.name for path in sorted(stale_files, key=numeric_pkl_key))
        raise FileExistsError(
            f"{output_dir} contains stale .pkl files from a different split: {stale_list}. "
            "Re-run with --overwrite to sync this output directory."
        )
    for stale_file in stale_files:
        stale_file.unlink()

    for src in files:
        dst = output_dir / src.name
        if dst.exists() and not overwrite:
            raise FileExistsError(
                f"{dst} already exists. Re-run with --overwrite to replace copied files."
            )
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    train_dir = (
        args.train_dir.resolve()
        if args.train_dir is not None
        else source.parent / "dual_switch_v5.1_round_1"
    )
    test_dir = (
        args.test_dir.resolve()
        if args.test_dir is not None
        else source.parent / "dual_switch_v5.2_round_1"
    )

    if not source.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source}")

    files = sorted(source.glob("*.pkl"), key=numeric_pkl_key)
    if len(files) != 80:
        raise ValueError(f"Expected exactly 80 .pkl files in {source}, found {len(files)}.")

    train_files, test_files = ten_file_stratified_split(files, args.seed)

    copy_split(train_files, train_dir, args.overwrite)
    copy_split(test_files, test_dir, args.overwrite)

    print(f"Source kept unchanged: {source}")
    print(f"Copied {len(train_files)} files to: {train_dir}")
    print(f"Copied {len(test_files)} files to: {test_dir}")
    print("10-file split, covering every numeric tens-prefix group:")
    for path in test_files:
        print(f"  {path.name}  ({block_label(path)})")


if __name__ == "__main__":
    main()
