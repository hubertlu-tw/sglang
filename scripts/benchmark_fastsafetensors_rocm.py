#!/usr/bin/env python3
"""Benchmark fastsafetensors vs safetensors loading.

Example:
  python scripts/benchmark_fastsafetensors_rocm.py --model-path /data/models/minimax-m2.1
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import torch

from sglang.srt.model_loader.weight_utils import (
    fastsafetensors_weights_iterator,
    safetensors_weights_iterator,
)


BYTES_IN_GB = 1024 ** 3


def _iter_weight_files(
    model_path: str,
    shard_glob: str,
    max_shards: int | None,
) -> list[str]:
    pattern = os.path.join(model_path, shard_glob)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No safetensors files found in {model_path} with pattern {shard_glob}"
        )
    if max_shards is not None:
        files = files[:max_shards]
    return files


def _run_iterator(iterator, *, device_sync: bool) -> tuple[float, int, int]:
    total_bytes = 0
    total_tensors = 0
    start = time.perf_counter()

    for _, tensor in iterator:
        total_bytes += tensor.numel() * tensor.element_size()
        total_tensors += 1

    if device_sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed, total_tensors, total_bytes


def _maybe_set_device(device_id: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fastsafetensors vs safetensors loading speed."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local model directory containing *.safetensors shards.",
    )
    parser.add_argument(
        "--shard-glob",
        type=str,
        default="*.safetensors",
        help="Glob pattern to select safetensors shards.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Limit number of shards to load (useful for large models).",
    )
    parser.add_argument(
        "--disable-mmap",
        action="store_true",
        help="Disable mmap for safetensors baseline loader.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to run each loader.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device id to use.",
    )
    args = parser.parse_args()

    _maybe_set_device(args.device_id)
    weight_files = _iter_weight_files(
        args.model_path,
        shard_glob=args.shard_glob,
        max_shards=args.max_shards,
    )

    print(f"Found {len(weight_files)} safetensors shards")

    for i in range(args.repeat):
        print(f"\nRun {i + 1}/{args.repeat} - safetensors baseline")
        iterator = safetensors_weights_iterator(
            weight_files,
            disable_mmap=args.disable_mmap,
        )
        elapsed, total_tensors, total_bytes = _run_iterator(
            iterator, device_sync=False
        )
        print(
            "Baseline: "
            f"{total_tensors} tensors, {total_bytes / BYTES_IN_GB:.2f} GB, "
            f"{elapsed:.2f}s"
        )

        print(f"Run {i + 1}/{args.repeat} - fastsafetensors")
        iterator = fastsafetensors_weights_iterator(weight_files)
        elapsed, total_tensors, total_bytes = _run_iterator(
            iterator, device_sync=True
        )
        print(
            "Fastsafetensors: "
            f"{total_tensors} tensors, {total_bytes / BYTES_IN_GB:.2f} GB, "
            f"{elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()
