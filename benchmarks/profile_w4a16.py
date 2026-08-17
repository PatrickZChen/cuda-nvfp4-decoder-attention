#!/usr/bin/env python3
"""Issue a small deterministic launch sequence for Nsight Compute.

The companion shell helper skips the warmups and profiles exactly one launch
whose kernel name matches the frozen direct W4A16 kernel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cuda_primitives  # noqa: E402

# Importing these preparation helpers does not execute the benchmark driver.
from benchmark_w4a16 import prepare_activation, prepare_weight  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--launches", type=int, default=1)
    args = parser.parse_args()
    if args.m < 1 or args.n < 1:
        parser.error("M and N must be positive")
    if args.k < 16 or args.k % 16 != 0:
        parser.error("K must be at least 16 and divisible by 16")
    if args.warmups < 0 or args.launches < 1:
        parser.error("warmups must be nonnegative and launches must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("an available BF16-capable CUDA device is required")

    with torch.inference_mode():
        weight = prepare_weight(args.n, args.k)
        x = prepare_activation(args.m, args.n, args.k)
        torch.cuda.synchronize()

        output: torch.Tensor | None = None
        for _ in range(args.warmups):
            output = cuda_primitives.cuda_w4a16_linear(x, weight)
        torch.cuda.synchronize()

        for _ in range(args.launches):
            output = cuda_primitives.cuda_w4a16_linear(x, weight)
        torch.cuda.synchronize()

        if output is None:
            raise AssertionError("no profiled output was produced")
        checksum = float(output.float().sum().item())
        print(
            f"profile_shape=M{args.m}xN{args.n}xK{args.k} "
            f"warmups={args.warmups} launches={args.launches} "
            f"checksum={checksum:.9g}"
        )


if __name__ == "__main__":
    main()
