#!/usr/bin/env python3
"""Reproducible CUDA-event baseline for the frozen direct Ada W4A16 kernel.

Input construction, quantization, correctness references, extension loading,
and host/device transfers all happen before the timed regions.  CUDA events on
the current stream measure only device work issued by the selected operation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Callable

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cuda_primitives  # noqa: E402
from reference import (  # noqa: E402
    NVFP4Tensor,
    dequantize_nvfp4_reference,
    quantize_nvfp4_reference,
    w4a16_linear_reference,
)


DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "results"
    / "rtx4080-laptop-sm89"
    / "baseline_w4a16.json"
)
BASE_SEED = 44_016
CORRECTNESS_MAX_BF16_DISTANCE = 1


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    m: int
    n: int
    k: int
    groups: tuple[str, ...]


# This deliberately avoids a Cartesian product.  Overlapping canonical cases
# carry multiple group labels so every scaling series has a common anchor.
DIRECT_CASES = (
    BenchmarkCase(
        "canonical_q_m1_n3072_k3072",
        1,
        3072,
        3072,
        ("canonical_q", "n_scaling", "k_scaling", "m_scaling_q"),
    ),
    BenchmarkCase(
        "canonical_kv_m1_n768_k3072",
        1,
        768,
        3072,
        ("canonical_kv", "n_scaling", "m_scaling_kv"),
    ),
    BenchmarkCase("n_scaling_m1_n256_k3072", 1, 256, 3072, ("n_scaling",)),
    BenchmarkCase("n_scaling_m1_n1536_k3072", 1, 1536, 3072, ("n_scaling",)),
    BenchmarkCase("k_scaling_m1_n3072_k128", 1, 3072, 128, ("k_scaling",)),
    BenchmarkCase("k_scaling_m1_n3072_k512", 1, 3072, 512, ("k_scaling",)),
    BenchmarkCase("k_scaling_m1_n3072_k1024", 1, 3072, 1024, ("k_scaling",)),
    BenchmarkCase("m_scaling_q_m2_n3072_k3072", 2, 3072, 3072, ("m_scaling_q",)),
    BenchmarkCase("m_scaling_q_m4_n3072_k3072", 4, 3072, 3072, ("m_scaling_q",)),
    BenchmarkCase("m_scaling_q_m8_n3072_k3072", 8, 3072, 3072, ("m_scaling_q",)),
    BenchmarkCase("m_scaling_q_m16_n3072_k3072", 16, 3072, 3072, ("m_scaling_q",)),
    BenchmarkCase("m_scaling_kv_m2_n768_k3072", 2, 768, 3072, ("m_scaling_kv",)),
    BenchmarkCase("m_scaling_kv_m4_n768_k3072", 4, 768, 3072, ("m_scaling_kv",)),
    BenchmarkCase("m_scaling_kv_m8_n768_k3072", 8, 768, 3072, ("m_scaling_kv",)),
    BenchmarkCase("m_scaling_kv_m16_n768_k3072", 16, 768, 3072, ("m_scaling_kv",)),
)

DEQUANT_CASES = (
    ("dequant_canonical_q_n3072_k3072", 3072, 3072),
    ("dequant_canonical_kv_n768_k3072", 768, 3072),
)


def _run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_metadata() -> dict[str, object]:
    commit = _run_text(["git", "rev-parse", "HEAD"])
    status = _run_text(["git", "status", "--short"])
    return {
        "commit": commit,
        "worktree_dirty": None if status is None else bool(status),
    }


def _nvidia_smi_metadata() -> dict[str, object] | None:
    fields = (
        "name,driver_version,power.limit,power.default_limit,power.max_limit"
    )
    output = _run_text(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
    )
    if not output:
        return None
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    if len(values) != 5:
        return {"raw_query": output}

    def optional_float(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    return {
        "name": values[0],
        "driver_version": values[1],
        # WSL may expose the default and maximum while reporting the live
        # power.limit field as unavailable; preserve that distinction.
        "power_limit_w": optional_float(values[2]),
        "default_power_limit_w": optional_float(values[3]),
        "maximum_power_limit_w": optional_float(values[4]),
    }


def _tool_version(command: list[str]) -> str | None:
    output = _run_text(command)
    return output or None


def _environment() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "nvidia_smi": _nvidia_smi_metadata(),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "pytorch_cuda_build": torch.version.cuda,
            "nvcc": _tool_version(["nvcc", "--version"]),
            "nsight_compute": _tool_version(["ncu", "--version"]),
            "platform": platform.platform(),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "clock_power_policy": (
            "Uncontrolled consumer-laptop conditions. GPU clocks, power limit, "
            "driver settings, and host/WSL settings were not changed."
        ),
    }


def _seed_for_weight(n: int, k: int) -> int:
    return BASE_SEED + n * 131 + k * 17


def _seed_for_activation(n: int, k: int) -> int:
    return BASE_SEED + n * 23 + k * 7


def prepare_weight(n: int, k: int) -> NVFP4Tensor:
    """Create deterministic representative storage outside timed regions."""

    generator = torch.Generator(device="cpu").manual_seed(_seed_for_weight(n, k))
    source_cpu = torch.randn((n, k), generator=generator, dtype=torch.float32) * 0.5
    source_cuda = source_cpu.cuda()
    del source_cpu
    weight = quantize_nvfp4_reference(source_cuda)
    del source_cuda
    return weight


def prepare_activation(m: int, n: int, k: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(_seed_for_activation(n, k))
    # Repeat one row so the M series changes only grid size and repeated work,
    # not the activation distribution or numerical conditioning.
    activation_row = (
        torch.randn((1, k), generator=generator, dtype=torch.float32) * 0.75
    ).to(torch.bfloat16)
    x_cpu = activation_row.expand(m, k).contiguous()
    return x_cpu.cuda()


def _bf16_ordered_keys(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    magnitude = bits & 0x7FFF
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def _correctness_guard(
    case: BenchmarkCase,
    x: torch.Tensor,
    weight: NVFP4Tensor,
) -> dict[str, object]:
    expected = w4a16_linear_reference(x, weight)
    actual = cuda_primitives.cuda_w4a16_linear(x, weight)
    torch.cuda.synchronize()

    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise AssertionError(
            f"{case.name}: output metadata does not match the frozen oracle"
        )
    if not bool(torch.isfinite(actual).all().item()):
        raise AssertionError(f"{case.name}: direct output contains nonfinite values")

    error = (actual.float() - expected.float()).abs()
    adjacency = (
        _bf16_ordered_keys(actual) - _bf16_ordered_keys(expected)
    ).abs()
    maximum_distance = int(adjacency.max().item())
    metrics = {
        "oracle": "w4a16_linear_reference",
        "checked_shape": list(actual.shape),
        "maximum_absolute_error": float(error.max().item()),
        "mean_absolute_error": float(error.mean().item()),
        "exact_bf16_fraction": float((actual == expected).float().mean().item()),
        "maximum_bf16_adjacency_distance": maximum_distance,
        "allowed_maximum_bf16_adjacency_distance": (
            CORRECTNESS_MAX_BF16_DISTANCE
        ),
        "passed": maximum_distance <= CORRECTNESS_MAX_BF16_DISTANCE,
    }
    if not metrics["passed"]:
        raise AssertionError(f"{case.name}: correctness guard failed: {metrics}")
    return metrics


def _dequant_correctness_guard(
    name: str,
    weight: NVFP4Tensor,
) -> dict[str, object]:
    expected = dequantize_nvfp4_reference(weight)
    actual = cuda_primitives.cuda_dequantize_nvfp4(weight)
    torch.cuda.synchronize()
    exact = torch.equal(actual, expected)
    maximum_error = float((actual - expected).abs().max().item())
    if not exact:
        raise AssertionError(
            f"{name}: CUDA dequantization differs from the frozen reference "
            f"(max abs {maximum_error})"
        )
    return {
        "oracle": "dequantize_nvfp4_reference",
        "checked_shape": list(actual.shape),
        "bitwise_equal_fp32": exact,
        "maximum_absolute_error": maximum_error,
        "passed": exact,
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sample")
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _summarize(samples_us: list[float]) -> dict[str, float]:
    ordered = sorted(samples_us)
    result = {
        "median_us": statistics.median(ordered),
        "mean_us": statistics.fmean(ordered),
        "min_us": ordered[0],
        "p95_us": _percentile(ordered, 0.95),
        "standard_deviation_us": statistics.pstdev(ordered),
    }
    return {name: round(value, 3) for name, value in result.items()}


def _time_cuda_operation(
    operation: Callable[[], torch.Tensor],
    warmups: int,
    repetitions: int,
    *,
    before_sample: Callable[[], torch.Tensor] | None = None,
) -> tuple[dict[str, float], list[float]]:
    output: torch.Tensor | None = None
    precondition_output: torch.Tensor | None = None
    for _ in range(warmups):
        if before_sample is not None:
            precondition_output = before_sample()
        output = operation()
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, end in zip(starts, ends, strict=True):
        if before_sample is not None:
            precondition_output = before_sample()
        start.record(stream)
        output = operation()
        end.record(stream)

    ends[-1].synchronize()
    samples_us = [start.elapsed_time(end) * 1_000.0 for start, end in zip(starts, ends, strict=True)]
    if output is None:
        raise AssertionError("the timed operation did not execute")
    if before_sample is not None and precondition_output is None:
        raise AssertionError("the timing precondition did not execute")
    # Raw event samples retain millisecond-to-microsecond conversion at three
    # decimal places; human-facing reports should round to observed stability.
    rounded_samples = [round(sample, 3) for sample in samples_us]
    return _summarize(samples_us), rounded_samples


def _logical_storage(m: int, n: int, k: int) -> dict[str, int]:
    components = {
        "packed_e2m1_payload": n * k // 2,
        "ue4m3_block_scales": n * k // 16,
        "global_decode_scale": 4,
        "bf16_activations": m * k * 2,
        "bf16_output": m * n * 2,
    }
    return {**components, "total": sum(components.values())}


def _print_result(name: str, statistics_us: dict[str, float]) -> None:
    print(
        f"{name}: median={statistics_us['median_us']:.2f} us "
        f"mean={statistics_us['mean_us']:.2f} us "
        f"min={statistics_us['min_us']:.2f} us "
        f"p95={statistics_us['p95_us']:.2f} us",
        flush=True,
    )


def _selected_cases(names: list[str] | None) -> tuple[BenchmarkCase, ...]:
    if not names:
        return DIRECT_CASES
    by_name = {case.name: case for case in DIRECT_CASES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown case name(s): {', '.join(unknown)}")
    return tuple(by_name[name] for name in names)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--diagnostic-repetitions", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="run one named direct case; may be repeated (default: full matrix)",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="print direct case names and exit",
    )
    args = parser.parse_args()
    if args.warmups < 1:
        parser.error("--warmups must be at least 1")
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    if args.diagnostic_repetitions < 2:
        parser.error("--diagnostic-repetitions must be at least 2")
    return args


def _weight_with_uniform_scale_code(
    weight: NVFP4Tensor,
    scale_code: int,
) -> NVFP4Tensor:
    return NVFP4Tensor(
        packed_values=weight.packed_values,
        block_scales=torch.full_like(weight.block_scales, scale_code),
        global_decode_scale=weight.global_decode_scale,
        logical_shape=weight.logical_shape,
    )


def _scale_characteristics(weight: NVFP4Tensor) -> dict[str, object]:
    scales = weight.block_scales
    exponents = (scales.to(torch.int16) >> 3) & 0x0F
    exponent_values, exponent_counts = torch.unique(
        exponents, return_counts=True
    )
    total = scales.numel()
    return {
        "block_scale_count": total,
        "scale_code_minimum": int(scales.min().item()),
        "scale_code_maximum": int(scales.max().item()),
        "exponent_histogram": {
            str(int(value)): int(count)
            for value, count in zip(
                exponent_values.cpu(), exponent_counts.cpu(), strict=True
            )
        },
        "exponent_zero_fraction": float(
            (exponents == 0).to(torch.float32).mean().item()
        ),
    }


def main() -> None:
    args = _parse_args()
    if args.list_cases:
        for case in DIRECT_CASES:
            print(case.name)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the CUDA device does not support BF16")

    selected = _selected_cases(args.cases)
    weights: dict[tuple[int, int], NVFP4Tensor] = {}
    direct_results: list[dict[str, object]] = []
    dequant_results: list[dict[str, object]] = []
    decode_path_results: list[dict[str, object]] = []
    cache_condition_results: list[dict[str, object]] = []
    input_characteristics: dict[str, object] = {}

    print(
        f"device={torch.cuda.get_device_name(0)} capability="
        f"{torch.cuda.get_device_capability(0)} warmups={args.warmups} "
        f"repetitions={args.repetitions}",
        flush=True,
    )

    with torch.inference_mode():
        for case in selected:
            key = (case.n, case.k)
            if key not in weights:
                weights[key] = prepare_weight(case.n, case.k)
            weight = weights[key]
            x = prepare_activation(case.m, case.n, case.k)
            correctness = _correctness_guard(case, x, weight)
            statistics_us, samples_us = _time_cuda_operation(
                lambda x=x, weight=weight: cuda_primitives.cuda_w4a16_linear(
                    x, weight
                ),
                args.warmups,
                args.repetitions,
            )
            result = {
                "operation": "direct_packed_w4a16",
                "case": asdict(case),
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                **statistics_us,
                "samples_us": samples_us,
                "correctness": correctness,
                "logical_storage_bytes": _logical_storage(
                    case.m, case.n, case.k
                ),
                "notes": (
                    "Packed NVFP4 is decoded in the frozen direct kernel; "
                    "CUDA-event timing contains device execution only."
                ),
            }
            direct_results.append(result)
            _print_result(case.name, statistics_us)

        # Contextual dequantization is emitted for the full default matrix only.
        # A case-filtered run remains narrowly scoped to its requested launches.
        if not args.cases:
            for name, n, k in DEQUANT_CASES:
                key = (n, k)
                if key not in weights:
                    weights[key] = prepare_weight(n, k)
                weight = weights[key]
                correctness = _dequant_correctness_guard(name, weight)
                statistics_us, samples_us = _time_cuda_operation(
                    lambda weight=weight: cuda_primitives.cuda_dequantize_nvfp4(
                        weight
                    ),
                    args.warmups,
                    args.repetitions,
                )
                result = {
                    "operation": "cuda_nvfp4_dequantize_to_fp32",
                    "case": {"name": name, "m": None, "n": n, "k": k},
                    "warmups": args.warmups,
                    "repetitions": args.repetitions,
                    **statistics_us,
                    "samples_us": samples_us,
                    "correctness": correctness,
                    "materialized_output_bytes": n * k * 4,
                    "notes": (
                        "Context only: software decode materializes FP32 W_hat; "
                        "this is not a competing projection implementation."
                    ),
                }
                dequant_results.append(result)
                _print_result(name, statistics_us)

            canonical_key = (3072, 3072)
            canonical_weight = weights[canonical_key]
            canonical_x = prepare_activation(1, *canonical_key)
            input_characteristics = {
                "canonical_q_weight": _scale_characteristics(
                    canonical_weight
                ),
                "m_scaling_activation_policy": (
                    "each M case repeats one deterministic activation row"
                ),
            }
            for name, scale_code, scale_class in (
                ("uniform_ue4m3_normal_code_0x70", 0x70, "normal exponent"),
                ("uniform_ue4m3_exponent_zero_code_0x07", 0x07, "exponent zero"),
            ):
                diagnostic_weight = _weight_with_uniform_scale_code(
                    canonical_weight, scale_code
                )
                diagnostic_case = BenchmarkCase(
                    name,
                    1,
                    canonical_key[0],
                    canonical_key[1],
                    ("decode_path_diagnostic",),
                )
                correctness = _correctness_guard(
                    diagnostic_case, canonical_x, diagnostic_weight
                )
                statistics_us, samples_us = _time_cuda_operation(
                    lambda diagnostic_weight=diagnostic_weight: (
                        cuda_primitives.cuda_w4a16_linear(
                            canonical_x, diagnostic_weight
                        )
                    ),
                    args.warmups,
                    args.repetitions,
                )
                result = {
                    "operation": "direct_packed_w4a16_decode_path_diagnostic",
                    "case": asdict(diagnostic_case),
                    "scale_code": scale_code,
                    "scale_class": scale_class,
                    "warmups": args.warmups,
                    "repetitions": args.repetitions,
                    **statistics_us,
                    "samples_us": samples_us,
                    "correctness": correctness,
                    "notes": (
                        "Artificial uniform-scale A/B diagnostic. Shape, packed "
                        "payload, activation, global scale, and kernel are fixed; "
                        "only the UE4M3 exponent path differs."
                    ),
                }
                decode_path_results.append(result)
                _print_result(name, statistics_us)

            l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
            eviction_buffer_bytes = l2_bytes * 2
            eviction_buffer = torch.ones(
                eviction_buffer_bytes,
                dtype=torch.uint8,
                device="cuda",
            )
            for name, n, k in (
                ("cache_condition_canonical_q", 3072, 3072),
                ("cache_condition_canonical_kv", 768, 3072),
            ):
                weight = weights[(n, k)]
                x = prepare_activation(1, n, k)
                statistics_us, samples_us = _time_cuda_operation(
                    lambda x=x, weight=weight: (
                        cuda_primitives.cuda_w4a16_linear(x, weight)
                    ),
                    args.warmups,
                    args.diagnostic_repetitions,
                    before_sample=lambda: torch.sum(
                        eviction_buffer, dtype=torch.int64
                    ),
                )
                result = {
                    "operation": "direct_packed_w4a16_after_l2_eviction_read",
                    "case": {"name": name, "m": 1, "n": n, "k": k},
                    "warmups": args.warmups,
                    "repetitions": args.diagnostic_repetitions,
                    **statistics_us,
                    "samples_us": samples_us,
                    "correctness": (
                        "reuses the passed primary case guard for the same inputs"
                    ),
                    "device_l2_bytes": l2_bytes,
                    "eviction_buffer_bytes": eviction_buffer_bytes,
                    "precondition": (
                        "torch.sum reads a uint8 buffer of 2x reported L2 "
                        "capacity on the same stream before the start event"
                    ),
                    "notes": (
                        "Cache-sensitivity diagnostic, not a direct measurement "
                        "of DRAM traffic or bandwidth and not a cold-cache guarantee."
                    ),
                }
                cache_condition_results.append(result)
                _print_result(name, statistics_us)

    result_document = {
        "schema_version": 1,
        "milestone": "M4B",
        "baseline_commit": "ee6f7efeae04cd504dd879ac4d147b09e600778a",
        "environment": _environment(),
        "methodology": {
            "timer": "torch.cuda.Event",
            "stream": "current CUDA stream for events and operation",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "completion_synchronization": (
                "one synchronization after warmup and one on the final end "
                "event after all measured launches"
            ),
            "p95_definition": (
                "linear interpolation at rank (sample_count - 1) * 0.95"
            ),
            "excluded": [
                "extension loading",
                "compilation",
                "Python setup",
                "random generation",
                "quantization",
                "benchmark input allocation",
                "host-to-device copies",
                "correctness reference",
            ],
            "conditions": (
                "uncontrolled consumer-laptop clocks and power; no clock, "
                "power-limit, driver, Windows, or WSL changes"
            ),
        },
        "matrix": [asdict(case) for case in selected],
        "input_characteristics": input_characteristics,
        "direct_w4a16": direct_results,
        "cuda_dequantize_nvfp4": dequant_results,
        "diagnostics": {
            "ue4m3_decode_path": decode_path_results,
            "cache_condition": cache_condition_results,
        },
        "contextual_torch_matmul": {
            "measured": False,
            "reason": (
                "BF16/FP32 matmul has different precision and storage semantics; "
                "M4B keeps it out of the primary baseline."
            ),
        },
        "logical_bytes_note": (
            "Logical storage footprints are derived from tensor formats. They "
            "are not measured DRAM traffic or DRAM bandwidth."
        ),
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".bench.tmp")
    temporary_path.write_text(
        json.dumps(result_document, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(f"wrote={output_path}", flush=True)


if __name__ == "__main__":
    main()
