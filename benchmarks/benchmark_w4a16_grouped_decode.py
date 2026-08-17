#!/usr/bin/env python3
"""M4C alternating-round A/B benchmark for grouped W4A16 decode.

Both operations consume the same deterministic tensors. Quantization,
correctness, environment inspection, and static binary inspection stay outside
the CUDA-event timing regions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
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
    quantize_nvfp4_reference,
    w4a16_linear_reference,
)


DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "results"
    / "rtx4080-laptop-sm89"
    / "m4c_grouped_decode.json"
)
LIBRARY_PATH = REPOSITORY_ROOT / "build-cuda" / "cuda_primitives.so"
FROZEN_BASELINE_PATH = REPOSITORY_ROOT / "src" / "w4a16.cu"
FROZEN_M4B_RESULT_PATH = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "results"
    / "rtx4080-laptop-sm89"
    / "baseline_w4a16.json"
)
BASELINE_COMMIT = "ee6f7efeae04cd504dd879ac4d147b09e600778a"
M4B_COMMIT = "511b1c5"
BASE_SEED = 54_019
CORRECTNESS_MAX_BF16_DISTANCE = 1
ROUND_ORDERS = (
    ("baseline", "candidate"),
    ("candidate", "baseline"),
    ("baseline", "candidate"),
    ("candidate", "baseline"),
    ("baseline", "candidate"),
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    m: int
    n: int
    k: int
    groups: tuple[str, ...]


CASES = (
    BenchmarkCase(
        "canonical_q_m1_n3072_k3072",
        1,
        3072,
        3072,
        ("canonical_q", "m_scaling_q"),
    ),
    BenchmarkCase(
        "canonical_kv_m1_n768_k3072",
        1,
        768,
        3072,
        ("canonical_kv",),
    ),
    BenchmarkCase(
        "m_scaling_q_m2_n3072_k3072",
        2,
        3072,
        3072,
        ("m_scaling_q",),
    ),
    BenchmarkCase(
        "m_scaling_q_m4_n3072_k3072",
        4,
        3072,
        3072,
        ("m_scaling_q",),
    ),
    BenchmarkCase(
        "m_scaling_q_m8_n3072_k3072",
        8,
        3072,
        3072,
        ("m_scaling_q",),
    ),
    BenchmarkCase(
        "k_scaling_m1_n3072_k128",
        1,
        3072,
        128,
        ("k_scaling",),
    ),
    BenchmarkCase(
        "k_scaling_m1_n3072_k512",
        1,
        3072,
        512,
        ("k_scaling",),
    ),
    BenchmarkCase(
        "k_scaling_m1_n3072_k1024",
        1,
        3072,
        1024,
        ("k_scaling",),
    ),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict[str, object]:
    status = _run_text(["git", "status", "--short"])
    return {
        "head": _run_text(["git", "rev-parse", "HEAD"]),
        "baseline_commit": _run_text(["git", "rev-parse", BASELINE_COMMIT]),
        "m4b_commit": _run_text(["git", "rev-parse", M4B_COMMIT]),
        "worktree_dirty": None if status is None else bool(status),
        "worktree_status_short": [] if not status else status.splitlines(),
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
        "frozen_artifact_hashes": {
            "src/w4a16.cu": _sha256(FROZEN_BASELINE_PATH),
            "baseline_w4a16.json": _sha256(FROZEN_M4B_RESULT_PATH),
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "max_threads_per_multiprocessor": (
                properties.max_threads_per_multi_processor
            ),
            "registers_per_multiprocessor": properties.regs_per_multiprocessor,
            "warp_size": properties.warp_size,
            "l2_cache_bytes": properties.L2_cache_size,
            "nvidia_smi": _nvidia_smi_metadata(),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "pytorch_cuda_build": torch.version.cuda,
            "nvcc": _tool_version(["nvcc", "--version"]),
            "cuobjdump": _tool_version(["cuobjdump", "--version"]),
            "platform": platform.platform(),
            "extension_build": "Release, SM89, no fast-math or Debug rebuild",
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "clock_power_policy": (
            "Uncontrolled consumer-laptop conditions. GPU clocks, power limit, "
            "driver settings, and host/WSL settings were not changed."
        ),
    }


def _candidate_binary_evidence() -> dict[str, object]:
    resource_text = _run_text(
        ["cuobjdump", "--dump-resource-usage", str(LIBRARY_PATH)]
    )
    sass_text = _run_text(["cuobjdump", "--dump-sass", str(LIBRARY_PATH)])
    marker = "w4a16_linear_grouped_decode_kernel"
    if resource_text is None or sass_text is None:
        return {
            "available": False,
            "reason": "cuobjdump resource or SASS extraction failed",
        }

    resource_section = next(
        (
            section
            for section in resource_text.split(" Function ")[1:]
            if marker in section
        ),
        None,
    )
    sass_section = next(
        (
            section
            for section in sass_text.split("\t\tFunction : ")[1:]
            if marker in section
        ),
        None,
    )
    if resource_section is None or sass_section is None:
        return {
            "available": False,
            "reason": "candidate kernel was not found in cuobjdump output",
        }

    resource_match = re.search(
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        resource_section,
    )
    if resource_match is None:
        return {
            "available": False,
            "reason": "candidate resource fields could not be parsed",
        }
    registers, stack, shared, local = map(int, resource_match.groups())

    instructions = [
        (match.group(1), match.group(2), match.group(2).split(".")[0])
        for match in re.finditer(
            r"/\*([0-9a-f]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)",
            sass_section,
        )
    ]
    opcode_bases = (
        "LDG",
        "STG",
        "SHFL",
        "FMUL",
        "FADD",
        "FFMA",
        "MUFU",
        "IMAD",
        "IADD3",
        "BRA",
        "BAR",
    )
    static_opcode_sites = {
        opcode: sum(base == opcode for _, _, base in instructions)
        for opcode in opcode_bases
    }
    relevant_offsets = {
        opcode: [f"0x{offset}" for offset, _, base in instructions if base == opcode]
        for opcode in ("LDG", "SHFL", "FMUL", "FADD", "STG")
    }

    properties = torch.cuda.get_device_properties(0)
    block_threads = 256
    raw_registers_per_block = registers * block_threads
    blocks_from_threads = properties.max_threads_per_multi_processor // block_threads
    blocks_from_raw_registers = (
        properties.regs_per_multiprocessor // raw_registers_per_block
    )
    derived_blocks = min(blocks_from_threads, blocks_from_raw_registers)
    theoretical_thread_occupancy = (
        derived_blocks
        * block_threads
        / properties.max_threads_per_multi_processor
    )
    return {
        "available": True,
        "kernel_name_contains": marker,
        "registers_per_thread": registers,
        "static_shared_memory_bytes_per_block": shared,
        "local_memory_bytes_per_thread": local,
        "stack_bytes_per_thread": stack,
        "block_threads": block_threads,
        "derived_theoretical_thread_occupancy": {
            "blocks_from_thread_limit": blocks_from_threads,
            "blocks_from_raw_register_count": blocks_from_raw_registers,
            "raw_registers_per_block": raw_registers_per_block,
            "derived_resident_blocks_per_sm": derived_blocks,
            "active_threads_per_sm": derived_blocks * block_threads,
            "maximum_threads_per_sm": properties.max_threads_per_multi_processor,
            "fraction": theoretical_thread_occupancy,
            "label": "derived theoretical thread occupancy, not achieved occupancy",
            "limitation": (
                "Raw register arithmetic is reported as in M4B. Allocation "
                "granularity does not change the thread-limit conclusion at "
                "this register count; no runtime occupancy counter was used."
            ),
        },
        "sass": {
            "static_opcode_sites": static_opcode_sites,
            "relevant_static_offsets": relevant_offsets,
            "interpretation": (
                "Static sites only, including mutually exclusive helper paths; "
                "not dynamic instruction counts."
            ),
        },
    }


def _seed_for_weight(n: int, k: int) -> int:
    return BASE_SEED + n * 131 + k * 17


def _seed_for_activation(n: int, k: int) -> int:
    return BASE_SEED + n * 23 + k * 7


def _prepare_weight(n: int, k: int) -> NVFP4Tensor:
    generator = torch.Generator(device="cpu").manual_seed(_seed_for_weight(n, k))
    source_cpu = torch.randn((n, k), generator=generator, dtype=torch.float32) * 0.5
    source_cuda = source_cpu.cuda()
    del source_cpu
    weight = quantize_nvfp4_reference(source_cuda)
    del source_cuda
    return weight


def _prepare_activation(m: int, n: int, k: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        _seed_for_activation(n, k)
    )
    activation_row = (
        torch.randn((1, k), generator=generator, dtype=torch.float32) * 0.75
    ).to(torch.bfloat16)
    return activation_row.expand(m, k).contiguous().cuda()


def _bf16_ordered_keys(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    magnitude = bits & 0x7FFF
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def _comparison_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    enforce_policy: bool,
) -> dict[str, object]:
    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise AssertionError("output metadata differs from the BF16 comparison target")
    if not bool(torch.isfinite(actual).all().item()):
        raise AssertionError("projection output contains nonfinite values")

    error = (actual.float() - expected.float()).abs()
    adjacency = (
        _bf16_ordered_keys(actual) - _bf16_ordered_keys(expected)
    ).abs()
    maximum_distance = int(adjacency.max().item())
    result = {
        "maximum_absolute_error": float(error.max().item()),
        "mean_absolute_error": float(error.mean().item()),
        "exact_bf16_fraction": float((actual == expected).float().mean().item()),
        "maximum_bf16_adjacency_distance": maximum_distance,
    }
    if enforce_policy:
        result.update(
            {
                "allowed_maximum_bf16_adjacency_distance": (
                    CORRECTNESS_MAX_BF16_DISTANCE
                ),
                "passed": maximum_distance <= CORRECTNESS_MAX_BF16_DISTANCE,
            }
        )
    return result


def _correctness_guard(
    case: BenchmarkCase,
    x: torch.Tensor,
    weight: NVFP4Tensor,
) -> dict[str, object]:
    expected = w4a16_linear_reference(x, weight)
    baseline = cuda_primitives.cuda_w4a16_linear(x, weight)
    candidate = cuda_primitives.cuda_w4a16_linear_grouped_decode(x, weight)
    torch.cuda.synchronize()

    candidate_reference = _comparison_metrics(
        candidate,
        expected,
        enforce_policy=True,
    )
    baseline_reference = _comparison_metrics(
        baseline,
        expected,
        enforce_policy=True,
    )
    candidate_baseline = _comparison_metrics(
        candidate,
        baseline,
        enforce_policy=False,
    )
    if not candidate_reference["passed"] or not baseline_reference["passed"]:
        raise AssertionError(
            f"{case.name}: frozen one-adjacent-BF16 policy failed"
        )
    return {
        "oracle": "w4a16_linear_reference",
        "checked_shape": list(candidate.shape),
        "candidate_vs_reference": candidate_reference,
        "baseline_vs_reference": baseline_reference,
        "candidate_vs_baseline": candidate_baseline,
        "passed": True,
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _summarize(samples_us: list[float]) -> dict[str, float]:
    ordered = sorted(samples_us)
    values = {
        "median_us": statistics.median(ordered),
        "mean_us": statistics.fmean(ordered),
        "min_us": ordered[0],
        "p95_us": _percentile(ordered, 0.95),
        "standard_deviation_us": statistics.pstdev(ordered),
    }
    return {name: round(value, 3) for name, value in values.items()}


def _warm_both(
    operations: dict[str, Callable[[], torch.Tensor]],
    order: tuple[str, str],
    warmups: int,
) -> None:
    outputs: dict[str, torch.Tensor] = {}
    for _ in range(warmups):
        for name in order:
            outputs[name] = operations[name]()
    torch.cuda.synchronize()
    if set(outputs) != set(operations):
        raise AssertionError("both paths were not warmed")


def _time_cuda_operation(
    operation: Callable[[], torch.Tensor],
    repetitions: int,
) -> tuple[dict[str, float], list[float]]:
    stream = torch.cuda.current_stream()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    output: torch.Tensor | None = None
    for start, end in zip(starts, ends, strict=True):
        start.record(stream)
        output = operation()
        end.record(stream)

    ends[-1].synchronize()
    if output is None:
        raise AssertionError("timed operation did not execute")
    samples = [
        start.elapsed_time(end) * 1_000.0
        for start, end in zip(starts, ends, strict=True)
    ]
    return _summarize(samples), [round(sample, 3) for sample in samples]


def _benchmark_case(
    case: BenchmarkCase,
    x: torch.Tensor,
    weight: NVFP4Tensor,
    warmups: int,
    repetitions: int,
) -> dict[str, object]:
    correctness = _correctness_guard(case, x, weight)
    operations: dict[str, Callable[[], torch.Tensor]] = {
        "baseline": lambda: cuda_primitives.cuda_w4a16_linear(x, weight),
        "candidate": lambda: (
            cuda_primitives.cuda_w4a16_linear_grouped_decode(x, weight)
        ),
    }
    rounds: list[dict[str, object]] = []
    for round_index, order in enumerate(ROUND_ORDERS, start=1):
        _warm_both(operations, order, warmups)
        measurements: dict[str, dict[str, object]] = {}
        for path in order:
            statistics_us, samples_us = _time_cuda_operation(
                operations[path],
                repetitions,
            )
            measurements[path] = {
                **statistics_us,
                "samples_us": samples_us,
            }
        speedup = (
            measurements["baseline"]["median_us"]
            / measurements["candidate"]["median_us"]
        )
        rounds.append(
            {
                "round": round_index,
                "measurement_order": list(order),
                "baseline": measurements["baseline"],
                "candidate": measurements["candidate"],
                "speedup_baseline_median_over_candidate_median": round(
                    speedup,
                    6,
                ),
            }
        )
        print(
            f"{case.name} round={round_index} order={'->'.join(order)} "
            f"baseline={measurements['baseline']['median_us']:.3f}us "
            f"candidate={measurements['candidate']['median_us']:.3f}us "
            f"speedup={speedup:.4f}x",
            flush=True,
        )

    baseline_medians = [
        float(round_result["baseline"]["median_us"])
        for round_result in rounds
    ]
    candidate_medians = [
        float(round_result["candidate"]["median_us"])
        for round_result in rounds
    ]
    round_speedups = [
        float(round_result["speedup_baseline_median_over_candidate_median"])
        for round_result in rounds
    ]
    aggregate = {
        "median_of_round_baseline_medians_us": round(
            statistics.median(baseline_medians), 3
        ),
        "median_of_round_candidate_medians_us": round(
            statistics.median(candidate_medians), 3
        ),
        "median_round_speedup": round(statistics.median(round_speedups), 6),
        "minimum_round_speedup": round(min(round_speedups), 6),
        "maximum_round_speedup": round(max(round_speedups), 6),
        "candidate_faster_rounds": sum(value > 1.0 for value in round_speedups),
        "round_count": len(rounds),
    }
    return {
        "case": asdict(case),
        "identical_input_contract": {
            "same_x_tensor_object": True,
            "same_packed_values_tensor_object": True,
            "same_block_scales_tensor_object": True,
            "same_global_decode_scale_tensor_object": True,
            "quantization_and_setup_outside_timing": True,
        },
        "correctness": correctness,
        "rounds": rounds,
        "aggregate": aggregate,
    }


def _retention_decision(
    results: list[dict[str, object]],
    binary_evidence: dict[str, object],
    external_validation_passed: bool,
    validation_evidence: list[str],
) -> dict[str, object]:
    by_name = {result["case"]["name"]: result for result in results}
    required_names = {case.name for case in CASES}
    complete_matrix = set(by_name) == required_names
    all_correct = all(result["correctness"]["passed"] for result in results)

    q = by_name.get("canonical_q_m1_n3072_k3072")
    kv = by_name.get("canonical_kv_m1_n768_k3072")
    if q is None or kv is None:
        return {
            "decision": "PENDING_INCOMPLETE_MATRIX",
            "complete_matrix": False,
            "external_validation_passed": external_validation_passed,
            "validation_evidence": validation_evidence,
        }

    q_aggregate = q["aggregate"]
    kv_aggregate = kv["aggregate"]
    q_majority = q_aggregate["candidate_faster_rounds"] >= 3
    kv_majority = kv_aggregate["candidate_faster_rounds"] >= 3
    q_speedup = float(q_aggregate["median_round_speedup"])
    kv_speedup = float(kv_aggregate["median_round_speedup"])
    at_least_one_five_percent = max(q_speedup, kv_speedup) >= 1.05
    other_no_meaningful_regression = min(q_speedup, kv_speedup) >= 0.98

    k3072_speedups = [
        float(result["aggregate"]["median_round_speedup"])
        for result in results
        if result["case"]["k"] == 3072
    ]
    no_pathological_k3072_regression = min(k3072_speedups) >= 0.90
    secondary_speedups = [
        float(result["aggregate"]["median_round_speedup"])
        for result in results
        if result["case"]["k"] < 3072
    ]
    no_severe_secondary_regression = min(secondary_speedups) >= 0.75

    resource_ok = bool(binary_evidence.get("available"))
    if resource_ok:
        occupancy = binary_evidence["derived_theoretical_thread_occupancy"]
        resource_ok = float(occupancy["fraction"]) >= 0.5

    performance_passed = all(
        (
            q_majority,
            kv_majority,
            at_least_one_five_percent,
            other_no_meaningful_regression,
            no_pathological_k3072_regression,
            no_severe_secondary_regression,
        )
    )
    eligible = all(
        (
            complete_matrix,
            all_correct,
            external_validation_passed,
            performance_passed,
            resource_ok,
        )
    )
    return {
        "decision": "RETAIN" if eligible else "REJECT",
        "complete_matrix": complete_matrix,
        "correctness_passed": all_correct,
        "external_validation_passed": external_validation_passed,
        "validation_evidence": validation_evidence,
        "primary_performance": {
            "canonical_q_candidate_faster_rounds": q_aggregate[
                "candidate_faster_rounds"
            ],
            "canonical_kv_candidate_faster_rounds": kv_aggregate[
                "candidate_faster_rounds"
            ],
            "required_majority_rounds": 3,
            "canonical_q_median_round_speedup": q_speedup,
            "canonical_kv_median_round_speedup": kv_speedup,
            "at_least_one_canonical_speedup_at_least_1.05": (
                at_least_one_five_percent
            ),
            "other_canonical_speedup_at_least_0.98": (
                other_no_meaningful_regression
            ),
            "passed": performance_passed,
        },
        "engineering": {
            "single_optimization_family": True,
            "resource_check_passed": resource_ok,
            "minimum_representative_k3072_speedup": min(k3072_speedups),
            "no_pathological_k3072_regression_threshold": 0.90,
            "minimum_secondary_speedup": min(secondary_speedups),
            "severe_secondary_regression_threshold": 0.75,
            "passed": (
                resource_ok
                and no_pathological_k3072_regression
                and no_severe_secondary_regression
            ),
        },
        "criteria_note": (
            "Ratios at or below about two percent are treated cautiously; "
            "round consistency, the five-percent canonical threshold, and "
            "regression guards determine eligibility."
        ),
    }


def _selected_cases(names: list[str] | None) -> tuple[BenchmarkCase, ...]:
    if not names:
        return CASES
    by_name = {case.name: case for case in CASES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown case name(s): {', '.join(unknown)}")
    return tuple(by_name[name] for name in names)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="run one named case; may be repeated (default: full matrix)",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--external-validation-passed",
        action="store_true",
        help=(
            "assert that candidate/full pytest and both required memcheck "
            "runs were completed successfully before this benchmark"
        ),
    )
    parser.add_argument(
        "--validation-evidence",
        action="append",
        default=[],
        help="record one externally completed validation result",
    )
    args = parser.parse_args()
    if args.warmups < 25:
        parser.error("--warmups must be at least 25 for M4C")
    if args.repetitions < 200:
        parser.error("--repetitions must be at least 200 for M4C")
    return args


def main() -> None:
    args = _parse_args()
    if args.list_cases:
        for case in CASES:
            print(case.name)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the CUDA device does not support BF16")

    selected = _selected_cases(args.cases)
    weights: dict[tuple[int, int], NVFP4Tensor] = {}
    results: list[dict[str, object]] = []
    binary_evidence = _candidate_binary_evidence()
    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"capability={torch.cuda.get_device_capability(0)} "
        f"rounds={len(ROUND_ORDERS)} warmups={args.warmups} "
        f"repetitions={args.repetitions}",
        flush=True,
    )

    with torch.inference_mode():
        for case in selected:
            key = (case.n, case.k)
            if key not in weights:
                weights[key] = _prepare_weight(case.n, case.k)
            weight = weights[key]
            x = _prepare_activation(case.m, case.n, case.k)
            results.append(
                _benchmark_case(
                    case,
                    x,
                    weight,
                    args.warmups,
                    args.repetitions,
                )
            )

    decision = _retention_decision(
        results,
        binary_evidence,
        args.external_validation_passed,
        args.validation_evidence,
    )
    result_document = {
        "schema_version": 1,
        "milestone": "M4C",
        "experiment": "grouped_decode",
        "baseline_commit": BASELINE_COMMIT,
        "m4b_commit": _run_text(["git", "rev-parse", M4B_COMMIT]),
        "environment": _environment(),
        "methodology": {
            "timer": "torch.cuda.Event",
            "stream": "current CUDA stream for events and both operations",
            "rounds": len(ROUND_ORDERS),
            "round_orders": [list(order) for order in ROUND_ORDERS],
            "warmups_per_path_before_each_round": args.warmups,
            "samples_per_path_per_round": args.repetitions,
            "warmup_policy": (
                "both paths are warmed in that round's measurement order, "
                "then synchronized before either timed path"
            ),
            "completion_synchronization": (
                "one synchronization after each round warmup and one on the "
                "final end event for each path measurement"
            ),
            "p95_definition": (
                "linear interpolation at rank (sample_count - 1) * 0.95"
            ),
            "speedup_definition": (
                "baseline round median divided by candidate round median"
            ),
            "aggregate_definition": (
                "medians are taken across the five per-round medians or "
                "per-round speedup ratios; no round is discarded"
            ),
            "excluded": [
                "extension loading",
                "compilation",
                "random generation",
                "quantization",
                "input allocation and transfer",
                "correctness reference",
                "static binary inspection",
            ],
            "conditions": (
                "uncontrolled consumer-laptop clocks and power; no clock, "
                "power-limit, driver, Windows, or WSL changes"
            ),
        },
        "structural_source_accounting": {
            "baseline_per_output": {
                "packed_byte_loads_requested_at_logical_weight_granularity": "K",
                "ue4m3_decode_invocations": "K",
                "e2m1_decode_invocations": "K",
            },
            "candidate_per_output": {
                "packed_byte_loads": "K/2",
                "ue4m3_decode_invocations": "K/16",
                "e2m1_decode_invocations": "K",
            },
            "unchanged": {
                "activation_logical_loads": "K",
                "products": "K",
            },
            "classification": (
                "derived from source organization, not hardware-counter or "
                "dynamic instruction measurements"
            ),
        },
        "candidate_binary_evidence": binary_evidence,
        "matrix": [asdict(case) for case in selected],
        "cases": results,
        "candidate_decision": decision,
        "limitations": [
            "One uncontrolled consumer laptop GPU was measured.",
            "CUDA-event timing excludes Python dispatch and host allocator time.",
            "No Nsight performance counters, achieved occupancy, SOL, DRAM "
            "bandwidth, L2 hit rate, pipeline utilization, or warp stalls are claimed.",
            "SASS opcode sites are static and are not dynamic instruction counts.",
            "Inputs are deterministic representative random tensors, not trained weights.",
        ],
    }

    output_path = args.output.resolve()
    if output_path == FROZEN_M4B_RESULT_PATH.resolve():
        raise ValueError("M4C output must not overwrite baseline_w4a16.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".bench.tmp")
    temporary_path.write_text(
        json.dumps(result_document, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(f"decision={decision['decision']}", flush=True)
    print(f"wrote={output_path}", flush=True)


if __name__ == "__main__":
    main()
