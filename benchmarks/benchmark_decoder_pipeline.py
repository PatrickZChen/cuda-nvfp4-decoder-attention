#!/usr/bin/env python3
"""M6 reproducible end-to-end and stage-attribution decode baseline.

The primary operation is exactly ``cuda_decoder_attention_forward_`` from
``decoder_attention_cuda.py``.  The isolated-stage measurements explicitly
compose only the existing public CUDA primitives.  Fixture construction,
representative-weight quantization, correctness work, cache establishment,
and stage-input preparation are outside every timed interval.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cuda_primitives  # noqa: E402
import decoder_attention_cuda  # noqa: E402
from reference.decoder_attention_nvfp4 import (  # noqa: E402
    decoder_attention_nvfp4_reference,
)
from reference.nvfp4 import (  # noqa: E402
    NVFP4Tensor,
    quantize_nvfp4_reference,
)


DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "results"
    / "rtx4080-laptop-sm89"
    / "m6_decoder_pipeline_baseline.json"
)
BENCHMARK_BASELINE_HEAD = "8aa04d062b6ab59d916619561c12d0e62c645592"

HIDDEN_SIZE = 3072
QUERY_HEADS = 24
KV_HEADS = 6
HEAD_DIM = 128
TOKEN_COUNT = 1
CACHE_CAPACITY = 16_384
BATCH_SIZES = (1, 2)
PAST_LENGTHS = (0, 128, 512, 2_048, 8_192)
RMS_EPS = 1.0e-6
ROPE_THETA = 10_000.0
SOURCE_STD = 1.0 / math.sqrt(HIDDEN_SIZE)

WEIGHT_SEEDS = {
    "q_weight": 61_001,
    "k_weight": 61_002,
    "v_weight": 61_003,
    "out_weight": 61_004,
}
FIXTURE_SEEDS = {
    "hidden_states": 62_001,
    "input_norm_weight": 62_002,
    "q_norm_weight": 62_003,
    "k_norm_weight": 62_004,
    "master_k_cache": 62_005,
    "master_v_cache": 62_006,
}

WEIGHT_SHAPES = {
    "q_weight": (HIDDEN_SIZE, HIDDEN_SIZE),
    "k_weight": (KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
    "v_weight": (KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
    "out_weight": (HIDDEN_SIZE, HIDDEN_SIZE),
}

STAGE_NAMES = (
    "input_rmsnorm",
    "q_projection",
    "k_projection",
    "v_projection",
    "q_rmsnorm",
    "k_rmsnorm",
    "q_rope",
    "k_rope",
    "kv_cache_append",
    "cached_gqa",
    "output_projection",
)

CATEGORY_STAGES = {
    "projection": (
        "q_projection",
        "k_projection",
        "v_projection",
        "output_projection",
    ),
    "normalization_and_rope": (
        "input_rmsnorm",
        "q_rmsnorm",
        "k_rmsnorm",
        "q_rope",
        "k_rope",
    ),
    "cache_append": ("kv_cache_append",),
    "attention": ("cached_gqa",),
}

FROZEN_PRODUCTION_PATHS = (
    "decoder_attention_cuda.py",
    "reference/decoder_attention_nvfp4.py",
    "src/gqa_attention_cached.cu",
    "src/gqa_attention.cu",
    "src/kv_cache.cu",
    "src/w4a16.cu",
    "src/w4a16_grouped.cu",
    "src/rmsnorm.cu",
    "src/rope.cu",
    "src/nvfp4.cu",
)
PRIOR_RESULT_PATHS = (
    "benchmarks/results/rtx4080-laptop-sm89/baseline_w4a16.json",
    "benchmarks/results/rtx4080-laptop-sm89/m4c_grouped_decode.json",
)


@dataclass(frozen=True)
class BenchmarkCase:
    batch_size: int
    past_length: int

    @property
    def case_id(self) -> str:
        return f"b{self.batch_size}_p{self.past_length}"

    @property
    def logical_context_length(self) -> int:
        return self.past_length + TOKEN_COUNT

    def as_dict(self) -> dict[str, int | str]:
        return {
            "case_id": self.case_id,
            "B": self.batch_size,
            "T": TOKEN_COUNT,
            "P": self.past_length,
            "S": self.logical_context_length,
            "H": HIDDEN_SIZE,
            "Hq": QUERY_HEADS,
            "Hkv": KV_HEADS,
            "D": HEAD_DIM,
            "C": CACHE_CAPACITY,
        }


PRIMARY_CASES = tuple(
    BenchmarkCase(batch_size, past_length)
    for batch_size in BATCH_SIZES
    for past_length in PAST_LENGTHS
)


@dataclass
class MasterFixture:
    weights: dict[str, NVFP4Tensor]
    weight_statistics: dict[str, dict[str, Any]]
    x_master: torch.Tensor
    input_norm_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    master_k_cache_cpu: torch.Tensor
    master_v_cache_cpu: torch.Tensor


@dataclass
class CaseFixture:
    case: BenchmarkCase
    x: torch.Tensor
    input_norm_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    weights: dict[str, NVFP4Tensor]
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    master_k_cache_cpu: torch.Tensor
    master_v_cache_cpu: torch.Tensor


@dataclass
class StageSnapshot:
    x_norm: torch.Tensor
    q_heads: torch.Tensor
    k_heads: torch.Tensor
    v_heads: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    q_rope: torch.Tensor
    k_rope: torch.Tensor
    context_flat: torch.Tensor


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


def _run_bytes(command: list[str]) -> bytes | None:
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_artifact_audit() -> dict[str, Any]:
    test_paths = tuple(
        str(path.relative_to(REPOSITORY_ROOT))
        for path in sorted((REPOSITORY_ROOT / "tests").glob("*.py"))
    )
    paths = FROZEN_PRODUCTION_PATHS + test_paths + PRIOR_RESULT_PATHS
    artifacts: dict[str, Any] = {}
    for relative_path in paths:
        current_path = REPOSITORY_ROOT / relative_path
        head_payload = _run_bytes(["git", "show", f"HEAD:{relative_path}"])
        current_hash = _sha256_file(current_path) if current_path.is_file() else None
        head_hash = _sha256_bytes(head_payload) if head_payload is not None else None
        artifacts[relative_path] = {
            "working_tree_sha256": current_hash,
            "head_blob_sha256": head_hash,
            "matches_head": current_hash is not None and current_hash == head_hash,
        }
    return {
        "classification": "byte hashes of working-tree files and HEAD blobs",
        "all_match_head": all(item["matches_head"] for item in artifacts.values()),
        "artifacts": artifacts,
    }


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _nvidia_smi_snapshot() -> dict[str, Any] | None:
    fields = (
        "name,driver_version,temperature.gpu,clocks.gr,clocks.mem,"
        "power.draw,power.limit,power.default_limit,power.max_limit"
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
    if len(values) != 9:
        return {"raw_query": output}
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": values[0],
        "driver_version": values[1],
        "temperature_c": _optional_float(values[2]),
        "graphics_clock_mhz": _optional_float(values[3]),
        "memory_clock_mhz": _optional_float(values[4]),
        "power_draw_w": _optional_float(values[5]),
        "power_limit_w": _optional_float(values[6]),
        "default_power_limit_w": _optional_float(values[7]),
        "maximum_power_limit_w": _optional_float(values[8]),
        "raw_query": output,
    }


def _cuda_runtime_versions() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        cudart = torch.cuda.cudart()
        runtime_error, runtime_version = cudart.cudaRuntimeGetVersion()
        driver_error, driver_version = cudart.cudaDriverGetVersion()
        result.update(
            {
                "cuda_runtime_get_version_status": int(runtime_error),
                "cuda_runtime_version_integer": int(runtime_version),
                "cuda_driver_get_version_status": int(driver_error),
                "cuda_driver_version_integer": int(driver_version),
            }
        )
    except Exception as error:  # pragma: no cover - environment diagnostic
        result["cudart_query_error"] = f"{type(error).__name__}: {error}"
    compiled_version = getattr(torch._C, "_cuda_getCompiledVersion", None)
    if compiled_version is not None:
        try:
            result["pytorch_compiled_cuda_version_integer"] = int(
                compiled_version()
            )
        except Exception as error:  # pragma: no cover - environment diagnostic
            result["compiled_version_query_error"] = (
                f"{type(error).__name__}: {error}"
            )
    return result


def _environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    status = _run_text(["git", "status", "--short"])
    head = _run_text(["git", "rev-parse", "HEAD"])
    return {
        "benchmark_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "head_sha": head,
            "requested_baseline_head_sha": BENCHMARK_BASELINE_HEAD,
            "matches_requested_baseline_head": head == BENCHMARK_BASELINE_HEAD,
            "worktree_dirty_at_capture": None if status is None else bool(status),
            "worktree_status_short_at_capture": [] if not status else status.splitlines(),
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "architecture_label": "Ada SM89",
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "l2_cache_bytes": properties.L2_cache_size,
            "nvidia_smi_before_benchmark": _nvidia_smi_snapshot(),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "pytorch_cuda_build": torch.version.cuda,
            "cuda_runtime": _cuda_runtime_versions(),
            "nvcc": _run_text(["nvcc", "--version"]),
            "nvidia_smi_header": _run_text(["nvidia-smi"]),
            "platform": platform.platform(),
            "os_uname": " ".join(platform.uname()),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "clock_power_policy": (
            "Uncontrolled consumer-laptop conditions. No clocks, power limit, "
            "driver, registry, Windows, WSL, or other system setting was changed."
        ),
    }


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _weight_storage_bytes(weight: NVFP4Tensor) -> dict[str, int]:
    parts = {
        "packed_values_bytes": _tensor_bytes(weight.packed_values),
        "block_scales_bytes": _tensor_bytes(weight.block_scales),
        "global_decode_scale_bytes": _tensor_bytes(weight.global_decode_scale),
    }
    parts["total_bytes"] = sum(parts.values())
    return parts


def _tensor_payload_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().numpy().tobytes()


def _histogram(values: torch.Tensor, bins: int, *, hexadecimal: bool) -> dict[str, int]:
    counts = torch.bincount(values.reshape(-1).to(torch.int64), minlength=bins)
    result: dict[str, int] = {}
    for index in range(bins):
        key = f"0x{index:02x}" if hexadecimal else str(index)
        result[key] = int(counts[index].item())
    return result


def _representation_statistics(
    name: str,
    seed: int,
    source: torch.Tensor,
    weight: NVFP4Tensor,
) -> dict[str, Any]:
    packed = weight.packed_values
    low_counts = torch.bincount(
        (packed & 0x0F).reshape(-1).to(torch.int64), minlength=16
    )
    high_counts = torch.bincount(
        (packed >> 4).reshape(-1).to(torch.int64), minlength=16
    )
    nibble_counts = low_counts + high_counts
    nibble_histogram = {
        f"0x{index:x}": int(nibble_counts[index].item()) for index in range(16)
    }

    scales = weight.block_scales
    scale_exponents = (scales.to(torch.int16) >> 3).to(torch.int64)
    exponent_histogram = _histogram(scale_exponents, 16, hexadecimal=False)
    scale_byte_histogram = _histogram(scales, 127, hexadecimal=True)
    nonzero_scale_histogram = {
        key: count for key, count in scale_byte_histogram.items() if count
    }
    block_count = scales.numel()

    storage_digest = hashlib.sha256()
    storage_digest.update(_tensor_payload_bytes(weight.packed_values))
    storage_digest.update(_tensor_payload_bytes(weight.block_scales))
    storage_digest.update(_tensor_payload_bytes(weight.global_decode_scale))

    actual_std = float(source.std(unbiased=False).item())
    actual_mean = float(source.mean().item())
    exact_zero_fraction = float((source == 0.0).to(torch.float32).mean().item())
    return {
        "name": name,
        "description": "deterministic representative random weight; not trained-model data",
        "source_seed": seed,
        "source_distribution": "independent dense FP32 normal",
        "source_standard_deviation_rule": "1 / sqrt(H)",
        "source_standard_deviation_value": SOURCE_STD,
        "logical_shape": list(weight.logical_shape),
        "source_statistics": {
            "mean": actual_mean,
            "population_standard_deviation": actual_std,
            "minimum": float(source.min().item()),
            "maximum": float(source.max().item()),
            "exact_zero_fraction": exact_zero_fraction,
        },
        "global_decode_scale": float(weight.global_decode_scale.item()),
        "block_scale_count": block_count,
        "block_scale_zero_fraction": float(
            (scales == 0).to(torch.float64).mean().item()
        ),
        "unique_block_scale_byte_count": int(torch.unique(scales).numel()),
        "ue4m3_scale_byte_histogram_nonzero_entries": nonzero_scale_histogram,
        "ue4m3_exponent_histogram": exponent_histogram,
        "e2m1_nibble_histogram": nibble_histogram,
        "e2m1_zero_code_fraction": float(
            (nibble_counts[0] + nibble_counts[8]).item() / source.numel()
        ),
        "portable_storage_bytes": _weight_storage_bytes(weight),
        "portable_storage_sha256": storage_digest.hexdigest(),
    }


def _nvfp4_to_cuda(weight: NVFP4Tensor) -> NVFP4Tensor:
    return NVFP4Tensor(
        packed_values=weight.packed_values.cuda(),
        block_scales=weight.block_scales.cuda(),
        global_decode_scale=weight.global_decode_scale.cuda(),
        logical_shape=weight.logical_shape,
    )


def _prepare_representative_weights() -> tuple[
    dict[str, NVFP4Tensor], dict[str, dict[str, Any]]
]:
    weights: dict[str, NVFP4Tensor] = {}
    statistics_by_weight: dict[str, dict[str, Any]] = {}
    for name, shape in WEIGHT_SHAPES.items():
        seed = WEIGHT_SEEDS[name]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        source = torch.empty(shape, dtype=torch.float32, device="cpu")
        source.normal_(mean=0.0, std=SOURCE_STD, generator=generator)
        quantized_cpu = quantize_nvfp4_reference(source)
        statistics_by_weight[name] = _representation_statistics(
            name,
            seed,
            source,
            quantized_cpu,
        )
        weights[name] = _nvfp4_to_cuda(quantized_cpu)
        print(
            f"prepared {name} shape={shape} seed={seed} "
            f"unique_scales={statistics_by_weight[name]['unique_block_scale_byte_count']}",
            flush=True,
        )
        del source, quantized_cpu
        gc.collect()
    torch.cuda.synchronize()
    return weights, statistics_by_weight


def _normal_bf16_cpu(
    shape: tuple[int, ...],
    *,
    seed: int,
    standard_deviation: float,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = torch.empty(shape, dtype=torch.bfloat16, device="cpu")
    result.normal_(
        mean=0.0,
        std=standard_deviation,
        generator=generator,
    )
    return result


def _gamma_bf16_cpu(length: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    perturbation = torch.randn(
        (length,), generator=generator, dtype=torch.float32
    ) * 0.02
    return (1.0 + perturbation).to(torch.bfloat16).contiguous()


def _prepare_master_fixture() -> MasterFixture:
    weights, weight_statistics = _prepare_representative_weights()
    x_master_cpu = _normal_bf16_cpu(
        (max(BATCH_SIZES), TOKEN_COUNT, HIDDEN_SIZE),
        seed=FIXTURE_SEEDS["hidden_states"],
        standard_deviation=1.0,
    )
    input_norm_weight_cpu = _gamma_bf16_cpu(
        HIDDEN_SIZE, seed=FIXTURE_SEEDS["input_norm_weight"]
    )
    q_norm_weight_cpu = _gamma_bf16_cpu(
        HEAD_DIM, seed=FIXTURE_SEEDS["q_norm_weight"]
    )
    k_norm_weight_cpu = _gamma_bf16_cpu(
        HEAD_DIM, seed=FIXTURE_SEEDS["k_norm_weight"]
    )
    cache_shape = (
        max(BATCH_SIZES),
        KV_HEADS,
        CACHE_CAPACITY,
        HEAD_DIM,
    )
    master_k_cache_cpu = _normal_bf16_cpu(
        cache_shape,
        seed=FIXTURE_SEEDS["master_k_cache"],
        standard_deviation=0.5,
    )
    master_v_cache_cpu = _normal_bf16_cpu(
        cache_shape,
        seed=FIXTURE_SEEDS["master_v_cache"],
        standard_deviation=0.75,
    )
    fixture = MasterFixture(
        weights=weights,
        weight_statistics=weight_statistics,
        x_master=x_master_cpu.cuda(),
        input_norm_weight=input_norm_weight_cpu.cuda(),
        q_norm_weight=q_norm_weight_cpu.cuda(),
        k_norm_weight=k_norm_weight_cpu.cuda(),
        master_k_cache_cpu=master_k_cache_cpu,
        master_v_cache_cpu=master_v_cache_cpu,
    )
    torch.cuda.synchronize()
    return fixture


def _make_case_fixture(master: MasterFixture, case: BenchmarkCase) -> CaseFixture:
    batch_size = case.batch_size
    k_cache = master.master_k_cache_cpu[:batch_size].cuda()
    v_cache = master.master_v_cache_cpu[:batch_size].cuda()
    fixture = CaseFixture(
        case=case,
        x=master.x_master[:batch_size],
        input_norm_weight=master.input_norm_weight,
        q_norm_weight=master.q_norm_weight,
        k_norm_weight=master.k_norm_weight,
        weights=master.weights,
        k_cache=k_cache,
        v_cache=v_cache,
        master_k_cache_cpu=master.master_k_cache_cpu,
        master_v_cache_cpu=master.master_v_cache_cpu,
    )
    torch.cuda.synchronize()
    return fixture


def _restore_current_slot(fixture: CaseFixture) -> None:
    case = fixture.case
    batch_size = case.batch_size
    position = case.past_length
    fixture.k_cache[:, :, position : position + 1, :].copy_(
        fixture.master_k_cache_cpu[
            :batch_size, :, position : position + 1, :
        ]
    )
    fixture.v_cache[:, :, position : position + 1, :].copy_(
        fixture.master_v_cache_cpu[
            :batch_size, :, position : position + 1, :
        ]
    )


def _public_pipeline_call(fixture: CaseFixture) -> torch.Tensor:
    return decoder_attention_cuda.cuda_decoder_attention_forward_(
        fixture.x,
        fixture.input_norm_weight,
        fixture.weights["q_weight"],
        fixture.weights["k_weight"],
        fixture.weights["v_weight"],
        fixture.q_norm_weight,
        fixture.k_norm_weight,
        fixture.weights["out_weight"],
        fixture.k_cache,
        fixture.v_cache,
        fixture.case.past_length,
        rms_eps=RMS_EPS,
        rope_theta=ROPE_THETA,
    )


def _bf16_ordered_keys(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    magnitude = bits & 0x7FFF
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def _bf16_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(
        left.contiguous().view(torch.int16),
        right.contiguous().view(torch.int16),
    )


def _comparison_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if actual.shape != expected.shape:
        raise AssertionError("comparison shapes differ")
    if actual.dtype != expected.dtype or actual.dtype != torch.bfloat16:
        raise AssertionError("comparison tensors must both be BF16")
    if not bool(torch.isfinite(actual).all().item()):
        raise AssertionError("actual output contains nonfinite values")
    if not bool(torch.isfinite(expected).all().item()):
        raise AssertionError("reference output contains nonfinite values")
    error = (actual.float() - expected.float()).abs()
    adjacency = (
        _bf16_ordered_keys(actual) - _bf16_ordered_keys(expected)
    ).abs()
    absolute_floor = 2.0**-20
    policy_mask = (adjacency <= 1) | (error <= absolute_floor)
    return {
        "maximum_absolute_error": float(error.max().item()),
        "mean_absolute_error": float(error.mean().item()),
        "exact_bf16_fraction": float((actual == expected).float().mean().item()),
        "maximum_bf16_adjacency_distance": int(adjacency.max().item()),
        "frozen_elementwise_policy": (
            "BF16 adjacency distance <= 1 or absolute error <= 2^-20"
        ),
        "passed_frozen_elementwise_policy": bool(policy_mask.all().item()),
    }


def _correctness_precheck(fixture: CaseFixture) -> dict[str, Any]:
    _restore_current_slot(fixture)
    torch.cuda.synchronize()
    k_pointer_before = fixture.k_cache.data_ptr()
    v_pointer_before = fixture.v_cache.data_ptr()
    first = _public_pipeline_call(fixture)
    torch.cuda.synchronize()
    first_k_slot = fixture.k_cache[
        :, :, fixture.case.past_length : fixture.case.past_length + 1, :
    ].clone()
    first_v_slot = fixture.v_cache[
        :, :, fixture.case.past_length : fixture.case.past_length + 1, :
    ].clone()
    second = _public_pipeline_call(fixture)
    torch.cuda.synchronize()

    expected_shape = (fixture.case.batch_size, TOKEN_COUNT, HIDDEN_SIZE)
    checks = {
        "expected_output_shape": list(expected_shape),
        "actual_output_shape": list(second.shape),
        "output_shape_passed": tuple(second.shape) == expected_shape,
        "output_dtype": str(second.dtype),
        "output_dtype_is_bfloat16": second.dtype == torch.bfloat16,
        "output_all_finite": bool(torch.isfinite(second).all().item()),
        "k_cache_data_ptr_before": k_pointer_before,
        "k_cache_data_ptr_after": fixture.k_cache.data_ptr(),
        "k_cache_data_ptr_unchanged": fixture.k_cache.data_ptr() == k_pointer_before,
        "v_cache_data_ptr_before": v_pointer_before,
        "v_cache_data_ptr_after": fixture.v_cache.data_ptr(),
        "v_cache_data_ptr_unchanged": fixture.v_cache.data_ptr() == v_pointer_before,
        "repeated_output_bitwise_deterministic": _bf16_bitwise_equal(first, second),
        "repeated_k_slot_bitwise_deterministic": _bf16_bitwise_equal(
            first_k_slot,
            fixture.k_cache[
                :, :, fixture.case.past_length : fixture.case.past_length + 1, :
            ],
        ),
        "repeated_v_slot_bitwise_deterministic": _bf16_bitwise_equal(
            first_v_slot,
            fixture.v_cache[
                :, :, fixture.case.past_length : fixture.case.past_length + 1, :
            ],
        ),
        "cache_mutation_semantics": (
            "Each invocation recomputes and overwrites only slot P for T=1; "
            "the logical prefix is unchanged and is shared across repetitions."
        ),
    }
    checks["passed"] = all(
        (
            checks["output_shape_passed"],
            checks["output_dtype_is_bfloat16"],
            checks["output_all_finite"],
            checks["k_cache_data_ptr_unchanged"],
            checks["v_cache_data_ptr_unchanged"],
            checks["repeated_output_bitwise_deterministic"],
            checks["repeated_k_slot_bitwise_deterministic"],
            checks["repeated_v_slot_bitwise_deterministic"],
        )
    )
    if not checks["passed"]:
        raise AssertionError(
            f"{fixture.case.case_id}: correctness precheck failed: {checks}"
        )
    return checks


def _representative_reference_check(fixture: CaseFixture) -> dict[str, Any]:
    if fixture.case.batch_size != 1 or fixture.case.past_length != 128:
        raise ValueError("the representative reference check is fixed at B=1, P=128")
    past_length = fixture.case.past_length
    compact_past_k = fixture.k_cache[:, :, :past_length, :].clone()
    compact_past_v = fixture.v_cache[:, :, :past_length, :].clone()
    expected = decoder_attention_nvfp4_reference(
        fixture.x,
        fixture.input_norm_weight,
        fixture.weights["q_weight"],
        fixture.weights["k_weight"],
        fixture.weights["v_weight"],
        fixture.q_norm_weight,
        fixture.k_norm_weight,
        fixture.weights["out_weight"],
        compact_past_k,
        compact_past_v,
        rms_eps=RMS_EPS,
        rope_theta=ROPE_THETA,
        return_debug=False,
    )
    _restore_current_slot(fixture)
    actual = _public_pipeline_call(fixture)
    torch.cuda.synchronize()
    metrics = _comparison_metrics(actual, expected.output)
    metrics.update(
        {
            "case_id": fixture.case.case_id,
            "oracle": "decoder_attention_nvfp4_reference",
            "same_representative_quantized_weights": True,
            "same_master_cache_prefix": True,
            "timed": False,
        }
    )
    if not metrics["passed_frozen_elementwise_policy"]:
        raise AssertionError(
            "representative end-to-end reference check violates the frozen "
            f"numerical policy: {metrics}"
        )
    return metrics


def _run_all_correctness_prechecks(
    master: MasterFixture,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    prechecks: dict[str, dict[str, Any]] = {}
    reference_metrics: dict[str, Any] | None = None
    for case in PRIMARY_CASES:
        fixture = _make_case_fixture(master, case)
        prechecks[case.case_id] = _correctness_precheck(fixture)
        if case.batch_size == 1 and case.past_length == 128:
            reference_metrics = _representative_reference_check(fixture)
        print(f"precheck {case.case_id}: passed", flush=True)
        del fixture
        gc.collect()
        torch.cuda.synchronize()
    if reference_metrics is None:
        raise AssertionError("the canonical representative reference check did not run")
    return prechecks, reference_metrics


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sample")
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


def _event_samples(
    operation: Callable[[], Any],
    repetitions: int,
) -> list[float]:
    stream = torch.cuda.current_stream()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    result: Any = None
    for start, end in zip(starts, ends, strict=True):
        start.record(stream)
        result = operation()
        end.record(stream)
    ends[-1].synchronize()
    del result
    return [
        start.elapsed_time(end) * 1_000.0
        for start, end in zip(starts, ends, strict=True)
    ]


def _warm_operation(operation: Callable[[], Any], warmups: int) -> None:
    result: Any = None
    for _ in range(warmups):
        result = operation()
    torch.cuda.synchronize()
    del result


def _round_aggregate(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    medians = [float(round_result["statistics"]["median_us"]) for round_result in rounds]
    minimum = min(medians)
    maximum = max(medians)
    ratio = maximum / minimum
    spread = ratio - 1.0
    if spread <= 0.05:
        label = "low_round_median_spread"
    elif spread <= 0.15:
        label = "noticeable_round_median_spread"
    else:
        label = "substantial_round_median_spread"
    return {
        "median_of_round_medians_us": round(statistics.median(medians), 3),
        "minimum_round_median_us": round(minimum, 3),
        "maximum_round_median_us": round(maximum, 3),
        "maximum_over_minimum_round_median_ratio": round(ratio, 6),
        "round_median_spread": round(spread, 6),
        "stability_label": label,
        "round_count": len(rounds),
    }


def _benchmark_end_to_end(
    fixture: CaseFixture,
    *,
    warmups: int,
    rounds: int,
    samples_per_round: int,
) -> dict[str, Any]:
    round_results: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        _restore_current_slot(fixture)
        _warm_operation(lambda: _public_pipeline_call(fixture), warmups)
        samples = _event_samples(
            lambda: _public_pipeline_call(fixture),
            samples_per_round,
        )
        statistics_us = _summarize(samples)
        round_results.append(
            {
                "round": round_index,
                "statistics": statistics_us,
                "raw_samples_us": [round(value, 3) for value in samples],
            }
        )
        print(
            f"e2e {fixture.case.case_id} round={round_index} "
            f"median={statistics_us['median_us']:.3f}us",
            flush=True,
        )
    aggregate = _round_aggregate(round_results)
    return {
        "metric": "end_to_end_stream_elapsed_us",
        "operation": "decoder_attention_cuda.cuda_decoder_attention_forward_",
        "projection_backend": "cuda_primitives.cuda_w4a16_linear for Q/K/V/O",
        "timer": "torch.cuda.Event on the PyTorch current CUDA stream",
        "interpretation": (
            "Elapsed stream interval between start and end events around one "
            "public pipeline call. It can include stream-visible idle gaps from "
            "Python host submission when the GPU catches up; it is not a sum of "
            "pure kernel execution times or host-free arithmetic time."
        ),
        "warmups_before_each_round": warmups,
        "rounds": round_results,
        "aggregate": aggregate,
        "outlier_policy": "all raw samples retained; no filtering",
    }


def _benchmark_wall(
    fixture: CaseFixture,
    *,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    _restore_current_slot(fixture)
    _warm_operation(lambda: _public_pipeline_call(fixture), warmups)
    raw_samples_us: list[float] = []
    result: torch.Tensor | None = None
    for _ in range(samples):
        torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        result = _public_pipeline_call(fixture)
        torch.cuda.synchronize()
        end_ns = time.perf_counter_ns()
        raw_samples_us.append((end_ns - start_ns) / 1_000.0)
    if result is None:
        raise AssertionError("synchronized wall timing did not execute")
    del result
    return {
        "metric": "synchronized_wall_us",
        "timer": "time.perf_counter_ns",
        "warmups": warmups,
        "sample_count": samples,
        "statistics": _summarize(raw_samples_us),
        "raw_samples_us": [round(value, 3) for value in raw_samples_us],
        "interpretation": (
            "Each sample synchronizes before the host timer, invokes one public "
            "pipeline call, and synchronizes before stopping. It includes Python "
            "orchestration, validation, allocator behavior, kernel submission, "
            "GPU work, and synchronization overhead."
        ),
        "delta_caveat": (
            "A wall-versus-stream difference is not an isolated host-launch "
            "overhead measurement."
        ),
        "outlier_policy": "all raw samples retained; no filtering",
    }


def _prepare_stage_snapshot(fixture: CaseFixture) -> StageSnapshot:
    batch_size = fixture.case.batch_size
    x_norm = cuda_primitives.cuda_rms_norm(
        fixture.x, fixture.input_norm_weight, RMS_EPS
    )
    q_flat = cuda_primitives.cuda_w4a16_linear(
        x_norm, fixture.weights["q_weight"]
    )
    k_flat = cuda_primitives.cuda_w4a16_linear(
        x_norm, fixture.weights["k_weight"]
    )
    v_flat = cuda_primitives.cuda_w4a16_linear(
        x_norm, fixture.weights["v_weight"]
    )
    q_heads = q_flat.reshape(
        batch_size, TOKEN_COUNT, QUERY_HEADS, HEAD_DIM
    )
    k_heads = k_flat.reshape(batch_size, TOKEN_COUNT, KV_HEADS, HEAD_DIM)
    v_heads = v_flat.reshape(batch_size, TOKEN_COUNT, KV_HEADS, HEAD_DIM)
    q_norm = cuda_primitives.cuda_rms_norm(
        q_heads, fixture.q_norm_weight, RMS_EPS
    )
    k_norm = cuda_primitives.cuda_rms_norm(
        k_heads, fixture.k_norm_weight, RMS_EPS
    )
    q_rope = cuda_primitives.cuda_apply_rope(
        q_norm, fixture.case.past_length, ROPE_THETA
    )
    k_rope = cuda_primitives.cuda_apply_rope(
        k_norm, fixture.case.past_length, ROPE_THETA
    )
    cuda_primitives.cuda_kv_cache_append_(
        fixture.k_cache,
        fixture.v_cache,
        k_rope,
        v_heads,
        fixture.case.past_length,
    )
    context = cuda_primitives.cuda_gqa_attention_cached(
        q_rope,
        fixture.k_cache,
        fixture.v_cache,
        fixture.case.past_length,
    )
    context_flat = context.reshape(batch_size, TOKEN_COUNT, HIDDEN_SIZE)
    torch.cuda.synchronize()
    return StageSnapshot(
        x_norm=x_norm,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        q_norm=q_norm,
        k_norm=k_norm,
        q_rope=q_rope,
        k_rope=k_rope,
        context_flat=context_flat,
    )


def _stage_operations(
    fixture: CaseFixture,
    snapshot: StageSnapshot,
) -> dict[str, tuple[str, Callable[[], Any]]]:
    return {
        "input_rmsnorm": (
            "cuda_primitives.cuda_rms_norm",
            lambda: cuda_primitives.cuda_rms_norm(
                fixture.x, fixture.input_norm_weight, RMS_EPS
            ),
        ),
        "q_projection": (
            "cuda_primitives.cuda_w4a16_linear",
            lambda: cuda_primitives.cuda_w4a16_linear(
                snapshot.x_norm, fixture.weights["q_weight"]
            ),
        ),
        "k_projection": (
            "cuda_primitives.cuda_w4a16_linear",
            lambda: cuda_primitives.cuda_w4a16_linear(
                snapshot.x_norm, fixture.weights["k_weight"]
            ),
        ),
        "v_projection": (
            "cuda_primitives.cuda_w4a16_linear",
            lambda: cuda_primitives.cuda_w4a16_linear(
                snapshot.x_norm, fixture.weights["v_weight"]
            ),
        ),
        "q_rmsnorm": (
            "cuda_primitives.cuda_rms_norm",
            lambda: cuda_primitives.cuda_rms_norm(
                snapshot.q_heads, fixture.q_norm_weight, RMS_EPS
            ),
        ),
        "k_rmsnorm": (
            "cuda_primitives.cuda_rms_norm",
            lambda: cuda_primitives.cuda_rms_norm(
                snapshot.k_heads, fixture.k_norm_weight, RMS_EPS
            ),
        ),
        "q_rope": (
            "cuda_primitives.cuda_apply_rope",
            lambda: cuda_primitives.cuda_apply_rope(
                snapshot.q_norm, fixture.case.past_length, ROPE_THETA
            ),
        ),
        "k_rope": (
            "cuda_primitives.cuda_apply_rope",
            lambda: cuda_primitives.cuda_apply_rope(
                snapshot.k_norm, fixture.case.past_length, ROPE_THETA
            ),
        ),
        "kv_cache_append": (
            "cuda_primitives.cuda_kv_cache_append_",
            lambda: cuda_primitives.cuda_kv_cache_append_(
                fixture.k_cache,
                fixture.v_cache,
                snapshot.k_rope,
                snapshot.v_heads,
                fixture.case.past_length,
            ),
        ),
        "cached_gqa": (
            "cuda_primitives.cuda_gqa_attention_cached",
            lambda: cuda_primitives.cuda_gqa_attention_cached(
                snapshot.q_rope,
                fixture.k_cache,
                fixture.v_cache,
                fixture.case.past_length,
            ),
        ),
        "output_projection": (
            "cuda_primitives.cuda_w4a16_linear",
            lambda: cuda_primitives.cuda_w4a16_linear(
                snapshot.context_flat, fixture.weights["out_weight"]
            ),
        ),
    }


def _benchmark_isolated_stages(
    fixture: CaseFixture,
    *,
    warmups: int,
    rounds: int,
    samples_per_round: int,
) -> dict[str, Any]:
    _restore_current_slot(fixture)
    snapshot = _prepare_stage_snapshot(fixture)
    operations = _stage_operations(fixture, snapshot)
    if tuple(operations) != STAGE_NAMES:
        raise AssertionError("isolated stage order differs from the M6 contract")

    stage_results: dict[str, Any] = {}
    for stage_name, (public_primitive, operation) in operations.items():
        stage_rounds: list[dict[str, Any]] = []
        for round_index in range(1, rounds + 1):
            _warm_operation(operation, warmups)
            samples = _event_samples(operation, samples_per_round)
            stage_rounds.append(
                {
                    "round": round_index,
                    "statistics": _summarize(samples),
                    "raw_samples_us": [round(value, 3) for value in samples],
                }
            )
        aggregate = _round_aggregate(stage_rounds)
        stage_results[stage_name] = {
            "public_primitive": public_primitive,
            "input_prepared_outside_timing": True,
            "warmups_before_each_round": warmups,
            "rounds": stage_rounds,
            "aggregate": aggregate,
            "outlier_policy": "all raw samples retained; no filtering",
        }
        print(
            f"stage {fixture.case.case_id} {stage_name} "
            f"median={aggregate['median_of_round_medians_us']:.3f}us",
            flush=True,
        )

    stage_medians = {
        name: float(result["aggregate"]["median_of_round_medians_us"])
        for name, result in stage_results.items()
    }
    isolated_sum = sum(stage_medians.values())
    category_results: dict[str, Any] = {}
    for category, stage_names in CATEGORY_STAGES.items():
        latency = sum(stage_medians[name] for name in stage_names)
        category_results[category] = {
            "stages": list(stage_names),
            "absolute_latency_us": round(latency, 3),
            "fraction_of_isolated_stage_sum": round(latency / isolated_sum, 6),
        }

    largest_stage = max(stage_medians, key=stage_medians.__getitem__)
    largest_category = max(
        category_results,
        key=lambda name: category_results[name]["absolute_latency_us"],
    )
    return {
        "method": (
            "Each logical stage is called independently with an untimed frozen "
            "stage-state input prepared by existing public primitives."
        ),
        "stages": stage_results,
        "metadata_only_operations": {
            "q_k_v_head_reshapes": {
                "timed_cuda_stage": False,
                "source_derived_kernel_launch_count": 0,
                "attributed_cuda_stage_latency_us": 0.0,
                "classification": "metadata-only reshape views",
            },
            "context_flatten": {
                "timed_cuda_stage": False,
                "source_derived_kernel_launch_count": 0,
                "attributed_cuda_stage_latency_us": 0.0,
                "classification": "metadata-only reshape view",
            },
        },
        "isolated_stage_sum_us": round(isolated_sum, 3),
        "category_attribution": category_results,
        "largest_measured_logical_stage": {
            "stage": largest_stage,
            "latency_us": stage_medians[largest_stage],
        },
        "largest_isolated_stage_latency_category": {
            "category": largest_category,
            "latency_us": category_results[largest_category][
                "absolute_latency_us"
            ],
        },
        "non_additivity_caveat": (
            "The isolated stage sum is a ranking and decomposition diagnostic. "
            "It is not forced to equal end-to-end latency because allocation "
            "reuse, Python submission timing, event placement, inter-stage gaps, "
            "isolated execution context, and cache/thermal state differ."
        ),
    }


def _allocator_peak(fixture: CaseFixture) -> dict[str, Any]:
    _restore_current_slot(fixture)
    warm_output = _public_pipeline_call(fixture)
    torch.cuda.synchronize()
    del warm_output
    gc.collect()
    torch.cuda.synchronize()
    _restore_current_slot(fixture)
    torch.cuda.synchronize()

    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    retained_output = _public_pipeline_call(fixture)
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    current_allocated_after = torch.cuda.memory_allocated()
    current_reserved_after = torch.cuda.memory_reserved()
    result = {
        "classification": "incremental PyTorch CUDA caching-allocator metric",
        "persistent_allocated_bytes_before_call": baseline_allocated,
        "persistent_reserved_bytes_before_call": baseline_reserved,
        "maximum_allocated_bytes_during_call": peak_allocated,
        "maximum_reserved_bytes_during_call": peak_reserved,
        "incremental_peak_allocated_bytes": peak_allocated - baseline_allocated,
        "current_allocated_bytes_after_call_with_output_retained": (
            current_allocated_after
        ),
        "current_reserved_bytes_after_call_with_output_retained": (
            current_reserved_after
        ),
        "returned_output_retained_through_measurement": True,
        "interpretation": (
            "This is allocated memory tracked by PyTorch's caching allocator "
            "above the already-created persistent fixture. It is not physical "
            "VRAM traffic, bandwidth, or a count of physical cudaMalloc calls."
        ),
    }
    del retained_output
    torch.cuda.synchronize()
    return result


def _analytical_memory(case: BenchmarkCase) -> dict[str, int | str]:
    score_bytes = (
        case.batch_size
        * QUERY_HEADS
        * TOKEN_COUNT
        * case.logical_context_length
        * torch.tensor([], dtype=torch.float32).element_size()
    )
    probability_bytes = score_bytes
    cache_bytes = (
        2
        * case.batch_size
        * KV_HEADS
        * CACHE_CAPACITY
        * HEAD_DIM
        * torch.tensor([], dtype=torch.bfloat16).element_size()
    )
    return {
        "scores_bytes": score_bytes,
        "probabilities_bytes": probability_bytes,
        "attention_materialization_bytes": score_bytes + probability_bytes,
        "persistent_kv_cache_bytes": cache_bytes,
        "scores_formula": "B * Hq * T * S * sizeof(float)",
        "probabilities_formula": "B * Hq * T * S * sizeof(float)",
        "persistent_kv_cache_formula": (
            "2 caches * B * Hkv * C * D * sizeof(BF16)"
        ),
    }


def _source_launch_accounting() -> dict[str, Any]:
    stages = {
        "input_rmsnorm": {
            "count": 1,
            "kernel": "rms_norm_kernel",
            "source": "src/rmsnorm.cu",
        },
        "q_k_v_projections": {
            "count": 3,
            "kernel": "w4a16_linear_kernel",
            "source": "src/w4a16.cu",
        },
        "q_k_rmsnorm": {
            "count": 2,
            "kernel": "rms_norm_kernel",
            "source": "src/rmsnorm.cu",
        },
        "q_k_rope": {
            "count": 2,
            "kernel": "apply_rope_kernel",
            "source": "src/rope.cu",
        },
        "kv_cache_append": {
            "count": 1,
            "kernel": "kv_cache_append_kernel",
            "source": "src/kv_cache.cu",
        },
        "cached_gqa_qk_softmax_pv": {
            "count": 3,
            "kernels": [
                "qk_scores_cached_kernel",
                "softmax_cached_kernel",
                "pv_context_cached_kernel",
            ],
            "source": "src/gqa_attention_cached.cu",
        },
        "output_projection": {
            "count": 1,
            "kernel": "w4a16_linear_kernel",
            "source": "src/w4a16.cu",
        },
    }
    return {
        "classification": "source-derived launch count",
        "production_operation": (
            "decoder_attention_cuda.cuda_decoder_attention_forward_"
        ),
        "projection_backend": "cuda_w4a16_linear (frozen baseline)",
        "stage_structure": stages,
        "total_cuda_kernel_launches_per_call": sum(
            int(stage["count"]) for stage in stages.values()
        ),
        "metadata_only_reshape_launches": 0,
    }


def _allocation_accounting(master: MasterFixture) -> dict[str, Any]:
    weight_parts = {
        name: _weight_storage_bytes(weight)
        for name, weight in master.weights.items()
    }
    total_weight_bytes = sum(
        int(parts["total_bytes"]) for parts in weight_parts.values()
    )
    return {
        "classification": "source-derived logical allocation structure",
        "persistent_inputs_and_state": {
            "x": "BF16 [B,1,3072] (B=1 uses batch 0 of the B=2 master input)",
            "normalization_weights": (
                "BF16 input gamma [3072], Q gamma [128], K gamma [128]"
            ),
            "packed_nvfp4_projection_weights": {
                "per_weight": weight_parts,
                "shared_total_bytes": total_weight_bytes,
            },
            "k_v_cache": "two persistent BF16 [B,6,16384,128] tensors",
        },
        "pipeline_outputs_and_intermediates": [
            {
                "name": "input RMSNorm output",
                "dtype_shape": "BF16 [B,1,3072]",
                "allocation": True,
            },
            {
                "name": "Q/K/V projection outputs",
                "dtype_shape": (
                    "BF16 [B,1,3072], [B,1,768], [B,1,768]"
                ),
                "allocation": True,
            },
            {
                "name": "Q/K/V head reshapes",
                "dtype_shape": (
                    "BF16 [B,1,24,128], [B,1,6,128], [B,1,6,128]"
                ),
                "allocation": False,
                "classification": "metadata-only views",
            },
            {
                "name": "Q/K per-head RMSNorm outputs",
                "dtype_shape": "BF16 [B,1,24,128], [B,1,6,128]",
                "allocation": True,
            },
            {
                "name": "Q/K RoPE outputs",
                "dtype_shape": "BF16 [B,1,24,128], [B,1,6,128]",
                "allocation": True,
            },
            {
                "name": "KV append",
                "dtype_shape": "in-place writes to persistent cache slot P",
                "allocation": False,
            },
            {
                "name": "cached-GQA scores",
                "dtype_shape": "FP32 [B,24,1,S]",
                "allocation": True,
            },
            {
                "name": "cached-GQA probabilities",
                "dtype_shape": "FP32 [B,24,1,S]",
                "allocation": True,
            },
            {
                "name": "cached-GQA context",
                "dtype_shape": "BF16 [B,1,24,128]",
                "allocation": True,
            },
            {
                "name": "context flatten",
                "dtype_shape": "BF16 [B,1,3072]",
                "allocation": False,
                "classification": "metadata-only view",
            },
            {
                "name": "final output",
                "dtype_shape": "BF16 [B,1,3072]",
                "allocation": True,
            },
        ],
        "allocator_caveat": (
            "Source-level tensor allocations are not physical cudaMalloc counts; "
            "PyTorch uses a CUDA caching allocator and can reuse blocks."
        ),
    }


def _dynamic_launch_validation(
    master: MasterFixture,
    *,
    skip: bool,
) -> dict[str, Any]:
    if skip:
        return {
            "status": "unavailable",
            "reason": "disabled by --skip-dynamic-launch-validation",
        }
    case = BenchmarkCase(1, 128)
    fixture = _make_case_fixture(master, case)
    try:
        from torch.profiler import ProfilerActivity, profile

        _warm_operation(lambda: _public_pipeline_call(fixture), 1)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            output = _public_pipeline_call(fixture)
            torch.cuda.synchronize()
        del output
        cuda_names = [
            event.name
            for event in profiler.events()
            if "cuda" in str(event.device_type).lower()
        ]
        if not cuda_names:
            return {
                "status": "unavailable",
                "tool": "torch.profiler with CUDA activity",
                "case": case.as_dict(),
                "reason": (
                    "The profiler context completed but emitted no CUDA "
                    "activity records in this WSL environment."
                ),
                "all_cuda_activity_count": 0,
                "timings_used_as_primary_latency": False,
            }
        expected = {
            "rms_norm_kernel": 3,
            "w4a16_linear_kernel": 4,
            "apply_rope_kernel": 2,
            "kv_cache_append_kernel": 1,
            "qk_scores_cached_kernel": 1,
            "softmax_cached_kernel": 1,
            "pv_context_cached_kernel": 1,
        }
        observed = {
            marker: sum(marker in name for name in cuda_names)
            for marker in expected
        }
        matched_count = sum(observed.values())
        unexpected = [
            name
            for name in cuda_names
            if not any(marker in name for marker in expected)
        ]
        validated = observed == expected
        return {
            "status": "validated" if validated else "diagnostic_mismatch",
            "tool": "torch.profiler with CUDA activity",
            "case": case.as_dict(),
            "expected_kernel_name_counts": expected,
            "observed_kernel_name_counts": observed,
            "matched_production_kernel_launch_count": matched_count,
            "all_cuda_activity_count": len(cuda_names),
            "cuda_activity_names": cuda_names,
            "unmatched_cuda_activity_names": unexpected,
            "timings_used_as_primary_latency": False,
            "interpretation": (
                "Diagnostic dynamic validation of launch count and broad names; "
                "profiler timing does not replace CUDA-event latency."
            ),
        }
    except Exception as error:  # pragma: no cover - profiler availability varies
        return {
            "status": "unavailable",
            "reason": f"{type(error).__name__}: {error}",
            "timings_used_as_primary_latency": False,
        }
    finally:
        del fixture
        gc.collect()
        torch.cuda.synchronize()


def _case_stage_median(case_result: dict[str, Any], stage: str) -> float:
    return float(
        case_result["isolated_stage_attribution"]["stages"][stage]["aggregate"][
            "median_of_round_medians_us"
        ]
    )


def _scaling_analysis(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (int(result["case"]["B"]), int(result["case"]["P"])): result
        for result in case_results
    }
    p_scaling: dict[str, list[dict[str, Any]]] = {}
    for batch_size in BATCH_SIZES:
        rows: list[dict[str, Any]] = []
        base_e2e = float(
            by_key[(batch_size, 0)]["end_to_end_stream_elapsed_us"]["aggregate"][
                "median_of_round_medians_us"
            ]
        )
        for past_length in PAST_LENGTHS:
            result = by_key[(batch_size, past_length)]
            e2e = float(
                result["end_to_end_stream_elapsed_us"]["aggregate"][
                    "median_of_round_medians_us"
                ]
            )
            category = result["isolated_stage_attribution"]["category_attribution"]
            rows.append(
                {
                    "P": past_length,
                    "S": past_length + TOKEN_COUNT,
                    "end_to_end_median_us": e2e,
                    "end_to_end_ratio_vs_p0": round(e2e / base_e2e, 6),
                    "attention_isolated_median_us": _case_stage_median(
                        result, "cached_gqa"
                    ),
                    "projection_category_us": category["projection"][
                        "absolute_latency_us"
                    ],
                    "attention_fraction_of_isolated_stage_sum": category[
                        "attention"
                    ]["fraction_of_isolated_stage_sum"],
                    "attention_materialization_bytes": result[
                        "analytical_memory"
                    ]["attention_materialization_bytes"],
                }
            )
        p_scaling[f"B={batch_size}"] = rows

    b_scaling: list[dict[str, Any]] = []
    for past_length in PAST_LENGTHS:
        b1 = by_key[(1, past_length)]
        b2 = by_key[(2, past_length)]
        b1_e2e = float(
            b1["end_to_end_stream_elapsed_us"]["aggregate"][
                "median_of_round_medians_us"
            ]
        )
        b2_e2e = float(
            b2["end_to_end_stream_elapsed_us"]["aggregate"][
                "median_of_round_medians_us"
            ]
        )
        b1_categories = b1["isolated_stage_attribution"]["category_attribution"]
        b2_categories = b2["isolated_stage_attribution"]["category_attribution"]
        b1_attention = _case_stage_median(b1, "cached_gqa")
        b2_attention = _case_stage_median(b2, "cached_gqa")
        b_scaling.append(
            {
                "P": past_length,
                "end_to_end_b2_over_b1": round(b2_e2e / b1_e2e, 6),
                "projection_category_b2_over_b1": round(
                    float(b2_categories["projection"]["absolute_latency_us"])
                    / float(b1_categories["projection"]["absolute_latency_us"]),
                    6,
                ),
                "cached_gqa_b2_over_b1": round(b2_attention / b1_attention, 6),
                "temporary_attention_bytes_b2_over_b1": round(
                    int(b2["analytical_memory"]["attention_materialization_bytes"])
                    / int(b1["analytical_memory"]["attention_materialization_bytes"]),
                    6,
                ),
                "persistent_cache_bytes_b2_over_b1": round(
                    int(b2["analytical_memory"]["persistent_kv_cache_bytes"])
                    / int(b1["analytical_memory"]["persistent_kv_cache_bytes"]),
                    6,
                ),
            }
        )
    return {"p_scaling": p_scaling, "b_scaling": b_scaling}


def _next_milestone_recommendation(
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (int(result["case"]["B"]), int(result["case"]["P"])): result
        for result in case_results
    }
    long_context = [by_key[(batch_size, 8_192)] for batch_size in BATCH_SIZES]
    attention_dominates_long = all(
        result["isolated_stage_attribution"][
            "largest_isolated_stage_latency_category"
        ]["category"]
        == "attention"
        for result in long_context
    )
    if attention_dominates_long:
        recommendation_class = "attention_algorithm_and_materialization_optimization"
        recommendation = (
            "For M7, investigate an isolated same-semantics cached-attention "
            "algorithm/materialization optimization, with the first goal of "
            "reducing or eliminating the separate FP32 score and probability "
            "materializations. Keep projection-backend promotion as a separate "
            "future end-to-end A/B because short-context regimes may still be "
            "projection-led."
        )
    else:
        dominant_categories = [
            result["isolated_stage_attribution"][
                "largest_isolated_stage_latency_category"
            ]["category"]
            for result in case_results
        ]
        most_common = statistics.mode(dominant_categories)
        if most_common == "projection":
            recommendation_class = "projection_backend_end_to_end_ab"
            recommendation = (
                "For M7, run a controlled same-fixture end-to-end projection-"
                "backend A/B before any promotion. M6 does not substitute the "
                "grouped-decode candidate or estimate a pipeline speedup."
            )
        else:
            recommendation_class = "launch_count_and_fusion_investigation"
            recommendation = (
                "For M7, investigate launch-count reduction with an isolated "
                "same-semantics experiment, preserving the frozen M6 baseline."
            )

    evidence = []
    for result in case_results:
        case = result["case"]
        category = result["isolated_stage_attribution"]["category_attribution"]
        evidence.append(
            {
                "case_id": case["case_id"],
                "largest_category": result["isolated_stage_attribution"][
                    "largest_isolated_stage_latency_category"
                ]["category"],
                "projection_us": category["projection"]["absolute_latency_us"],
                "attention_us": category["attention"]["absolute_latency_us"],
                "attention_fraction": category["attention"][
                    "fraction_of_isolated_stage_sum"
                ],
                "attention_materialization_bytes": result["analytical_memory"][
                    "attention_materialization_bytes"
                ],
            }
        )
    return {
        "recommendation_class": recommendation_class,
        "recommendation": recommendation,
        "evidence": evidence,
        "implemented_in_m6": False,
        "boundary": (
            "Stage attribution supports relative prioritization only. No DRAM-, "
            "L2-, compute-, instruction-, SFU-, or occupancy-bound cause is "
            "asserted without hardware counters."
        ),
    }


def _all_finite_nonnegative(values: list[float]) -> bool:
    return all(math.isfinite(value) and value >= 0.0 for value in values)


def _sanity_checks(document: dict[str, Any], args: argparse.Namespace) -> dict[str, bool]:
    cases = document["cases"]
    expected_ids = {case.case_id for case in PRIMARY_CASES}
    actual_ids = {result["case"]["case_id"] for result in cases}
    raw_counts_ok = True
    finite_ok = True
    stages_ok = True
    for result in cases:
        e2e_rounds = result["end_to_end_stream_elapsed_us"]["rounds"]
        raw_counts_ok &= len(e2e_rounds) == args.end_to_end_rounds
        for round_result in e2e_rounds:
            raw = round_result["raw_samples_us"]
            raw_counts_ok &= len(raw) == args.end_to_end_samples
            finite_ok &= _all_finite_nonnegative(raw)
        wall_raw = result["synchronized_wall_us"]["raw_samples_us"]
        raw_counts_ok &= len(wall_raw) == args.wall_samples
        finite_ok &= _all_finite_nonnegative(wall_raw)
        stage_results = result["isolated_stage_attribution"]["stages"]
        stages_ok &= tuple(stage_results) == STAGE_NAMES
        for stage in stage_results.values():
            raw_counts_ok &= len(stage["rounds"]) == args.stage_rounds
            for round_result in stage["rounds"]:
                raw = round_result["raw_samples_us"]
                raw_counts_ok &= len(raw) == args.stage_samples
                finite_ok &= _all_finite_nonnegative(raw)

    checks = {
        "all_ten_primary_cases_exist": actual_ids == expected_ids and len(cases) == 10,
        "required_raw_sample_counts_present": raw_counts_ok,
        "all_latency_samples_finite_and_nonnegative": finite_ok,
        "raw_samples_unfiltered": True,
        "case_labels_match_actual_fixture": all(
            result["case"]["S"] == result["case"]["P"] + 1 for result in cases
        ),
        "capacity_is_16384_for_every_case": all(
            result["case"]["C"] == CACHE_CAPACITY for result in cases
        ),
        "representative_weights_shared_across_cases": True,
        "public_production_pipeline_used_for_end_to_end": all(
            result["end_to_end_stream_elapsed_us"]["operation"]
            == "decoder_attention_cuda.cuda_decoder_attention_forward_"
            for result in cases
        ),
        "baseline_w4a16_used_for_all_projections": all(
            "cuda_w4a16_linear"
            in result["end_to_end_stream_elapsed_us"]["projection_backend"]
            for result in cases
        ),
        "eleven_isolated_stages_use_existing_public_primitives": stages_ok,
        "cache_reset_or_copy_excluded_from_timing": True,
        "quantization_and_setup_excluded_from_timing": True,
        "memory_metrics_not_labeled_bandwidth": True,
        "launch_count_labeled_source_derived": (
            document["launch_accounting"]["classification"]
            == "source-derived launch count"
        ),
        "correctness_precheck_passed_all_cases": all(
            result["correctness_precheck"]["passed"] for result in cases
        ),
        "representative_reference_policy_passed": document[
            "representative_correctness_reference"
        ]["passed_frozen_elementwise_policy"],
        "frozen_artifacts_match_head_at_capture": document[
            "frozen_artifact_audit"
        ]["all_match_head"],
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"M6 benchmark sanity checks failed: {failures}")
    return checks


def _fixture_metadata(master: MasterFixture) -> dict[str, Any]:
    return {
        "master_batch_size": max(BATCH_SIZES),
        "hidden_states": {
            "shape": list(master.x_master.shape),
            "dtype": str(master.x_master.dtype),
            "distribution": "deterministic BF16 Normal(0,1)",
            "seed": FIXTURE_SEEDS["hidden_states"],
            "b1_policy": "batch 0 view of the same B=2 master hidden-state fixture",
        },
        "normalization_weights": {
            "rule": "BF16 gamma = 1 + Normal(0, 0.02)",
            "input_norm_seed": FIXTURE_SEEDS["input_norm_weight"],
            "q_norm_seed": FIXTURE_SEEDS["q_norm_weight"],
            "k_norm_seed": FIXTURE_SEEDS["k_norm_weight"],
            "shared_across_all_cases": True,
        },
        "master_cache": {
            "shape": [max(BATCH_SIZES), KV_HEADS, CACHE_CAPACITY, HEAD_DIM],
            "dtype": "torch.bfloat16",
            "k_distribution": "deterministic BF16 Normal(0,0.5)",
            "v_distribution": "deterministic BF16 Normal(0,0.75)",
            "k_seed": FIXTURE_SEEDS["master_k_cache"],
            "v_seed": FIXTURE_SEEDS["master_v_cache"],
            "b1_policy": "batch 0 of the same B=2 CPU master cache",
            "unused_suffix": (
                "populated with deterministic finite values; attention uses only S"
            ),
            "fresh_case_policy": (
                "Each correctness or timing case starts from the corresponding "
                "B slice of the same master cache. Before each benchmark round, "
                "slot P is restored outside timing; the prefix is never changed "
                "by repeated T=1 calls."
            ),
        },
        "fixed_capacity_reason": (
            "C=16384 is held constant so changing P changes only logical S=P+1, "
            "not the physical K/V batch-head stride or cache allocation size."
        ),
    }


def _methodology(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "primary_end_to_end": {
            "metric": "end_to_end_stream_elapsed_us",
            "operation": "exactly one cuda_decoder_attention_forward_ call per sample",
            "stream": "PyTorch current CUDA stream",
            "timer": "torch.cuda.Event",
            "rounds": args.end_to_end_rounds,
            "warmups_before_each_round": args.warmups,
            "raw_samples_per_round": args.end_to_end_samples,
            "completion_synchronization": (
                "one synchronization after warmup and one on the final queued "
                "end event per round; elapsed samples extracted afterward"
            ),
        },
        "synchronized_wall": {
            "metric": "synchronized_wall_us",
            "timer": "time.perf_counter_ns",
            "warmups": args.warmups,
            "samples": args.wall_samples,
            "sample_boundary": (
                "synchronize, start timer, one public call, synchronize, stop timer"
            ),
        },
        "isolated_stages": {
            "timer": "torch.cuda.Event",
            "logical_stage_count": len(STAGE_NAMES),
            "stage_names": list(STAGE_NAMES),
            "rounds": args.stage_rounds,
            "warmups_before_each_round": args.warmups,
            "raw_samples_per_round": args.stage_samples,
            "stage_input_policy": (
                "all prerequisite outputs are prepared before the isolated "
                "stage's timed region using existing public primitives"
            ),
            "metadata_views": (
                "head reshapes and context flatten are untimed zero-kernel views"
            ),
        },
        "statistics": {
            "per_round": [
                "median",
                "mean",
                "minimum",
                "p95",
                "population standard deviation",
                "raw samples",
            ],
            "p95_definition": (
                "linear interpolation at rank (sample_count - 1) * 0.95"
            ),
            "across_rounds": (
                "median/minimum/maximum of round medians plus max/min ratio"
            ),
            "outlier_policy": "retain every sample; no filtering or rerun selection",
        },
        "excluded_from_all_timing": [
            "extension loading and compilation",
            "weight source generation",
            "quantize_nvfp4_reference",
            "weight representation statistics",
            "host-to-device fixture transfer",
            "random fixture construction",
            "correctness prechecks and numerical reference",
            "cache establishment or restoration",
            "stage-state preparation",
            "environment and source inspection",
            "JSON serialization and printing",
        ],
        "repeated_cache_mutation": (
            "At fixed P, every T=1 invocation recomputes K/V and overwrites the "
            "same slot P. No full cache reset, clone, or copy occurs in a timed "
            "region; every invocation sees the same logical prefix and x."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=25)
    parser.add_argument("--end-to-end-rounds", type=int, default=5)
    parser.add_argument("--end-to-end-samples", type=int, default=100)
    parser.add_argument("--wall-samples", type=int, default=30)
    parser.add_argument("--stage-rounds", type=int, default=3)
    parser.add_argument("--stage-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--skip-dynamic-launch-validation", action="store_true")
    args = parser.parse_args()
    if args.warmups < 25:
        parser.error("--warmups must be at least 25 for M6")
    if args.end_to_end_rounds < 5:
        parser.error("--end-to-end-rounds must be at least 5 for M6")
    if args.end_to_end_samples < 100:
        parser.error("--end-to-end-samples must be at least 100 for M6")
    if args.wall_samples < 30:
        parser.error("--wall-samples must be at least 30 for M6")
    if args.stage_rounds < 3:
        parser.error("--stage-rounds must be at least 3 for M6")
    if args.stage_samples < 50:
        parser.error("--stage-samples must be at least 50 for M6")
    return args


def main() -> None:
    args = _parse_args()
    if args.list_cases:
        for case in PRIMARY_CASES:
            print(case.case_id)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the CUDA device does not support BF16")

    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"capability={torch.cuda.get_device_capability(0)} cases={len(PRIMARY_CASES)}",
        flush=True,
    )
    environment = _environment()
    frozen_audit = _frozen_artifact_audit()
    if not frozen_audit["all_match_head"]:
        raise RuntimeError("a frozen production/test/prior-result artifact differs from HEAD")

    with torch.inference_mode():
        master = _prepare_master_fixture()
        prechecks, reference_metrics = _run_all_correctness_prechecks(master)
        case_results: list[dict[str, Any]] = []
        for case in PRIMARY_CASES:
            fixture = _make_case_fixture(master, case)
            allocator = _allocator_peak(fixture)
            end_to_end = _benchmark_end_to_end(
                fixture,
                warmups=args.warmups,
                rounds=args.end_to_end_rounds,
                samples_per_round=args.end_to_end_samples,
            )
            wall = _benchmark_wall(
                fixture,
                warmups=args.warmups,
                samples=args.wall_samples,
            )
            stages = _benchmark_isolated_stages(
                fixture,
                warmups=args.warmups,
                rounds=args.stage_rounds,
                samples_per_round=args.stage_samples,
            )
            e2e_median = float(
                end_to_end["aggregate"]["median_of_round_medians_us"]
            )
            stages["isolated_stage_sum_over_end_to_end_stream_elapsed"] = round(
                float(stages["isolated_stage_sum_us"]) / e2e_median,
                6,
            )
            case_results.append(
                {
                    "case": case.as_dict(),
                    "correctness_precheck": prechecks[case.case_id],
                    "end_to_end_stream_elapsed_us": end_to_end,
                    "synchronized_wall_us": wall,
                    "isolated_stage_attribution": stages,
                    "analytical_memory": _analytical_memory(case),
                    "pytorch_allocator_memory": allocator,
                }
            )
            print(
                f"completed {case.case_id}: e2e={e2e_median:.3f}us "
                f"wall={wall['statistics']['median_us']:.3f}us",
                flush=True,
            )
            del fixture, stages, end_to_end, wall, allocator
            gc.collect()
            torch.cuda.synchronize()

        dynamic_launch_validation = _dynamic_launch_validation(
            master,
            skip=args.skip_dynamic_launch_validation,
        )
        scaling = _scaling_analysis(case_results)
        recommendation = _next_milestone_recommendation(case_results)
        environment["gpu"]["nvidia_smi_after_benchmark"] = _nvidia_smi_snapshot()
        document: dict[str, Any] = {
            "schema_version": "m6_decoder_pipeline_baseline.v1",
            "milestone": "M6",
            "benchmark_commit": environment["repository"]["head_sha"],
            "environment": environment,
            "methodology": _methodology(args),
            "fixed_dimensions": {
                "H": HIDDEN_SIZE,
                "Hq": QUERY_HEADS,
                "Hkv": KV_HEADS,
                "D": HEAD_DIM,
                "T": TOKEN_COUNT,
                "C": CACHE_CAPACITY,
            },
            "seeds": {
                "weights": WEIGHT_SEEDS,
                "fixture": FIXTURE_SEEDS,
            },
            "representative_weight_generation": {
                "description": (
                    "Four independent deterministic dense FP32 normal source "
                    "matrices, quantized by the frozen quantize_nvfp4_reference. "
                    "They are representative random weights, not trained weights."
                ),
                "source_standard_deviation_rule": "1 / sqrt(H)",
                "source_standard_deviation_value": SOURCE_STD,
                "source_generation_device": "CPU",
                "quantization_device": "CPU",
                "storage_transfer": "portable NVFP4 storage copied to CUDA after quantization",
                "setup_excluded_from_timing": True,
                "dense_random_not_one_hot": True,
                "uniform_scales_manufactured": False,
            },
            "nvfp4_representation_statistics": master.weight_statistics,
            "fixture": _fixture_metadata(master),
            "primary_matrix": [case.as_dict() for case in PRIMARY_CASES],
            "cases": case_results,
            "representative_correctness_reference": reference_metrics,
            "launch_accounting": _source_launch_accounting(),
            "dynamic_launch_validation": dynamic_launch_validation,
            "allocation_accounting": _allocation_accounting(master),
            "scaling_analysis": scaling,
            "next_milestone_recommendation": recommendation,
            "limitations": [
                "One RTX 4080 Laptop GPU and one captured run under uncontrolled consumer-laptop clocks, thermals, and power.",
                "CUDA-event stream intervals can include stream-visible idle gaps from Python submission; they are not sums of pure kernel execution time.",
                "Synchronized wall latency includes synchronization and allocator/orchestration effects; wall-minus-stream is not isolated host overhead.",
                "Isolated-stage medians are non-additive and are used only for ranking and decomposition.",
                "The PyTorch allocator peak is not physical VRAM traffic, bandwidth, or a cudaMalloc count.",
                "Analytical logical bytes are not measured memory traffic or DRAM bandwidth.",
                "No Nsight Compute hardware counters were collected; no DRAM-, L2-, compute-, instruction-, SFU-, or occupancy-bound claim is made.",
                "The deterministic dense random weights are representative fixtures, not trained-model weights.",
                "The caching allocator and CUDA path are warmed; this is a steady-state modular-path baseline rather than cold-start latency.",
                "Cases run in the documented fixed B-major, ascending-P order; all samples and timing-level changes are retained.",
                "No comparison to native Blackwell FP4, FlashAttention, cuBLAS, TensorRT-LLM, or another model/server is made because semantics are not matched.",
            ],
            "frozen_artifact_audit": frozen_audit,
        }
        document["sanity_checks"] = _sanity_checks(document, args)

    output_path = args.output.resolve()
    if output_path in {
        (REPOSITORY_ROOT / path).resolve() for path in PRIOR_RESULT_PATHS
    }:
        raise ValueError("M6 output must not overwrite an M4 benchmark result")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".bench.tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(
        f"dynamic_launch_validation={dynamic_launch_validation['status']}",
        flush=True,
    )
    print(f"wrote={output_path}", flush=True)


if __name__ == "__main__":
    main()
