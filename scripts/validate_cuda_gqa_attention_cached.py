"""Exercise capacity-aware causal GQA under CUDA validation tools."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cuda_primitives  # noqa: E402
from reference import gqa_attention_reference  # noqa: E402


ACCEPTED_MAXIMUM_ABSOLUTE_ERROR = 2.0**-7
ACCEPTED_MEAN_ABSOLUTE_ERROR = 2.0**-16
ADJACENCY_ABSOLUTE_FLOOR = 2.0**-20

CASES = (
    # label, B, T, Hq, Hkv, D, P, C, seed
    ("b2-t3-ratio4-spare", 2, 3, 8, 2, 32, 5, 257, 100_003),
    ("p128-t2-d128-c8192", 1, 2, 8, 2, 128, 128, 8192, 100_019),
    ("canonical-p128-c8192", 1, 1, 24, 6, 128, 128, 8192, 100_043),
)


def _ordered_keys(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    magnitude = bits & 0x7FFF
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def _check(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float, int, int]:
    error = (actual.float() - expected.float()).abs()
    adjacency = (_ordered_keys(actual) - _ordered_keys(expected)).abs()
    maximum = float(error.max().item())
    mean = float(error.mean().item())
    maximum_distance = int(adjacency.max().item())
    exact = int(torch.count_nonzero(actual == expected).item())
    print(
        f"case={label} shape={tuple(actual.shape)} max_abs={maximum:.9g} "
        f"mean_abs={mean:.9g} exact_fraction={exact / actual.numel():.9g} "
        f"max_bf16_distance={maximum_distance}"
    )
    if maximum > ACCEPTED_MAXIMUM_ABSOLUTE_ERROR:
        raise AssertionError(f"{label} maximum absolute error was {maximum}")
    if mean > ACCEPTED_MEAN_ABSOLUTE_ERROR:
        raise AssertionError(f"{label} mean absolute error was {mean}")
    if not torch.all(
        (adjacency <= 1) | (error <= ADJACENCY_ABSOLUTE_FLOOR)
    ).item():
        raise AssertionError(f"{label} exceeded the M5A BF16 adjacency policy")
    return maximum, mean, exact, actual.numel()


def _run_generated(
    label: str,
    batch_size: int,
    token_count: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    past_length: int,
    capacity: int,
    seed: int,
) -> tuple[float, float, int, int]:
    logical_length = past_length + token_count
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = (
        torch.randn(
            (batch_size, token_count, query_heads, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16).cuda()
    k_cache = torch.full(
        (batch_size, kv_heads, capacity, head_dim),
        71.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.full_like(k_cache, -73.0)
    k_prefix = (
        torch.randn(
            (batch_size, kv_heads, logical_length, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16).cuda()
    v_prefix = (
        torch.randn(
            (batch_size, kv_heads, logical_length, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16).cuda()
    k_cache[:, :, :logical_length].copy_(k_prefix)
    v_cache[:, :, :logical_length].copy_(v_prefix)

    # The compact tensors exist only in this independent validation oracle.
    compact_k = k_cache[:, :, :logical_length].clone()
    compact_v = v_cache[:, :, :logical_length].clone()
    _, _, expected = gqa_attention_reference(
        q,
        compact_k,
        compact_v,
        past_length,
        return_attention=False,
    )
    compact_actual = cuda_primitives.cuda_gqa_attention(
        q,
        compact_k,
        compact_v,
        past_length,
    )
    actual = cuda_primitives.cuda_gqa_attention_cached(
        q,
        k_cache,
        v_cache,
        past_length,
    )
    if not torch.equal(actual, compact_actual):
        raise AssertionError(f"{label} differed from frozen compact M5A")
    return _check(label, actual, expected)


def _run_hand_stride_case() -> tuple[float, float, int, int]:
    q = torch.zeros((1, 1, 4, 8), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros((1, 2, 7, 8), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.full_like(k_cache, -12_000.0)
    v_cache[0, 0, 0].fill_(2.0)
    v_cache[0, 0, 1].fill_(4.0)
    v_cache[0, 1, 0].fill_(20.0)
    v_cache[0, 1, 1].fill_(40.0)
    expected = torch.empty_like(q)
    expected[0, 0, 0:2].fill_(3.0)
    expected[0, 0, 2:4].fill_(30.0)
    actual = cuda_primitives.cuda_gqa_attention_cached(q, k_cache, v_cache, 1)
    if not torch.equal(actual, expected):
        raise AssertionError("hand C-versus-S physical stride case was not exact")
    return _check("hand-c7-s2-hkv2", actual, expected)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the CUDA device does not support BF16")

    print(f"device={torch.cuda.get_device_name()}")
    print(f"capability={torch.cuda.get_device_capability()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")
    results = [_run_hand_stride_case()]
    results.extend(_run_generated(*case) for case in CASES)
    torch.cuda.synchronize()

    maximum = max(result[0] for result in results)
    mean = sum(result[1] * result[3] for result in results) / sum(
        result[3] for result in results
    )
    exact = sum(result[2] for result in results) / sum(
        result[3] for result in results
    )
    print(f"maximum_observed_abs_error={maximum:.9g}")
    print(f"aggregate_mean_abs_error={mean:.9g}")
    print(f"aggregate_exact_bf16_fraction={exact:.9g}")
    print("cached_qk_softmax_pv_kernels_executed=true")
    print("physical_capacity_stride_validated=true")


if __name__ == "__main__":
    main()
