"""Run the in-place CUDA BF16 KV-cache append under validation tools."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cuda_primitives  # noqa: E402


CASES = (
    # label, B, T, Hkv, D, P, C, seed
    ("p0-t3-b2-h2-spare", 2, 3, 2, 8, 0, 19, 80_003),
    ("p128-t4-h3-spare", 1, 4, 3, 16, 128, 512, 80_009),
    ("exact-capacity", 1, 3, 2, 32, 5, 8, 80_011),
    ("canonical-p2048", 1, 1, 6, 128, 2048, 4096, 80_021),
)


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int16)


def _require_bitwise_equal(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label} shape mismatch: {actual.shape} != {expected.shape}"
        )
    if not torch.equal(_bits(actual), _bits(expected)):
        raise AssertionError(f"{label} BF16 storage bits differ")


def _run_case(
    label: str,
    batch_size: int,
    token_count: int,
    kv_head_count: int,
    head_dim: int,
    past_length: int,
    capacity: int,
    seed: int,
) -> None:
    k_cache = torch.full(
        (batch_size, kv_head_count, capacity, head_dim),
        -23.5,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.full_like(k_cache, 29.25)
    if past_length > 0:
        prefix_count = batch_size * kv_head_count * past_length * head_dim
        prefix = (
            torch.arange(prefix_count, dtype=torch.float32, device="cuda")
            .remainder(97)
            .sub(48)
            .reshape(batch_size, kv_head_count, past_length, head_dim)
            .to(torch.bfloat16)
        )
        k_cache[:, :, :past_length].copy_(prefix)
        v_cache[:, :, :past_length].copy_(-prefix)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    new_shape = (batch_size, token_count, kv_head_count, head_dim)
    new_k = (
        torch.randn(new_shape, generator=generator) * 0.5
    ).to(torch.bfloat16).cuda()
    new_v = (
        torch.randn(new_shape, generator=generator) * 0.75
    ).to(torch.bfloat16).cuda()
    original_k = k_cache.clone()
    original_v = v_cache.clone()
    original_new_k = new_k.clone()
    original_new_v = new_v.clone()
    expected_k = original_k.clone()
    expected_v = original_v.clone()
    expected_k[:, :, past_length : past_length + token_count] = (
        new_k.permute(0, 2, 1, 3)
    )
    expected_v[:, :, past_length : past_length + token_count] = (
        new_v.permute(0, 2, 1, 3)
    )
    k_pointer = k_cache.data_ptr()
    v_pointer = v_cache.data_ptr()

    result = cuda_primitives.cuda_kv_cache_append_(
        k_cache,
        v_cache,
        new_k,
        new_v,
        past_length,
    )
    torch.cuda.synchronize()

    if result is not None:
        raise AssertionError(f"{label} append did not return None")
    if k_cache.data_ptr() != k_pointer or v_cache.data_ptr() != v_pointer:
        raise AssertionError(f"{label} replaced cache storage")
    _require_bitwise_equal(f"{label} complete K cache", k_cache, expected_k)
    _require_bitwise_equal(f"{label} complete V cache", v_cache, expected_v)
    _require_bitwise_equal(f"{label} source K", new_k, original_new_k)
    _require_bitwise_equal(f"{label} source V", new_v, original_new_v)
    _require_bitwise_equal(
        f"{label} K prefix",
        k_cache[:, :, :past_length],
        original_k[:, :, :past_length],
    )
    _require_bitwise_equal(
        f"{label} V suffix",
        v_cache[:, :, past_length + token_count :],
        original_v[:, :, past_length + token_count :],
    )
    print(
        f"case={label} cache_shape={tuple(k_cache.shape)} "
        f"new_shape={tuple(new_k.shape)} append=[{past_length},"
        f"{past_length + token_count}) exact=true"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the CUDA device does not support BF16")

    print(f"device={torch.cuda.get_device_name()}")
    print(f"capability={torch.cuda.get_device_capability()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")
    for case in CASES:
        _run_case(*case)
    print("kv_cache_append_kernel_executed=true")


if __name__ == "__main__":
    main()
