"""Exercise the complete modular M5C CUDA pipeline under validation tools."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from decoder_attention_cuda import cuda_decoder_attention_forward_  # noqa: E402
from reference import (  # noqa: E402
    NVFP4Tensor,
    decoder_attention_nvfp4_reference,
)


ADJACENCY_ABSOLUTE_FLOOR = 2.0**-20


def _one_hot_weight(
    rows: int,
    columns: int,
    *,
    offset: int = 0,
) -> NVFP4Tensor:
    packed = torch.zeros(
        (rows, columns // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    output_indices = torch.arange(rows, dtype=torch.int64, device="cuda")
    input_indices = (output_indices + offset) % columns
    codes = torch.full((rows,), 0x2, dtype=torch.uint8, device="cuda")
    packed[output_indices, input_indices // 2] = torch.where(
        (input_indices & 1) == 0,
        codes,
        codes << 4,
    )
    return NVFP4Tensor(
        packed_values=packed,
        block_scales=torch.full(
            (rows, columns // 16),
            0x38,
            dtype=torch.uint8,
            device="cuda",
        ),
        global_decode_scale=torch.tensor(
            1.0,
            dtype=torch.float32,
            device="cuda",
        ),
        logical_shape=(rows, columns),
    )


def _weights(
    hidden_size: int,
    kv_head_count: int,
    head_dim: int,
) -> tuple[NVFP4Tensor, NVFP4Tensor, NVFP4Tensor, NVFP4Tensor]:
    kv_width = kv_head_count * head_dim
    return (
        _one_hot_weight(hidden_size, hidden_size),
        _one_hot_weight(kv_width, hidden_size, offset=head_dim),
        _one_hot_weight(kv_width, hidden_size, offset=2 * head_dim),
        _one_hot_weight(hidden_size, hidden_size),
    )


def _ordered_keys(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    magnitude = bits & 0x7FFF
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def _check_output(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    error = (actual.float() - expected.float()).abs()
    adjacency = (_ordered_keys(actual) - _ordered_keys(expected)).abs()
    maximum = float(error.max().item())
    mean = float(error.mean().item())
    exact = float((actual == expected).float().mean().item())
    maximum_distance = int(adjacency.max().item())
    print(
        f"case={label} output_shape={tuple(actual.shape)} "
        f"max_abs={maximum:.9g} mean_abs={mean:.9g} "
        f"exact_fraction={exact:.9g} max_bf16_distance={maximum_distance}"
    )
    if not torch.all(
        (adjacency <= 1) | (error <= ADJACENCY_ABSOLUTE_FLOOR)
    ).item():
        raise AssertionError(f"{label} exceeded the BF16 stage policy")


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int16)


def _require_bits_equal(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    if actual.shape != expected.shape or not torch.equal(
        _bits(actual),
        _bits(expected),
    ):
        raise AssertionError(f"{label} BF16 storage differs")


def _reduced_validated_case() -> None:
    batch_size, token_count, hidden_size = 2, 3, 32
    query_heads, kv_heads, head_dim = 4, 2, 8
    past_length, capacity = 2, 17
    assert hidden_size == query_heads * head_dim
    generator = torch.Generator(device="cpu").manual_seed(120_011)
    x = (
        torch.randn(
            (batch_size, token_count, hidden_size),
            generator=generator,
        )
        * 0.75
    ).to(torch.bfloat16).cuda()
    x[1].add_(0.25)
    input_norm_weight = torch.linspace(
        0.75,
        1.25,
        hidden_size,
        dtype=torch.float32,
    ).to(torch.bfloat16).cuda()
    q_norm_weight = torch.linspace(
        0.8,
        1.2,
        head_dim,
        dtype=torch.float32,
    ).to(torch.bfloat16).cuda()
    k_norm_weight = torch.linspace(
        1.1,
        0.7,
        head_dim,
        dtype=torch.float32,
    ).to(torch.bfloat16).cuda()
    q_weight, k_weight, v_weight, out_weight = _weights(
        hidden_size,
        kv_heads,
        head_dim,
    )
    k_cache = torch.full(
        (batch_size, kv_heads, capacity, head_dim),
        -29.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.full_like(k_cache, 43.0)
    prefix_shape = (batch_size, kv_heads, past_length, head_dim)
    k_cache[:, :, :past_length].copy_(
        (torch.randn(prefix_shape, generator=generator) * 0.5)
        .to(torch.bfloat16)
        .cuda()
    )
    v_cache[:, :, :past_length].copy_(
        (torch.randn(prefix_shape, generator=generator) * 0.75)
        .to(torch.bfloat16)
        .cuda()
    )
    original_k, original_v = k_cache.clone(), v_cache.clone()
    k_pointer, v_pointer = k_cache.data_ptr(), v_cache.data_ptr()
    expected = decoder_attention_nvfp4_reference(
        x,
        input_norm_weight,
        q_weight,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        out_weight,
        original_k[:, :, :past_length].clone(),
        original_v[:, :, :past_length].clone(),
    )

    actual = cuda_decoder_attention_forward_(
        x,
        input_norm_weight,
        q_weight,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        out_weight,
        k_cache,
        v_cache,
        past_length,
    )
    downstream_output = actual.float().square().sum()
    downstream_cache = (
        k_cache[:, :, past_length : past_length + token_count].float().sum()
        + v_cache[:, :, past_length : past_length + token_count].float().sum()
    )
    _check_output("reduced-b2-t3-p2-c17", actual, expected.output)
    _require_bits_equal(
        "reduced K prefix",
        k_cache[:, :, : past_length + token_count],
        expected.present_k,
    )
    _require_bits_equal(
        "reduced V prefix",
        v_cache[:, :, : past_length + token_count],
        expected.present_v,
    )
    _require_bits_equal(
        "reduced K suffix",
        k_cache[:, :, past_length + token_count :],
        original_k[:, :, past_length + token_count :],
    )
    _require_bits_equal(
        "reduced V suffix",
        v_cache[:, :, past_length + token_count :],
        original_v[:, :, past_length + token_count :],
    )
    if k_cache.data_ptr() != k_pointer or v_cache.data_ptr() != v_pointer:
        raise AssertionError("reduced case replaced cache storage")
    if not torch.isfinite(downstream_output) or not torch.isfinite(downstream_cache):
        raise AssertionError("reduced downstream consumer was nonfinite")
    print("case=reduced-b2-t3-p2-c17 cache_prefix_exact=true suffix_exact=true")


def _canonical_smoke_case() -> None:
    hidden_size, query_heads, kv_heads, head_dim = 3072, 24, 6, 128
    past_length, capacity = 4, 32
    assert hidden_size == query_heads * head_dim
    generator = torch.Generator(device="cpu").manual_seed(121_013)
    x = (
        torch.randn((1, 1, hidden_size), generator=generator) * 0.5
    ).to(torch.bfloat16).cuda()
    input_norm_weight = torch.ones(
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_norm_weight = torch.ones(
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_norm_weight = torch.ones_like(q_norm_weight)
    q_weight, k_weight, v_weight, out_weight = _weights(
        hidden_size,
        kv_heads,
        head_dim,
    )
    k_cache = torch.zeros(
        (1, kv_heads, capacity, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    k_suffix = k_cache[:, :, past_length + 1 :].clone()
    v_suffix = v_cache[:, :, past_length + 1 :].clone()
    k_pointer, v_pointer = k_cache.data_ptr(), v_cache.data_ptr()

    output = cuda_decoder_attention_forward_(
        x,
        input_norm_weight,
        q_weight,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        out_weight,
        k_cache,
        v_cache,
        past_length,
    )
    if output.shape != (1, 1, hidden_size) or output.dtype != torch.bfloat16:
        raise AssertionError("canonical output metadata is wrong")
    if not torch.isfinite(output).all():
        raise AssertionError("canonical output is nonfinite")
    if k_cache.data_ptr() != k_pointer or v_cache.data_ptr() != v_pointer:
        raise AssertionError("canonical case replaced cache storage")
    _require_bits_equal(
        "canonical K suffix",
        k_cache[:, :, past_length + 1 :],
        k_suffix,
    )
    _require_bits_equal(
        "canonical V suffix",
        v_cache[:, :, past_length + 1 :],
        v_suffix,
    )
    print(
        "case=canonical-t1-h3072-hq24-hkv6-d128-p4-c32 "
        "output_finite=true suffix_exact=true"
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
    _reduced_validated_case()
    _canonical_smoke_case()
    torch.cuda.synchronize()
    print("rmsnorm_input_executed=true")
    print("w4a16_q_k_v_out_executed=true")
    print("rmsnorm_q_k_executed=true")
    print("rope_q_k_executed=true")
    print("kv_cache_append_executed=true")
    print("cached_qk_softmax_pv_executed=true")


if __name__ == "__main__":
    main()
