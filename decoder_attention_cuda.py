"""First complete modular CUDA decoder-attention forward composition.

The trailing underscore on :func:`cuda_decoder_attention_forward_` denotes
that the supplied physical K/V caches are mutated in place.
"""

from __future__ import annotations

import math

import torch

import cuda_primitives
from reference.nvfp4 import NVFP4Tensor


_INT64_MAX = 2**63 - 1


def _require_bf16_cuda_contiguous(
    name: str,
    value: object,
    *,
    rank: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.dtype != torch.bfloat16:
        raise TypeError(
            f"{name} must have dtype torch.bfloat16, got {value.dtype}"
        )
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got rank {value.ndim}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if any(dimension <= 0 for dimension in value.shape):
        raise ValueError(f"{name} dimensions must all be positive")
    return value


def _require_same_device(
    name: str,
    tensor: torch.Tensor,
    device: torch.device,
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")


def _require_shape(
    name: str,
    tensor: torch.Tensor,
    expected: tuple[int, ...],
) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}, got {actual}")


def _require_positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return converted


def _require_nvfp4_weight(
    name: str,
    weight: object,
    expected_shape: tuple[int, int],
    device: torch.device,
) -> NVFP4Tensor:
    if not isinstance(weight, NVFP4Tensor):
        raise TypeError(f"{name} must be an NVFP4Tensor")
    if weight.logical_shape != expected_shape:
        raise ValueError(
            f"{name}.logical_shape must be {expected_shape}, "
            f"got {weight.logical_shape}"
        )

    rows, columns = expected_shape
    expected_packed_shape = (rows, columns // 2)
    expected_scale_shape = (rows, columns // 16)
    fields = (
        ("packed_values", weight.packed_values, torch.uint8, 2),
        ("block_scales", weight.block_scales, torch.uint8, 2),
        ("global_decode_scale", weight.global_decode_scale, torch.float32, 0),
    )
    for field_name, tensor, dtype, rank in fields:
        qualified_name = f"{name}.{field_name}"
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{qualified_name} must be a torch.Tensor")
        if tensor.device.type != "cuda":
            raise ValueError(f"{qualified_name} must be a CUDA tensor")
        if tensor.device != device:
            raise ValueError(
                f"{qualified_name} must be on device {device}, got {tensor.device}"
            )
        if tensor.dtype != dtype:
            raise TypeError(
                f"{qualified_name} must have dtype {dtype}, got {tensor.dtype}"
            )
        if tensor.ndim != rank:
            raise ValueError(
                f"{qualified_name} must have rank {rank}, got rank {tensor.ndim}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{qualified_name} must be contiguous")

    _require_shape(
        f"{name}.packed_values",
        weight.packed_values,
        expected_packed_shape,
    )
    _require_shape(
        f"{name}.block_scales",
        weight.block_scales,
        expected_scale_shape,
    )
    _require_shape(
        f"{name}.global_decode_scale",
        weight.global_decode_scale,
        (),
    )
    return weight


def _require_stage(
    name: str,
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"internal {name} shape mismatch: expected {expected_shape}, "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.bfloat16 or tensor.device != device:
        raise RuntimeError(f"internal {name} dtype/device contract was violated")
    if not tensor.is_contiguous():
        raise RuntimeError(f"internal {name} must be contiguous for metadata-only reshape")


def cuda_decoder_attention_forward_(
    x: torch.Tensor,
    input_norm_weight: torch.Tensor,
    q_weight: NVFP4Tensor,
    k_weight: NVFP4Tensor,
    v_weight: NVFP4Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    out_weight: NVFP4Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    past_length: int,
    *,
    rms_eps: float = 1.0e-6,
    rope_theta: float = 10_000.0,
) -> torch.Tensor:
    """Run the modular NVFP4 decoder-attention path and append K/V in place.

    All four projections use the frozen baseline ``cuda_w4a16_linear``. The
    capacity-backed caches have physical layout ``[B,Hkv,C,D]`` and attention
    derives its logical context length as ``past_length + T``.
    """

    x = _require_bf16_cuda_contiguous("x", x, rank=3)
    input_norm_weight = _require_bf16_cuda_contiguous(
        "input_norm_weight",
        input_norm_weight,
        rank=1,
    )
    q_norm_weight = _require_bf16_cuda_contiguous(
        "q_norm_weight",
        q_norm_weight,
        rank=1,
    )
    k_norm_weight = _require_bf16_cuda_contiguous(
        "k_norm_weight",
        k_norm_weight,
        rank=1,
    )
    k_cache = _require_bf16_cuda_contiguous("k_cache", k_cache, rank=4)
    v_cache = _require_bf16_cuda_contiguous("v_cache", v_cache, rank=4)

    if isinstance(past_length, bool) or not isinstance(past_length, int):
        raise TypeError("past_length must be an integer")
    if past_length < 0:
        raise ValueError(f"past_length must be nonnegative, got {past_length}")
    rms_eps_value = _require_positive_finite("rms_eps", rms_eps)
    rope_theta_value = _require_positive_finite("rope_theta", rope_theta)

    batch_size, token_count, hidden_size = x.shape
    cache_batch, kv_head_count, cache_capacity, head_dim = k_cache.shape
    if tuple(v_cache.shape) != tuple(k_cache.shape):
        raise ValueError(
            "v_cache must have the same shape as k_cache, got "
            f"{tuple(v_cache.shape)} and {tuple(k_cache.shape)}"
        )
    if cache_batch != batch_size:
        raise ValueError(
            "x and cache batch sizes must match "
            f"({batch_size} != {cache_batch})"
        )
    if hidden_size % head_dim != 0:
        raise ValueError(
            "hidden size must be divisible by cache head dimension "
            f"({hidden_size} % {head_dim} != 0)"
        )
    query_head_count = hidden_size // head_dim
    if query_head_count % kv_head_count != 0:
        raise ValueError(
            "number of query heads must be divisible by number of KV heads "
            f"({query_head_count} % {kv_head_count} != 0)"
        )
    if head_dim < 2 or head_dim % 2 != 0:
        raise ValueError(
            f"cache head dimension must be even and at least 2, got {head_dim}"
        )
    if past_length > _INT64_MAX - token_count:
        raise OverflowError("past_length + current token count overflows int64")
    logical_context_length = past_length + token_count
    if logical_context_length > cache_capacity:
        raise ValueError(
            "past_length + current token count exceeds cache capacity "
            f"({logical_context_length} > {cache_capacity})"
        )

    device = x.device
    for name, tensor in (
        ("input_norm_weight", input_norm_weight),
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
    ):
        _require_same_device(name, tensor, device)
    _require_shape("input_norm_weight", input_norm_weight, (hidden_size,))
    _require_shape("q_norm_weight", q_norm_weight, (head_dim,))
    _require_shape("k_norm_weight", k_norm_weight, (head_dim,))

    kv_width = kv_head_count * head_dim
    q_weight = _require_nvfp4_weight(
        "q_weight",
        q_weight,
        (hidden_size, hidden_size),
        device,
    )
    k_weight = _require_nvfp4_weight(
        "k_weight",
        k_weight,
        (kv_width, hidden_size),
        device,
    )
    v_weight = _require_nvfp4_weight(
        "v_weight",
        v_weight,
        (kv_width, hidden_size),
        device,
    )
    out_weight = _require_nvfp4_weight(
        "out_weight",
        out_weight,
        (hidden_size, hidden_size),
        device,
    )

    x_norm = cuda_primitives.cuda_rms_norm(
        x,
        input_norm_weight,
        rms_eps_value,
    )
    _require_stage(
        "input RMSNorm",
        x_norm,
        (batch_size, token_count, hidden_size),
        device,
    )

    q_flat = cuda_primitives.cuda_w4a16_linear(x_norm, q_weight)
    k_flat = cuda_primitives.cuda_w4a16_linear(x_norm, k_weight)
    v_flat = cuda_primitives.cuda_w4a16_linear(x_norm, v_weight)
    _require_stage(
        "Q projection",
        q_flat,
        (batch_size, token_count, hidden_size),
        device,
    )
    _require_stage(
        "K projection",
        k_flat,
        (batch_size, token_count, kv_width),
        device,
    )
    _require_stage(
        "V projection",
        v_flat,
        (batch_size, token_count, kv_width),
        device,
    )

    q_heads = q_flat.reshape(
        batch_size,
        token_count,
        query_head_count,
        head_dim,
    )
    k_heads = k_flat.reshape(
        batch_size,
        token_count,
        kv_head_count,
        head_dim,
    )
    v_heads = v_flat.reshape(
        batch_size,
        token_count,
        kv_head_count,
        head_dim,
    )
    _require_stage(
        "Q head view",
        q_heads,
        (batch_size, token_count, query_head_count, head_dim),
        device,
    )
    _require_stage(
        "K head view",
        k_heads,
        (batch_size, token_count, kv_head_count, head_dim),
        device,
    )
    _require_stage(
        "V head view",
        v_heads,
        (batch_size, token_count, kv_head_count, head_dim),
        device,
    )

    q_norm = cuda_primitives.cuda_rms_norm(
        q_heads,
        q_norm_weight,
        rms_eps_value,
    )
    k_norm = cuda_primitives.cuda_rms_norm(
        k_heads,
        k_norm_weight,
        rms_eps_value,
    )
    _require_stage(
        "Q per-head RMSNorm",
        q_norm,
        (batch_size, token_count, query_head_count, head_dim),
        device,
    )
    _require_stage(
        "K per-head RMSNorm",
        k_norm,
        (batch_size, token_count, kv_head_count, head_dim),
        device,
    )
    q_rope = cuda_primitives.cuda_apply_rope(
        q_norm,
        past_length,
        rope_theta_value,
    )
    k_rope = cuda_primitives.cuda_apply_rope(
        k_norm,
        past_length,
        rope_theta_value,
    )
    _require_stage(
        "Q RoPE",
        q_rope,
        (batch_size, token_count, query_head_count, head_dim),
        device,
    )
    _require_stage(
        "K RoPE",
        k_rope,
        (batch_size, token_count, kv_head_count, head_dim),
        device,
    )

    cuda_primitives.cuda_kv_cache_append_(
        k_cache,
        v_cache,
        k_rope,
        v_heads,
        past_length,
    )
    context = cuda_primitives.cuda_gqa_attention_cached(
        q_rope,
        k_cache,
        v_cache,
        past_length,
    )
    _require_stage(
        "attention context",
        context,
        (batch_size, token_count, query_head_count, head_dim),
        device,
    )

    context_flat = context.reshape(batch_size, token_count, hidden_size)
    output = cuda_primitives.cuda_w4a16_linear(context_flat, out_weight)
    _require_stage(
        "output projection",
        output,
        (batch_size, token_count, hidden_size),
        device,
    )
    return output


__all__ = ["cuda_decoder_attention_forward_"]
