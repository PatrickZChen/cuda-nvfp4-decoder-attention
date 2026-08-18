"""Transparent composed NVFP4 decoder-attention correctness oracle."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .decoder_attention import (
    apply_rope_reference,
    gqa_attention_reference,
    reshape_heads_reference,
    rms_norm_reference,
)
from .nvfp4 import NVFP4Tensor
from .w4a16 import w4a16_linear_reference


@dataclass(frozen=True)
class NVFP4DecoderAttentionDebug:
    """Stored BF16 stages used for end-to-end numerical diagnosis."""

    input_normalized: Tensor
    q_projected: Tensor
    k_projected: Tensor
    v_projected: Tensor
    q_normalized: Tensor
    k_normalized: Tensor
    q_rope: Tensor
    k_rope: Tensor
    context: Tensor


@dataclass(frozen=True)
class NVFP4DecoderAttentionResult:
    """Composed output, compact present caches, and optional stage state."""

    output: Tensor
    present_k: Tensor
    present_v: Tensor
    debug: NVFP4DecoderAttentionDebug | None = None


def decoder_attention_nvfp4_reference(
    x: Tensor,
    input_norm_weight: Tensor,
    q_weight: NVFP4Tensor,
    k_weight: NVFP4Tensor,
    v_weight: NVFP4Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    out_weight: NVFP4Tensor,
    past_k: Tensor,
    past_v: Tensor,
    *,
    rms_eps: float = 1.0e-6,
    rope_theta: float = 10_000.0,
    return_debug: bool = False,
) -> NVFP4DecoderAttentionResult:
    """Compose the validated references with compact concatenation semantics."""

    if not isinstance(x, Tensor) or x.ndim != 3:
        raise ValueError("x must be a rank-3 torch.Tensor")
    if not isinstance(past_k, Tensor) or not isinstance(past_v, Tensor):
        raise TypeError("past_k and past_v must be torch.Tensor objects")
    if past_k.ndim != 4 or past_v.ndim != 4:
        raise ValueError("past_k and past_v must be rank 4")
    if tuple(past_v.shape) != tuple(past_k.shape):
        raise ValueError("past_v must have the same shape as past_k")

    batch_size, token_count, hidden_size = x.shape
    cache_batch, kv_head_count, past_length, head_dim = past_k.shape
    if cache_batch != batch_size:
        raise ValueError("x and past cache batch sizes must match")
    if hidden_size % head_dim != 0:
        raise ValueError("hidden size must be divisible by head dimension")
    query_head_count = hidden_size // head_dim
    if query_head_count % kv_head_count != 0:
        raise ValueError("query head count must be divisible by KV head count")

    input_normalized = rms_norm_reference(x, input_norm_weight, rms_eps)
    q_projected = w4a16_linear_reference(input_normalized, q_weight)
    k_projected = w4a16_linear_reference(input_normalized, k_weight)
    v_projected = w4a16_linear_reference(input_normalized, v_weight)

    q_heads = reshape_heads_reference(
        q_projected,
        query_head_count,
        head_dim,
    )
    k_heads = reshape_heads_reference(k_projected, kv_head_count, head_dim)
    v_heads = reshape_heads_reference(v_projected, kv_head_count, head_dim)
    q_normalized = rms_norm_reference(q_heads, q_norm_weight, rms_eps)
    k_normalized = rms_norm_reference(k_heads, k_norm_weight, rms_eps)
    q_rope = apply_rope_reference(
        q_normalized,
        past_length,
        rope_theta,
        store_bf16=True,
    )
    k_rope = apply_rope_reference(
        k_normalized,
        past_length,
        rope_theta,
        store_bf16=True,
    )

    new_k_cache_major = k_rope.permute(0, 2, 1, 3).contiguous()
    new_v_cache_major = v_heads.permute(0, 2, 1, 3).contiguous()
    present_k = torch.cat((past_k, new_k_cache_major), dim=2)
    present_v = torch.cat((past_v, new_v_cache_major), dim=2)
    _, _, context = gqa_attention_reference(
        q_rope,
        present_k,
        present_v,
        past_length,
        return_attention=False,
    )
    context_flat = context.reshape(batch_size, token_count, hidden_size)
    output = w4a16_linear_reference(context_flat, out_weight)

    debug = None
    if return_debug:
        debug = NVFP4DecoderAttentionDebug(
            input_normalized=input_normalized,
            q_projected=q_projected,
            k_projected=k_projected,
            v_projected=v_projected,
            q_normalized=q_normalized,
            k_normalized=k_normalized,
            q_rope=q_rope,
            k_rope=k_rope,
            context=context,
        )
    return NVFP4DecoderAttentionResult(
        output=output,
        present_k=present_k,
        present_v=present_v,
        debug=debug,
    )


__all__ = [
    "NVFP4DecoderAttentionDebug",
    "NVFP4DecoderAttentionResult",
    "decoder_attention_nvfp4_reference",
]
