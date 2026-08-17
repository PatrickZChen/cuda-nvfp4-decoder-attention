"""Mathematically explicit PyTorch reference for decoder attention.

The functions in this module favor visible precision and layout transitions over
performance.  BF16 tensors represent architecture storage boundaries; sensitive
arithmetic and reductions are explicitly performed in FP32.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class DecoderAttentionConfig:
    """Frozen dimensions and numerical constants for decoder attention."""

    hidden_size: int = 3072
    num_query_heads: int = 24
    num_kv_heads: int = 6
    head_dim: int = 128
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        dimension_names = (
            "hidden_size",
            "num_query_heads",
            "num_kv_heads",
            "head_dim",
        )
        for name in dimension_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.hidden_size != self.num_query_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_query_heads * head_dim "
                f"({self.hidden_size} != {self.num_query_heads} * {self.head_dim})"
            )
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_kv_heads "
                f"({self.num_query_heads} % {self.num_kv_heads} != 0)"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for adjacent-pair RoPE, got {self.head_dim}"
            )

        _validate_positive_finite("rms_eps", self.rms_eps)
        _validate_positive_finite("rope_theta", self.rope_theta)


@dataclass(frozen=True)
class DecoderAttentionDebug:
    """Stored stage outputs and FP32 attention intermediates for debugging."""

    input_normalized: Tensor
    q_projected: Tensor
    k_projected: Tensor
    v_projected: Tensor
    q_normalized: Tensor
    k_normalized: Tensor
    q_rope: Tensor
    k_rope: Tensor
    attention_scores: Tensor
    attention_probabilities: Tensor
    context: Tensor


@dataclass(frozen=True)
class DecoderAttentionResult:
    """Decoder-attention output and the resulting KV-cache state."""

    output: Tensor
    present_k: Tensor
    present_v: Tensor
    debug: DecoderAttentionDebug | None = None


def _validate_positive_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


def _require_tensor(name: str, tensor: object) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return tensor


def _require_bf16(name: str, tensor: Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must have dtype torch.bfloat16, got {tensor.dtype}")


def _require_shape(name: str, tensor: Tensor, expected: tuple[int, ...]) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}, got {actual}")


def _require_same_device(name: str, tensor: Tensor, device: torch.device) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")


def rms_norm_reference(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Apply RMSNorm over the final axis using FP32 arithmetic and BF16 storage."""

    x = _require_tensor("x", x)
    weight = _require_tensor("weight", weight)
    _require_bf16("x", x)
    _require_bf16("weight", weight)
    _validate_positive_finite("eps", eps)

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    _require_shape("weight", weight, (x.shape[-1],))
    _require_same_device("weight", weight, x.device)

    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    mean_square = torch.mean(
        x_fp32 * x_fp32,
        dim=-1,
        keepdim=True,
        dtype=torch.float32,
    )
    rms_inverse = torch.rsqrt(mean_square + float(eps))
    return (x_fp32 * rms_inverse * weight_fp32).to(torch.bfloat16)


def linear_reference(x: Tensor, weight: Tensor) -> Tensor:
    """Compute ``x @ weight.T`` in FP32 and store the result as BF16."""

    x = _require_tensor("x", x)
    weight = _require_tensor("weight", weight)
    _require_bf16("x", x)
    _require_bf16("weight", weight)

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank 2, got rank {weight.ndim}")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            "x final dimension must match weight input dimension "
            f"({x.shape[-1]} != {weight.shape[1]})"
        )
    _require_same_device("weight", weight, x.device)

    # Reduce one output feature at a time.  Besides bounding temporary storage,
    # the explicit FP32 multiply and sum avoid dependence on CUDA matmul modes.
    input_width = x.shape[-1]
    output_width = weight.shape[0]
    x_rows_fp32 = x.to(torch.float32).reshape(-1, input_width)
    weight_fp32 = weight.to(torch.float32)
    output_columns = [
        torch.sum(
            x_rows_fp32 * weight_fp32[index][None, :],
            dim=-1,
            dtype=torch.float32,
        )
        for index in range(output_width)
    ]
    result_shape = (*x.shape[:-1], output_width)
    result_fp32 = torch.stack(output_columns, dim=-1).reshape(result_shape)
    return result_fp32.to(torch.bfloat16)


def reshape_heads_reference(
    x: Tensor,
    num_heads: int,
    head_dim: int,
) -> Tensor:
    """Map flat index ``h * head_dim + d`` to ``[B, T, h, d]``."""

    x = _require_tensor("x", x)
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, T, width], got rank {x.ndim}")
    for name, value in (("num_heads", num_heads), ("head_dim", head_dim)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    expected_width = num_heads * head_dim
    if x.shape[-1] != expected_width:
        raise ValueError(
            f"x final dimension must be num_heads * head_dim ({expected_width}), "
            f"got {x.shape[-1]}"
        )
    return x.reshape(x.shape[0], x.shape[1], num_heads, head_dim)


def apply_rope_reference(
    x: Tensor,
    position_offset: int,
    rope_theta: float,
    *,
    store_bf16: bool = True,
) -> Tensor:
    """Apply adjacent-pair RoPE using absolute positions ``position_offset + i``.

    Setting ``store_bf16=False`` exposes the FP32 value immediately before the
    architecture's BF16 storage boundary.  It is useful for numerical tests of
    the rotation itself; the main decoder path always stores BF16.
    """

    x = _require_tensor("x", x)
    _require_bf16("x", x)
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B, T, heads, D], got rank {x.ndim}")
    if x.shape[-1] <= 0 or x.shape[-1] % 2 != 0:
        raise ValueError(
            f"x head dimension must be positive and even, got {x.shape[-1]}"
        )
    if isinstance(position_offset, bool) or not isinstance(position_offset, int):
        raise TypeError("position_offset must be an integer")
    if position_offset < 0:
        raise ValueError(f"position_offset must be nonnegative, got {position_offset}")
    _validate_positive_finite("rope_theta", rope_theta)
    if not isinstance(store_bf16, bool):
        raise TypeError("store_bf16 must be a bool")

    token_count = x.shape[1]
    head_dim = x.shape[-1]
    positions = torch.arange(
        position_offset,
        position_offset + token_count,
        dtype=torch.int64,
        device=x.device,
    ).to(torch.float32)
    pair_even_indices = torch.arange(
        0,
        head_dim,
        2,
        dtype=torch.float32,
        device=x.device,
    )
    pair_exponents = pair_even_indices / float(head_dim)
    angle_denominators = torch.pow(
        torch.tensor(float(rope_theta), dtype=torch.float32, device=x.device),
        pair_exponents,
    )
    angles = positions[:, None] / angle_denominators[None, :]
    cosines = torch.cos(angles)[None, :, None, :]
    sines = torch.sin(angles)[None, :, None, :]

    x_fp32 = x.to(torch.float32)
    x_even = x_fp32[..., 0::2]
    x_odd = x_fp32[..., 1::2]
    out_even = x_even * cosines - x_odd * sines
    out_odd = x_even * sines + x_odd * cosines

    # Stacking each (even, odd) result before flattening makes the adjacent-pair
    # mapping explicit and avoids the split-half RoPE convention.
    rotated_fp32 = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    if store_bf16:
        return rotated_fp32.to(torch.bfloat16)
    return rotated_fp32


def gqa_attention_reference(
    q: Tensor,
    present_k: Tensor,
    present_v: Tensor,
    past_length: int,
    *,
    return_attention: bool = True,
) -> tuple[Tensor | None, Tensor | None, Tensor]:
    """Evaluate causal GQA by directly mapping each query head to one KV head.

    Returned scores include the causal ``-inf`` mask.  Probabilities are FP32,
    and context crosses its architecture-defined BF16 storage boundary.
    ``return_attention=False`` avoids retaining the score/probability tensors.
    """

    q = _require_tensor("q", q)
    present_k = _require_tensor("present_k", present_k)
    present_v = _require_tensor("present_v", present_v)
    for name, tensor in (("q", q), ("present_k", present_k), ("present_v", present_v)):
        _require_bf16(name, tensor)
        if tensor.ndim != 4:
            raise ValueError(f"{name} must be rank 4, got rank {tensor.ndim}")
    if isinstance(past_length, bool) or not isinstance(past_length, int):
        raise TypeError("past_length must be an integer")
    if past_length < 0:
        raise ValueError(f"past_length must be nonnegative, got {past_length}")
    if not isinstance(return_attention, bool):
        raise TypeError("return_attention must be a bool")

    batch_size, token_count, num_query_heads, head_dim = q.shape
    cache_batch, num_kv_heads, context_length, cache_head_dim = present_k.shape
    if tuple(present_v.shape) != tuple(present_k.shape):
        raise ValueError(
            "present_v must have the same shape as present_k, got "
            f"{tuple(present_v.shape)} and {tuple(present_k.shape)}"
        )
    if batch_size != cache_batch:
        raise ValueError(
            f"q and cache batch sizes must match ({batch_size} != {cache_batch})"
        )
    if head_dim != cache_head_dim:
        raise ValueError(
            f"q and cache head dimensions must match ({head_dim} != {cache_head_dim})"
        )
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "number of query heads must be divisible by number of KV heads "
            f"({num_query_heads} % {num_kv_heads} != 0)"
        )
    if context_length != past_length + token_count:
        raise ValueError(
            "cache context length must equal past_length + current token count "
            f"({context_length} != {past_length} + {token_count})"
        )
    _require_same_device("present_k", present_k, q.device)
    _require_same_device("present_v", present_v, q.device)

    group_size = num_query_heads // num_kv_heads
    key_positions = torch.arange(context_length, device=q.device)
    query_positions = past_length + torch.arange(token_count, device=q.device)
    visible = key_positions[None, :] <= query_positions[:, None]
    inverse_sqrt_head_dim = 1.0 / math.sqrt(head_dim)

    score_heads: list[Tensor] | None = [] if return_attention else None
    probability_heads: list[Tensor] | None = [] if return_attention else None
    context_heads: list[Tensor] = []

    for query_head in range(num_query_heads):
        kv_head = query_head // group_size
        q_fp32 = q[:, :, query_head, :].to(torch.float32)
        k_fp32 = present_k[:, kv_head, :, :].to(torch.float32)
        v_fp32 = present_v[:, kv_head, :, :].to(torch.float32)

        score_rows = [
            torch.sum(
                q_fp32[:, token, None, :] * k_fp32,
                dim=-1,
                dtype=torch.float32,
            )
            for token in range(token_count)
        ]
        scores = torch.stack(score_rows, dim=1)
        scores = scores * inverse_sqrt_head_dim
        scores = scores.masked_fill(~visible[None, :, :], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        context_rows = [
            torch.sum(
                probabilities[:, token, :, None] * v_fp32,
                dim=1,
                dtype=torch.float32,
            )
            for token in range(token_count)
        ]
        context_fp32 = torch.stack(context_rows, dim=1)
        context_heads.append(context_fp32.to(torch.bfloat16))

        if return_attention:
            assert score_heads is not None and probability_heads is not None
            score_heads.append(scores)
            probability_heads.append(probabilities)

    context = torch.stack(context_heads, dim=2)
    if not return_attention:
        return None, None, context

    assert score_heads is not None and probability_heads is not None
    attention_scores = torch.stack(score_heads, dim=1)
    attention_probabilities = torch.stack(probability_heads, dim=1)
    return attention_scores, attention_probabilities, context


def _validate_main_inputs(
    x: Tensor,
    input_norm_weight: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    out_weight: Tensor,
    past_k: Tensor | None,
    past_v: Tensor | None,
    config: DecoderAttentionConfig,
) -> int:
    if not isinstance(config, DecoderAttentionConfig):
        raise TypeError("config must be a DecoderAttentionConfig")

    x = _require_tensor("x", x)
    _require_bf16("x", x)
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, T, H], got rank {x.ndim}")
    batch_size, token_count, hidden_size = x.shape
    if batch_size < 1:
        raise ValueError(f"x batch size must be at least 1, got {batch_size}")
    if token_count < 1:
        raise ValueError(f"x token count must be at least 1, got {token_count}")
    if hidden_size != config.hidden_size:
        raise ValueError(
            f"x hidden dimension must be {config.hidden_size}, got {hidden_size}"
        )

    kv_width = config.num_kv_heads * config.head_dim
    parameters = (
        ("input_norm_weight", input_norm_weight, (config.hidden_size,)),
        ("q_weight", q_weight, (config.hidden_size, config.hidden_size)),
        ("k_weight", k_weight, (kv_width, config.hidden_size)),
        ("v_weight", v_weight, (kv_width, config.hidden_size)),
        ("q_norm_weight", q_norm_weight, (config.head_dim,)),
        ("k_norm_weight", k_norm_weight, (config.head_dim,)),
        ("out_weight", out_weight, (config.hidden_size, config.hidden_size)),
    )
    for name, tensor, shape in parameters:
        tensor = _require_tensor(name, tensor)
        _require_bf16(name, tensor)
        _require_shape(name, tensor, shape)
        _require_same_device(name, tensor, x.device)

    if (past_k is None) != (past_v is None):
        raise ValueError("past_k and past_v must either both be provided or both be None")
    if past_k is None:
        return 0

    assert past_v is not None
    past_k = _require_tensor("past_k", past_k)
    past_v = _require_tensor("past_v", past_v)
    _require_bf16("past_k", past_k)
    _require_bf16("past_v", past_v)
    if past_k.ndim != 4:
        raise ValueError(f"past_k must be rank 4, got rank {past_k.ndim}")
    if past_v.ndim != 4:
        raise ValueError(f"past_v must be rank 4, got rank {past_v.ndim}")

    past_length = past_k.shape[2]
    expected_cache_shape = (
        batch_size,
        config.num_kv_heads,
        past_length,
        config.head_dim,
    )
    _require_shape("past_k", past_k, expected_cache_shape)
    _require_shape("past_v", past_v, expected_cache_shape)
    _require_same_device("past_k", past_k, x.device)
    _require_same_device("past_v", past_v, x.device)
    return past_length


def decoder_attention_reference(
    x: Tensor,
    input_norm_weight: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    out_weight: Tensor,
    *,
    past_k: Tensor | None = None,
    past_v: Tensor | None = None,
    config: DecoderAttentionConfig = DecoderAttentionConfig(),
    return_debug: bool = False,
) -> DecoderAttentionResult:
    """Run the frozen BF16/FP32 decoder-attention semantic pipeline."""

    if not isinstance(return_debug, bool):
        raise TypeError("return_debug must be a bool")
    past_length = _validate_main_inputs(
        x,
        input_norm_weight,
        q_weight,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        out_weight,
        past_k,
        past_v,
        config,
    )

    input_normalized = rms_norm_reference(x, input_norm_weight, config.rms_eps)
    q_projected = linear_reference(input_normalized, q_weight)
    k_projected = linear_reference(input_normalized, k_weight)
    v_projected = linear_reference(input_normalized, v_weight)

    q_heads = reshape_heads_reference(
        q_projected, config.num_query_heads, config.head_dim
    )
    k_heads = reshape_heads_reference(k_projected, config.num_kv_heads, config.head_dim)
    v_heads = reshape_heads_reference(v_projected, config.num_kv_heads, config.head_dim)

    q_normalized = rms_norm_reference(q_heads, q_norm_weight, config.rms_eps)
    k_normalized = rms_norm_reference(k_heads, k_norm_weight, config.rms_eps)
    q_rope = apply_rope_reference(
        q_normalized, past_length, config.rope_theta, store_bf16=True
    )
    k_rope = apply_rope_reference(
        k_normalized, past_length, config.rope_theta, store_bf16=True
    )

    new_k = k_rope.permute(0, 2, 1, 3).contiguous()
    new_v = v_heads.permute(0, 2, 1, 3).contiguous()
    if past_k is None:
        present_k = new_k
        present_v = new_v
    else:
        assert past_v is not None
        present_k = torch.cat((past_k, new_k), dim=2)
        present_v = torch.cat((past_v, new_v), dim=2)

    attention_scores, attention_probabilities, context = gqa_attention_reference(
        q_rope,
        present_k,
        present_v,
        past_length,
        return_attention=return_debug,
    )

    batch_size, token_count = x.shape[:2]
    context_flat = context.reshape(batch_size, token_count, config.hidden_size)
    output = linear_reference(context_flat, out_weight)

    debug: DecoderAttentionDebug | None = None
    if return_debug:
        assert attention_scores is not None and attention_probabilities is not None
        debug = DecoderAttentionDebug(
            input_normalized=input_normalized,
            q_projected=q_projected,
            k_projected=k_projected,
            v_projected=v_projected,
            q_normalized=q_normalized,
            k_normalized=k_normalized,
            q_rope=q_rope,
            k_rope=k_rope,
            attention_scores=attention_scores,
            attention_probabilities=attention_probabilities,
            context=context,
        )

    return DecoderAttentionResult(
        output=output,
        present_k=present_k,
        present_v=present_v,
        debug=debug,
    )


__all__ = [
    "DecoderAttentionConfig",
    "DecoderAttentionDebug",
    "DecoderAttentionResult",
    "apply_rope_reference",
    "decoder_attention_reference",
    "gqa_attention_reference",
    "linear_reference",
    "reshape_heads_reference",
    "rms_norm_reference",
]
