"""End-to-end tests for the M5C modular CUDA decoder-attention path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pytest
import torch

from reference import (
    NVFP4Tensor,
    decoder_attention_nvfp4_reference,
)


CUDA_READY = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
pytestmark = pytest.mark.skipif(
    not CUDA_READY,
    reason="requires a CUDA device with PyTorch BF16 support",
)

if CUDA_READY:
    import cuda_primitives
    import decoder_attention_cuda
else:
    cuda_primitives = None  # type: ignore[assignment]
    decoder_attention_cuda = None  # type: ignore[assignment]


@dataclass(frozen=True)
class OutputMetrics:
    maximum_absolute_error: float
    mean_absolute_error: float
    exact_bf16_fraction: float
    maximum_bf16_adjacency_distance: int


ADJACENCY_ABSOLUTE_FLOOR = 2.0**-20


def _bf16_ordered_keys(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    magnitude = bits & 0x7FFF
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> OutputMetrics:
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    error = (actual.float() - expected.float()).abs()
    adjacency = (
        _bf16_ordered_keys(actual) - _bf16_ordered_keys(expected)
    ).abs()
    return OutputMetrics(
        maximum_absolute_error=float(error.max().item()),
        mean_absolute_error=float(error.mean().item()),
        exact_bf16_fraction=float((actual == expected).float().mean().item()),
        maximum_bf16_adjacency_distance=int(adjacency.max().item()),
    )


def _assert_bf16_stage_policy(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> OutputMetrics:
    metrics = _metrics(actual, expected)
    error = (actual.float() - expected.float()).abs()
    adjacency = (
        _bf16_ordered_keys(actual) - _bf16_ordered_keys(expected)
    ).abs()
    assert torch.all(
        (adjacency <= 1) | (error <= ADJACENCY_ABSOLUTE_FLOOR)
    ), metrics
    return metrics


def _bf16_bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int16)


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(_bf16_bits(actual), _bf16_bits(expected))


def _one_hot_weight(
    rows: int,
    columns: int,
    *,
    offset: int = 0,
    zero: bool = False,
    device: str | torch.device = "cuda",
) -> NVFP4Tensor:
    assert columns >= 16 and columns % 16 == 0
    packed = torch.zeros(
        (rows, columns // 2),
        dtype=torch.uint8,
        device=device,
    )
    if not zero:
        output_indices = torch.arange(rows, dtype=torch.int64, device=device)
        input_indices = (output_indices + offset) % columns
        codes = torch.full((rows,), 0x2, dtype=torch.uint8, device=device)
        packed_codes = torch.where(
            (input_indices & 1) == 0,
            codes,
            codes << 4,
        )
        packed[output_indices, input_indices // 2] = packed_codes
    scales = torch.full(
        (rows, columns // 16),
        0x38,
        dtype=torch.uint8,
        device=device,
    )
    return NVFP4Tensor(
        packed_values=packed,
        block_scales=scales,
        global_decode_scale=torch.tensor(
            1.0,
            dtype=torch.float32,
            device=device,
        ),
        logical_shape=(rows, columns),
    )


def _weights(
    hidden_size: int,
    kv_head_count: int,
    head_dim: int,
    *,
    zero_qk: bool = False,
) -> dict[str, NVFP4Tensor]:
    kv_width = kv_head_count * head_dim
    return {
        "q_weight": _one_hot_weight(
            hidden_size,
            hidden_size,
            zero=zero_qk,
        ),
        "k_weight": _one_hot_weight(
            kv_width,
            hidden_size,
            offset=head_dim,
            zero=zero_qk,
        ),
        "v_weight": _one_hot_weight(
            kv_width,
            hidden_size,
            offset=2 * head_dim,
        ),
        "out_weight": _one_hot_weight(hidden_size, hidden_size),
    }


def _case(
    *,
    batch_size: int,
    token_count: int,
    hidden_size: int,
    kv_head_count: int,
    head_dim: int,
    past_length: int,
    capacity: int,
    seed: int,
    zero_qk: bool = False,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (
        torch.randn(
            (batch_size, token_count, hidden_size),
            generator=generator,
        )
        * 0.75
    ).to(torch.bfloat16)
    if batch_size > 1:
        for batch in range(batch_size):
            x[batch].add_(0.125 * batch)
    k_cache = torch.full(
        (batch_size, kv_head_count, capacity, head_dim),
        -31.0,
        dtype=torch.bfloat16,
    )
    v_cache = torch.full_like(k_cache, 47.0)
    if past_length > 0:
        prefix_shape = (batch_size, kv_head_count, past_length, head_dim)
        k_prefix = (
            torch.randn(prefix_shape, generator=generator) * 0.5
        ).to(torch.bfloat16)
        v_prefix = (
            torch.randn(prefix_shape, generator=generator) * 0.75
        ).to(torch.bfloat16)
        for batch in range(batch_size):
            k_prefix[batch].add_(0.25 * batch)
            v_prefix[batch].sub_(0.375 * batch)
        k_cache[:, :, :past_length].copy_(k_prefix)
        v_cache[:, :, :past_length].copy_(v_prefix)

    values: dict[str, object] = {
        "x": x.cuda(),
        "input_norm_weight": torch.linspace(
            0.75,
            1.25,
            hidden_size,
            dtype=torch.float32,
        ).to(torch.bfloat16).cuda(),
        "q_norm_weight": torch.linspace(
            0.8,
            1.2,
            head_dim,
            dtype=torch.float32,
        ).to(torch.bfloat16).cuda(),
        "k_norm_weight": torch.linspace(
            1.1,
            0.7,
            head_dim,
            dtype=torch.float32,
        ).to(torch.bfloat16).cuda(),
        "k_cache": k_cache.cuda(),
        "v_cache": v_cache.cuda(),
        "past_length": past_length,
    }
    values.update(
        _weights(
            hidden_size,
            kv_head_count,
            head_dim,
            zero_qk=zero_qk,
        )
    )
    return values


def _call(values: dict[str, object]) -> torch.Tensor:
    assert decoder_attention_cuda is not None
    return decoder_attention_cuda.cuda_decoder_attention_forward_(
        values["x"],  # type: ignore[arg-type]
        values["input_norm_weight"],  # type: ignore[arg-type]
        values["q_weight"],  # type: ignore[arg-type]
        values["k_weight"],  # type: ignore[arg-type]
        values["v_weight"],  # type: ignore[arg-type]
        values["q_norm_weight"],  # type: ignore[arg-type]
        values["k_norm_weight"],  # type: ignore[arg-type]
        values["out_weight"],  # type: ignore[arg-type]
        values["k_cache"],  # type: ignore[arg-type]
        values["v_cache"],  # type: ignore[arg-type]
        values["past_length"],  # type: ignore[arg-type]
    )


def _reference(
    values: dict[str, object],
    original_k: torch.Tensor,
    original_v: torch.Tensor,
    *,
    return_debug: bool = True,
):
    past_length = values["past_length"]
    assert isinstance(past_length, int)
    return decoder_attention_nvfp4_reference(
        values["x"],  # type: ignore[arg-type]
        values["input_norm_weight"],  # type: ignore[arg-type]
        values["q_weight"],  # type: ignore[arg-type]
        values["k_weight"],  # type: ignore[arg-type]
        values["v_weight"],  # type: ignore[arg-type]
        values["q_norm_weight"],  # type: ignore[arg-type]
        values["k_norm_weight"],  # type: ignore[arg-type]
        values["out_weight"],  # type: ignore[arg-type]
        original_k[:, :, :past_length, :].clone(),
        original_v[:, :, :past_length, :].clone(),
        return_debug=return_debug,
    )


def _explicit_cuda_stages(
    values: dict[str, object],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> dict[str, torch.Tensor]:
    assert cuda_primitives is not None
    x = values["x"]
    assert isinstance(x, torch.Tensor)
    past_length = values["past_length"]
    assert isinstance(past_length, int)
    batch_size, token_count, hidden_size = x.shape
    kv_head_count = k_cache.size(1)
    head_dim = k_cache.size(3)
    query_head_count = hidden_size // head_dim

    x_norm = cuda_primitives.cuda_rms_norm(
        x,
        values["input_norm_weight"],  # type: ignore[arg-type]
        1.0e-6,
    )
    q_flat = cuda_primitives.cuda_w4a16_linear(
        x_norm,
        values["q_weight"],  # type: ignore[arg-type]
    )
    k_flat = cuda_primitives.cuda_w4a16_linear(
        x_norm,
        values["k_weight"],  # type: ignore[arg-type]
    )
    v_flat = cuda_primitives.cuda_w4a16_linear(
        x_norm,
        values["v_weight"],  # type: ignore[arg-type]
    )
    q_heads = q_flat.reshape(batch_size, token_count, query_head_count, head_dim)
    k_heads = k_flat.reshape(batch_size, token_count, kv_head_count, head_dim)
    v_heads = v_flat.reshape(batch_size, token_count, kv_head_count, head_dim)
    q_norm = cuda_primitives.cuda_rms_norm(
        q_heads,
        values["q_norm_weight"],  # type: ignore[arg-type]
        1.0e-6,
    )
    k_norm = cuda_primitives.cuda_rms_norm(
        k_heads,
        values["k_norm_weight"],  # type: ignore[arg-type]
        1.0e-6,
    )
    q_rope = cuda_primitives.cuda_apply_rope(q_norm, past_length)
    k_rope = cuda_primitives.cuda_apply_rope(k_norm, past_length)
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
    context_flat = context.reshape(batch_size, token_count, hidden_size)
    output = cuda_primitives.cuda_w4a16_linear(
        context_flat,
        values["out_weight"],  # type: ignore[arg-type]
    )
    return {
        "input_normalized": x_norm,
        "q_projected": q_flat,
        "k_projected": k_flat,
        "v_projected": v_flat,
        "q_normalized": q_norm,
        "k_normalized": k_norm,
        "q_rope": q_rope,
        "k_rope": k_rope,
        "v_heads": v_heads,
        "context": context,
        "output": output,
    }


def _assert_cache_transition(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    original_k: torch.Tensor,
    original_v: torch.Tensor,
    reference,
    past_length: int,
    token_count: int,
) -> None:
    present_length = past_length + token_count
    _assert_bitwise_equal(
        k_cache[:, :, :present_length],
        reference.present_k,
    )
    _assert_bitwise_equal(
        v_cache[:, :, :present_length],
        reference.present_v,
    )
    _assert_bitwise_equal(
        k_cache[:, :, present_length:],
        original_k[:, :, present_length:],
    )
    _assert_bitwise_equal(
        v_cache[:, :, present_length:],
        original_v[:, :, present_length:],
    )


def test_small_deterministic_pipeline_matches_every_major_reference_stage() -> None:
    values = _case(
        batch_size=1,
        token_count=2,
        hidden_size=32,
        kv_head_count=2,
        head_dim=8,
        past_length=2,
        capacity=11,
        seed=110_003,
    )
    original_k = values["k_cache"].clone()  # type: ignore[union-attr]
    original_v = values["v_cache"].clone()  # type: ignore[union-attr]
    expected = _reference(values, original_k, original_v)
    assert expected.debug is not None

    explicit_k = original_k.clone()
    explicit_v = original_v.clone()
    stages = _explicit_cuda_stages(values, explicit_k, explicit_v)
    debug = expected.debug
    for name in (
        "input_normalized",
        "q_projected",
        "k_projected",
        "v_projected",
        "q_normalized",
        "k_normalized",
        "q_rope",
        "k_rope",
    ):
        _assert_bitwise_equal(stages[name], getattr(debug, name))
    context_metrics = _assert_bf16_stage_policy(stages["context"], debug.context)
    output_metrics = _assert_bf16_stage_policy(stages["output"], expected.output)
    _assert_cache_transition(
        explicit_k,
        explicit_v,
        original_k,
        original_v,
        expected,
        2,
        2,
    )

    public_k = values["k_cache"]
    public_v = values["v_cache"]
    assert isinstance(public_k, torch.Tensor) and isinstance(public_v, torch.Tensor)
    k_pointer, v_pointer = public_k.data_ptr(), public_v.data_ptr()
    actual = _call(values)

    assert torch.equal(actual, stages["output"])
    assert public_k.data_ptr() == k_pointer
    assert public_v.data_ptr() == v_pointer
    _assert_bitwise_equal(public_k, explicit_k)
    _assert_bitwise_equal(public_v, explicit_v)
    print(
        "small-pipeline "
        f"context_max_abs={context_metrics.maximum_absolute_error:.9g} "
        f"output_max_abs={output_metrics.maximum_absolute_error:.9g} "
        f"output_mean_abs={output_metrics.mean_absolute_error:.9g} "
        f"output_exact_fraction={output_metrics.exact_bf16_fraction:.9g} "
        "output_max_bf16_distance="
        f"{output_metrics.maximum_bf16_adjacency_distance}"
    )


def test_batch_two_end_to_end_has_no_batch_contamination() -> None:
    values = _case(
        batch_size=2,
        token_count=3,
        hidden_size=32,
        kv_head_count=2,
        head_dim=8,
        past_length=3,
        capacity=19,
        seed=111_007,
    )
    k_cache = values["k_cache"]
    v_cache = values["v_cache"]
    assert isinstance(k_cache, torch.Tensor) and isinstance(v_cache, torch.Tensor)
    original_k, original_v = k_cache.clone(), v_cache.clone()
    expected = _reference(values, original_k, original_v)
    k_pointer, v_pointer = k_cache.data_ptr(), v_cache.data_ptr()

    actual = _call(values)
    metrics = _assert_bf16_stage_policy(actual, expected.output)

    _assert_cache_transition(
        k_cache,
        v_cache,
        original_k,
        original_v,
        expected,
        3,
        3,
    )
    assert k_cache.data_ptr() == k_pointer
    assert v_cache.data_ptr() == v_pointer
    assert not torch.equal(actual[0], actual[1])

    for batch in range(2):
        isolated = dict(values)
        isolated["x"] = values["x"][batch : batch + 1].clone()  # type: ignore[index,union-attr]
        isolated["k_cache"] = original_k[batch : batch + 1].clone()
        isolated["v_cache"] = original_v[batch : batch + 1].clone()
        isolated_output = _call(isolated)
        assert torch.equal(isolated_output[0], actual[batch])
        _assert_bitwise_equal(
            isolated["k_cache"][0],  # type: ignore[index]
            k_cache[batch],
        )
        _assert_bitwise_equal(
            isolated["v_cache"][0],  # type: ignore[index]
            v_cache[batch],
        )
    print(
        "b2-pipeline "
        f"max_abs={metrics.maximum_absolute_error:.9g} "
        f"mean_abs={metrics.mean_absolute_error:.9g} "
        f"exact_fraction={metrics.exact_bf16_fraction:.9g} "
        f"max_bf16_distance={metrics.maximum_bf16_adjacency_distance}"
    )


def test_t_greater_than_one_future_appends_are_physically_present_but_hidden() -> None:
    values = _case(
        batch_size=1,
        token_count=3,
        hidden_size=32,
        kv_head_count=2,
        head_dim=8,
        past_length=1,
        capacity=13,
        seed=112_009,
        zero_qk=True,
    )
    x = values["x"]
    assert isinstance(x, torch.Tensor)
    x.zero_()
    x[0, 0, 16] = 1.0
    x[0, 1, 17] = 8.0
    x[0, 2, 18] = -16.0
    k_cache = values["k_cache"]
    v_cache = values["v_cache"]
    assert isinstance(k_cache, torch.Tensor) and isinstance(v_cache, torch.Tensor)
    k_cache[:, :, :1].zero_()
    v_cache[:, :, :1].zero_()
    original_k, original_v = k_cache.clone(), v_cache.clone()
    expected = _reference(values, original_k, original_v)

    actual = _call(values)
    _assert_bf16_stage_policy(actual, expected.output)

    one_token = dict(values)
    one_token["x"] = x[:, :1].clone()
    one_token["k_cache"] = original_k.clone()
    one_token["v_cache"] = original_v.clone()
    one_token_output = _call(one_token)
    assert torch.equal(actual[:, :1], one_token_output)

    # With zero Q/K, a deliberately noncausal oracle is the mean over all
    # physically present V slots. It differs strongly from the causal token 0.
    group_mapping = torch.tensor([0, 0, 1, 1], device="cuda")
    noncausal_heads = expected.present_v[:, group_mapping].float().mean(dim=2)
    noncausal_flat = noncausal_heads.to(torch.bfloat16).reshape(1, 1, 32)
    assert cuda_primitives is not None
    noncausal_output = cuda_primitives.cuda_w4a16_linear(
        noncausal_flat,
        values["out_weight"],  # type: ignore[arg-type]
    )
    assert not torch.equal(actual[:, :1], noncausal_output)
    assert torch.count_nonzero(v_cache[:, :, 2:4]).item() > 0
    _assert_cache_transition(
        k_cache,
        v_cache,
        original_k,
        original_v,
        expected,
        1,
        3,
    )


def test_p_zero_full_pipeline_obeys_in_chunk_causality() -> None:
    values = _case(
        batch_size=1,
        token_count=4,
        hidden_size=32,
        kv_head_count=1,
        head_dim=8,
        past_length=0,
        capacity=17,
        seed=113_017,
    )
    k_cache = values["k_cache"]
    v_cache = values["v_cache"]
    assert isinstance(k_cache, torch.Tensor) and isinstance(v_cache, torch.Tensor)
    original_k, original_v = k_cache.clone(), v_cache.clone()
    expected = _reference(values, original_k, original_v)

    actual = _call(values)
    metrics = _assert_bf16_stage_policy(actual, expected.output)
    _assert_cache_transition(
        k_cache,
        v_cache,
        original_k,
        original_v,
        expected,
        0,
        4,
    )
    print(
        "p0-pipeline "
        f"max_abs={metrics.maximum_absolute_error:.9g} "
        f"mean_abs={metrics.mean_absolute_error:.9g} "
        f"exact_fraction={metrics.exact_bf16_fraction:.9g} "
        f"max_bf16_distance={metrics.maximum_bf16_adjacency_distance}"
    )


def test_pipeline_calls_baseline_w4a16_exactly_four_times(monkeypatch) -> None:
    assert decoder_attention_cuda is not None
    values = _case(
        batch_size=1,
        token_count=1,
        hidden_size=32,
        kv_head_count=2,
        head_dim=8,
        past_length=1,
        capacity=7,
        seed=114_019,
    )
    baseline = decoder_attention_cuda.cuda_primitives.cuda_w4a16_linear
    calls: list[tuple[int, int]] = []

    def tracked(x: torch.Tensor, weight: NVFP4Tensor) -> torch.Tensor:
        calls.append(weight.logical_shape)
        return baseline(x, weight)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("grouped-decode projection must not be selected")

    monkeypatch.setattr(
        decoder_attention_cuda.cuda_primitives,
        "cuda_w4a16_linear",
        tracked,
    )
    monkeypatch.setattr(
        decoder_attention_cuda.cuda_primitives,
        "cuda_w4a16_linear_grouped_decode",
        forbidden,
    )

    _call(values)

    assert calls == [(32, 32), (16, 32), (16, 32), (32, 32)]


def test_full_pipeline_non_default_current_stream_orders_every_stage() -> None:
    assert decoder_attention_cuda is not None
    target = _case(
        batch_size=2,
        token_count=2,
        hidden_size=32,
        kv_head_count=2,
        head_dim=8,
        past_length=2,
        capacity=17,
        seed=115_021,
    )
    target_k = target["k_cache"]
    target_v = target["v_cache"]
    assert isinstance(target_k, torch.Tensor) and isinstance(target_v, torch.Tensor)
    expected_reference = _reference(target, target_k.clone(), target_v.clone())

    prepared = dict(target)
    prepared["x"] = torch.zeros_like(target["x"])  # type: ignore[arg-type]
    prepared["input_norm_weight"] = torch.zeros_like(
        target["input_norm_weight"]  # type: ignore[arg-type]
    )
    prepared["q_norm_weight"] = torch.zeros_like(
        target["q_norm_weight"]  # type: ignore[arg-type]
    )
    prepared["k_norm_weight"] = torch.zeros_like(
        target["k_norm_weight"]  # type: ignore[arg-type]
    )
    prepared["k_cache"] = torch.zeros_like(target_k)
    prepared["v_cache"] = torch.zeros_like(target_v)
    for weight_name in ("q_weight", "k_weight", "v_weight", "out_weight"):
        target_weight = target[weight_name]
        assert isinstance(target_weight, NVFP4Tensor)
        prepared[weight_name] = NVFP4Tensor(
            packed_values=torch.zeros_like(target_weight.packed_values),
            block_scales=torch.zeros_like(target_weight.block_scales),
            global_decode_scale=torch.zeros_like(
                target_weight.global_decode_scale
            ),
            logical_shape=target_weight.logical_shape,
        )
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        for tensor_name in (
            "x",
            "input_norm_weight",
            "q_norm_weight",
            "k_norm_weight",
            "k_cache",
            "v_cache",
        ):
            prepared[tensor_name].copy_(target[tensor_name])  # type: ignore[union-attr]
        for weight_name in ("q_weight", "k_weight", "v_weight", "out_weight"):
            prepared_weight = prepared[weight_name]
            target_weight = target[weight_name]
            assert isinstance(prepared_weight, NVFP4Tensor)
            assert isinstance(target_weight, NVFP4Tensor)
            prepared_weight.packed_values.copy_(target_weight.packed_values)
            prepared_weight.block_scales.copy_(target_weight.block_scales)
            prepared_weight.global_decode_scale.copy_(
                target_weight.global_decode_scale
            )
        actual = _call(prepared)
        output_consumer = actual.float().square().sum()
        cache_consumer = (
            prepared["k_cache"][:, :, 2:4].float().sum()  # type: ignore[index]
            + prepared["v_cache"][:, :, 2:4].float().sum()  # type: ignore[index]
        )
    stream.synchronize()

    _assert_bf16_stage_policy(actual, expected_reference.output)
    expected_output_consumer = actual.float().square().sum()
    expected_cache_consumer = (
        prepared["k_cache"][:, :, 2:4].float().sum()  # type: ignore[index]
        + prepared["v_cache"][:, :, 2:4].float().sum()  # type: ignore[index]
    )
    assert torch.equal(output_consumer, expected_output_consumer)
    assert torch.equal(cache_consumer, expected_cache_consumer)
    _assert_bitwise_equal(
        prepared["k_cache"][:, :, :4],  # type: ignore[index]
        expected_reference.present_k,
    )
    _assert_bitwise_equal(
        prepared["v_cache"][:, :, :4],  # type: ignore[index]
        expected_reference.present_v,
    )


def test_canonical_t1_p128_capacity_backed_pipeline_matches_reference() -> None:
    values = _case(
        batch_size=1,
        token_count=1,
        hidden_size=3072,
        kv_head_count=6,
        head_dim=128,
        past_length=128,
        capacity=512,
        seed=116_027,
    )
    k_cache = values["k_cache"]
    v_cache = values["v_cache"]
    assert isinstance(k_cache, torch.Tensor) and isinstance(v_cache, torch.Tensor)
    original_k, original_v = k_cache.clone(), v_cache.clone()
    expected = _reference(values, original_k, original_v, return_debug=False)
    k_pointer, v_pointer = k_cache.data_ptr(), v_cache.data_ptr()

    actual = _call(values)
    metrics = _assert_bf16_stage_policy(actual, expected.output)
    _assert_cache_transition(
        k_cache,
        v_cache,
        original_k,
        original_v,
        expected,
        128,
        1,
    )
    assert k_cache.data_ptr() == k_pointer
    assert v_cache.data_ptr() == v_pointer
    print(
        "canonical-pipeline H=3072 Hq=24 Hkv=6 D=128 P=128 C=512 "
        f"max_abs={metrics.maximum_absolute_error:.9g} "
        f"mean_abs={metrics.mean_absolute_error:.9g} "
        f"exact_fraction={metrics.exact_bf16_fraction:.9g} "
        f"max_bf16_distance={metrics.maximum_bf16_adjacency_distance}"
    )


def test_pipeline_validates_structural_metadata_early() -> None:
    values = _case(
        batch_size=1,
        token_count=2,
        hidden_size=32,
        kv_head_count=2,
        head_dim=8,
        past_length=1,
        capacity=7,
        seed=117_029,
    )

    invalid = dict(values)
    invalid["x"] = invalid["x"].float()  # type: ignore[union-attr]
    with pytest.raises(TypeError, match="x must have dtype torch.bfloat16"):
        _call(invalid)

    invalid = dict(values)
    invalid["x"] = invalid["x"].cpu()  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="x must be a CUDA tensor"):
        _call(invalid)

    invalid = dict(values)
    invalid["x"] = invalid["x"].transpose(1, 2)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="x must be contiguous"):
        _call(invalid)

    invalid = dict(values)
    invalid["input_norm_weight"] = torch.ones(
        31,
        dtype=torch.bfloat16,
        device="cuda",
    )
    with pytest.raises(ValueError, match="input_norm_weight must have shape"):
        _call(invalid)

    invalid = dict(values)
    invalid["v_cache"] = torch.ones(
        (1, 2, 8, 8),
        dtype=torch.bfloat16,
        device="cuda",
    )
    with pytest.raises(ValueError, match="same shape"):
        _call(invalid)

    invalid = dict(values)
    invalid["q_weight"] = torch.ones(1, device="cuda")
    with pytest.raises(TypeError, match="q_weight must be an NVFP4Tensor"):
        _call(invalid)

    invalid = dict(values)
    invalid["q_weight"] = _one_hot_weight(16, 32)
    with pytest.raises(ValueError, match="q_weight.logical_shape"):
        _call(invalid)

    invalid = dict(values)
    invalid["past_length"] = 6
    with pytest.raises(ValueError, match="exceeds cache capacity"):
        _call(invalid)

    invalid = dict(values)
    invalid["past_length"] = torch.iinfo(torch.int64).max
    with pytest.raises(OverflowError, match="overflows int64"):
        _call(invalid)


@pytest.mark.parametrize(
    ("name", "value", "error", "message"),
    (
        ("past_length", True, TypeError, "must be an integer"),
        ("past_length", -1, ValueError, "must be nonnegative"),
        ("rms_eps", 0.0, ValueError, "finite and positive"),
        ("rms_eps", math.nan, ValueError, "finite and positive"),
        ("rope_theta", -1.0, ValueError, "finite and positive"),
        ("rope_theta", math.inf, ValueError, "finite and positive"),
    ),
)
def test_pipeline_rejects_invalid_scalar_metadata(
    name: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    values = _case(
        batch_size=1,
        token_count=1,
        hidden_size=32,
        kv_head_count=2,
        head_dim=8,
        past_length=1,
        capacity=7,
        seed=118_031,
    )
    assert decoder_attention_cuda is not None
    kwargs = {"rms_eps": 1.0e-6, "rope_theta": 10_000.0}
    if name == "past_length":
        values["past_length"] = value
    else:
        kwargs[name] = value
    with pytest.raises(error, match=message):
        decoder_attention_cuda.cuda_decoder_attention_forward_(
            values["x"],  # type: ignore[arg-type]
            values["input_norm_weight"],  # type: ignore[arg-type]
            values["q_weight"],  # type: ignore[arg-type]
            values["k_weight"],  # type: ignore[arg-type]
            values["v_weight"],  # type: ignore[arg-type]
            values["q_norm_weight"],  # type: ignore[arg-type]
            values["k_norm_weight"],  # type: ignore[arg-type]
            values["out_weight"],  # type: ignore[arg-type]
            values["k_cache"],  # type: ignore[arg-type]
            values["v_cache"],  # type: ignore[arg-type]
            values["past_length"],  # type: ignore[arg-type]
            **kwargs,
        )


def test_pipeline_rejects_ambiguous_dimension_inference() -> None:
    values = _case(
        batch_size=1,
        token_count=1,
        hidden_size=32,
        kv_head_count=2,
        head_dim=8,
        past_length=1,
        capacity=7,
        seed=119_037,
    )
    invalid = dict(values)
    invalid["k_cache"] = torch.ones(
        (1, 2, 7, 12),
        dtype=torch.bfloat16,
        device="cuda",
    )
    invalid["v_cache"] = torch.ones_like(invalid["k_cache"])
    invalid["q_norm_weight"] = torch.ones(12, dtype=torch.bfloat16, device="cuda")
    invalid["k_norm_weight"] = torch.ones(12, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="hidden size must be divisible"):
        _call(invalid)

    invalid = dict(values)
    invalid["k_cache"] = torch.ones(
        (1, 3, 7, 8),
        dtype=torch.bfloat16,
        device="cuda",
    )
    invalid["v_cache"] = torch.ones_like(invalid["k_cache"])
    with pytest.raises(ValueError, match="query heads must be divisible"):
        _call(invalid)

    invalid = dict(values)
    invalid["k_cache"] = torch.ones(
        (1, 2, 7, 1),
        dtype=torch.bfloat16,
        device="cuda",
    )
    invalid["v_cache"] = torch.ones_like(invalid["k_cache"])
    invalid["q_norm_weight"] = torch.ones(1, dtype=torch.bfloat16, device="cuda")
    invalid["k_norm_weight"] = torch.ones(1, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="even and at least 2"):
        _call(invalid)


def test_production_pipeline_has_no_cache_compaction_or_grouped_substitution() -> None:
    source_path = Path(__file__).resolve().parents[1] / "decoder_attention_cuda.py"
    source = source_path.read_text()
    for forbidden in (
        "torch.cat",
        ".contiguous()",
        "clone(",
        "cuda_w4a16_linear_grouped_decode",
        "cuda_primitives.cuda_gqa_attention(",
    ):
        assert forbidden not in source
    assert source.count("cuda_primitives.cuda_w4a16_linear(") == 4
    assert "cuda_primitives.cuda_gqa_attention_cached(" in source
    assert "context.reshape(batch_size, token_count, hidden_size)" in source
