"""Correctness and integration tests for the Milestone 5B CUDA KV cache."""

from __future__ import annotations

import pytest
import torch


CUDA_READY = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
pytestmark = pytest.mark.skipif(
    not CUDA_READY,
    reason="requires a CUDA device with PyTorch BF16 support",
)

if CUDA_READY:
    import cuda_primitives
else:
    cuda_primitives = None  # type: ignore[assignment]


def _call(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    past_length: int,
) -> None:
    assert cuda_primitives is not None
    return cuda_primitives.cuda_kv_cache_append_(
        k_cache,
        v_cache,
        new_k,
        new_v,
        past_length,
    )


def _raw_call(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    past_length: int,
) -> None:
    return torch.ops.cuda_nvfp4_decoder_attention.cuda_kv_cache_append_(
        k_cache,
        v_cache,
        new_k,
        new_v,
        past_length,
    )


def _bf16_bits(values: torch.Tensor) -> torch.Tensor:
    assert values.dtype == torch.bfloat16
    return values.contiguous().view(torch.int16)


def _assert_bitwise_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(_bf16_bits(actual), _bf16_bits(expected))


def _generated_new_values(
    shape: tuple[int, int, int, int],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    new_k = (
        torch.randn(shape, generator=generator) * 0.75
    ).to(torch.bfloat16).cuda()
    new_v = (
        torch.randn(shape, generator=generator) * 1.25
    ).to(torch.bfloat16).cuda()
    return new_k, new_v


def _valid_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    k_cache = torch.full(
        (1, 2, 5, 4), -11.0, dtype=torch.bfloat16, device="cuda"
    )
    v_cache = torch.full_like(k_cache, 13.0)
    new_k = torch.arange(
        16, dtype=torch.float32, device="cuda"
    ).reshape(1, 2, 2, 4).to(torch.bfloat16)
    new_v = -new_k.clone()
    return k_cache, v_cache, new_k, new_v


def _assert_complete_state_transition(
    *,
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
        -37.5,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.full_like(k_cache, 43.25)

    # Give the prefix values distinct from both sentinels and appended data.
    if past_length > 0:
        prefix_count = batch_size * kv_head_count * past_length * head_dim
        prefix_values = (
            torch.arange(prefix_count, dtype=torch.float32, device="cuda")
            .remainder(251)
            .sub(125)
            .reshape(batch_size, kv_head_count, past_length, head_dim)
            .to(torch.bfloat16)
        )
        k_cache[:, :, :past_length, :].copy_(prefix_values)
        v_cache[:, :, :past_length, :].copy_(-prefix_values)

    new_k, new_v = _generated_new_values(
        (batch_size, token_count, kv_head_count, head_dim),
        seed,
    )
    original_k = k_cache.clone()
    original_v = v_cache.clone()
    original_new_k = new_k.clone()
    original_new_v = new_v.clone()
    expected_k = original_k.clone()
    expected_v = original_v.clone()
    expected_k[:, :, past_length : past_length + token_count, :] = (
        new_k.permute(0, 2, 1, 3)
    )
    expected_v[:, :, past_length : past_length + token_count, :] = (
        new_v.permute(0, 2, 1, 3)
    )

    result = _call(k_cache, v_cache, new_k, new_v, past_length)

    assert result is None
    _assert_bitwise_equal(k_cache, expected_k)
    _assert_bitwise_equal(v_cache, expected_v)
    _assert_bitwise_equal(
        k_cache[:, :, :past_length, :],
        original_k[:, :, :past_length, :],
    )
    _assert_bitwise_equal(
        v_cache[:, :, :past_length, :],
        original_v[:, :, :past_length, :],
    )
    _assert_bitwise_equal(
        k_cache[:, :, past_length + token_count :, :],
        original_k[:, :, past_length + token_count :, :],
    )
    _assert_bitwise_equal(
        v_cache[:, :, past_length + token_count :, :],
        original_v[:, :, past_length + token_count :, :],
    )
    _assert_bitwise_equal(new_k, original_new_k)
    _assert_bitwise_equal(new_v, original_new_v)

    expected_logical_k = torch.cat(
        (
            original_k[:, :, :past_length, :],
            new_k.permute(0, 2, 1, 3),
        ),
        dim=2,
    )
    expected_logical_v = torch.cat(
        (
            original_v[:, :, :past_length, :],
            new_v.permute(0, 2, 1, 3),
        ),
        dim=2,
    )
    _assert_bitwise_equal(
        k_cache[:, :, : past_length + token_count, :],
        expected_logical_k,
    )
    _assert_bitwise_equal(
        v_cache[:, :, : past_length + token_count, :],
        expected_logical_v,
    )


def test_hand_computable_token_major_to_cache_major_layout() -> None:
    k_cache = torch.full(
        (1, 2, 4, 2), -99.0, dtype=torch.bfloat16, device="cuda"
    )
    v_cache = torch.full_like(k_cache, 77.0)
    new_k = torch.tensor(
        [[[[10.0, 11.0], [20.0, 21.0]],
          [[30.0, 31.0], [40.0, 41.0]]]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    new_v = torch.tensor(
        [[[[110.0, 111.0], [120.0, 121.0]],
          [[130.0, 131.0], [140.0, 141.0]]]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected_k = k_cache.clone()
    expected_v = v_cache.clone()
    expected_k[0, 0, 1] = torch.tensor(
        [10.0, 11.0], dtype=torch.bfloat16, device="cuda"
    )
    expected_k[0, 1, 1] = torch.tensor(
        [20.0, 21.0], dtype=torch.bfloat16, device="cuda"
    )
    expected_k[0, 0, 2] = torch.tensor(
        [30.0, 31.0], dtype=torch.bfloat16, device="cuda"
    )
    expected_k[0, 1, 2] = torch.tensor(
        [40.0, 41.0], dtype=torch.bfloat16, device="cuda"
    )
    expected_v[0, 0, 1] = torch.tensor(
        [110.0, 111.0], dtype=torch.bfloat16, device="cuda"
    )
    expected_v[0, 1, 1] = torch.tensor(
        [120.0, 121.0], dtype=torch.bfloat16, device="cuda"
    )
    expected_v[0, 0, 2] = torch.tensor(
        [130.0, 131.0], dtype=torch.bfloat16, device="cuda"
    )
    expected_v[0, 1, 2] = torch.tensor(
        [140.0, 141.0], dtype=torch.bfloat16, device="cuda"
    )

    _call(k_cache, v_cache, new_k, new_v, past_length=1)

    _assert_bitwise_equal(k_cache, expected_k)
    _assert_bitwise_equal(v_cache, expected_v)


def test_batch_two_isolation_with_distinguishable_values() -> None:
    batch_size, token_count, kv_head_count, head_dim = 2, 3, 2, 4
    past_length, capacity = 2, 8
    k_cache = torch.empty(
        (batch_size, kv_head_count, capacity, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.empty_like(k_cache)
    for batch in range(batch_size):
        for head in range(kv_head_count):
            k_cache[batch, head].fill_(-100.0 - 10.0 * batch - head)
            v_cache[batch, head].fill_(100.0 + 10.0 * batch + head)

    new_k = torch.empty(
        (batch_size, token_count, kv_head_count, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    new_v = torch.empty_like(new_k)
    dimensions = torch.arange(head_dim, dtype=torch.float32, device="cuda")
    for batch in range(batch_size):
        for token in range(token_count):
            for head in range(kv_head_count):
                base = 1000 * batch + 100 * token + 10 * head
                new_k[batch, token, head] = (base + dimensions).to(
                    torch.bfloat16
                )
                new_v[batch, token, head] = (-base - dimensions - 1).to(
                    torch.bfloat16
                )

    expected_k = k_cache.clone()
    expected_v = v_cache.clone()
    expected_k[:, :, past_length : past_length + token_count, :] = (
        new_k.permute(0, 2, 1, 3)
    )
    expected_v[:, :, past_length : past_length + token_count, :] = (
        new_v.permute(0, 2, 1, 3)
    )

    _call(k_cache, v_cache, new_k, new_v, past_length)

    _assert_bitwise_equal(k_cache, expected_k)
    _assert_bitwise_equal(v_cache, expected_v)
    for batch in range(batch_size):
        for token in range(token_count):
            for head in range(kv_head_count):
                _assert_bitwise_equal(
                    k_cache[batch, head, past_length + token],
                    new_k[batch, token, head],
                )
                _assert_bitwise_equal(
                    v_cache[batch, head, past_length + token],
                    new_v[batch, token, head],
                )


@pytest.mark.parametrize(
    ("past_length", "token_count", "capacity"),
    (
        (0, 1, 5),
        (0, 4, 9),
        (1, 3, 9),
        (128, 1, 160),
        (128, 4, 1024),
        (4, 3, 7),
    ),
    ids=(
        "p0-t1-spare",
        "p0-t4",
        "p1-t3",
        "p128-t1",
        "p128-t4-large-spare",
        "exact-capacity",
    ),
)
def test_multiple_p_t_capacity_state_transitions(
    past_length: int,
    token_count: int,
    capacity: int,
) -> None:
    _assert_complete_state_transition(
        batch_size=1,
        token_count=token_count,
        kv_head_count=3,
        head_dim=8,
        past_length=past_length,
        capacity=capacity,
        seed=70_001 + past_length * 17 + token_count,
    )


@pytest.mark.parametrize(
    ("token_count", "past_length", "capacity"),
    ((1, 2048, 8192), (4, 128, 512)),
    ids=("canonical-p2048-c8192", "canonical-t4-p128"),
)
def test_canonical_hkv6_d128_and_large_capacity(
    token_count: int,
    past_length: int,
    capacity: int,
) -> None:
    _assert_complete_state_transition(
        batch_size=1,
        token_count=token_count,
        kv_head_count=6,
        head_dim=128,
        past_length=past_length,
        capacity=capacity,
        seed=71_003 + past_length,
    )


def test_signed_zero_and_raw_bf16_patterns_are_preserved() -> None:
    patterns = torch.tensor(
        [
            0x0000,
            0x8000,
            0x3F80,
            0xBF80,
            0x4000,
            0xC020,
            0x3E80,
            0xBE80,
        ],
        dtype=torch.int32,
    ).to(torch.int16)
    new_k = patterns.view(torch.bfloat16).reshape(1, 2, 1, 4).cuda()
    new_v = patterns.flip(0).view(torch.bfloat16).reshape(1, 2, 1, 4).cuda()
    original_new_k = new_k.clone()
    original_new_v = new_v.clone()
    k_cache = torch.full(
        (1, 1, 5, 4), 9.0, dtype=torch.bfloat16, device="cuda"
    )
    v_cache = torch.full_like(k_cache, -9.0)
    original_k = k_cache.clone()
    original_v = v_cache.clone()

    _call(k_cache, v_cache, new_k, new_v, past_length=2)

    _assert_bitwise_equal(k_cache[:, :, 2:4, :], new_k.permute(0, 2, 1, 3))
    _assert_bitwise_equal(v_cache[:, :, 2:4, :], new_v.permute(0, 2, 1, 3))
    _assert_bitwise_equal(k_cache[:, :, :2, :], original_k[:, :, :2, :])
    _assert_bitwise_equal(k_cache[:, :, 4:, :], original_k[:, :, 4:, :])
    _assert_bitwise_equal(v_cache[:, :, :2, :], original_v[:, :, :2, :])
    _assert_bitwise_equal(v_cache[:, :, 4:, :], original_v[:, :, 4:, :])
    _assert_bitwise_equal(new_k, original_new_k)
    _assert_bitwise_equal(new_v, original_new_v)
    assert int(_bf16_bits(k_cache[0, 0, 2, 0]).item()) == 0
    assert int(_bf16_bits(k_cache[0, 0, 2, 1]).item()) == -32768


def test_in_place_storage_identity_inputs_and_none_return() -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    k_object_id = id(k_cache)
    v_object_id = id(v_cache)
    k_data_ptr = k_cache.data_ptr()
    v_data_ptr = v_cache.data_ptr()
    k_storage_ptr = k_cache.untyped_storage().data_ptr()
    v_storage_ptr = v_cache.untyped_storage().data_ptr()
    original_new_k = new_k.clone()
    original_new_v = new_v.clone()

    result = _call(k_cache, v_cache, new_k, new_v, past_length=1)
    torch.cuda.synchronize()

    assert result is None
    assert id(k_cache) == k_object_id
    assert id(v_cache) == v_object_id
    assert k_cache.data_ptr() == k_data_ptr
    assert v_cache.data_ptr() == v_data_ptr
    assert k_cache.untyped_storage().data_ptr() == k_storage_ptr
    assert v_cache.untyped_storage().data_ptr() == v_storage_ptr
    _assert_bitwise_equal(new_k, original_new_k)
    _assert_bitwise_equal(new_v, original_new_v)


def test_raw_operator_schema_explicitly_marks_both_caches_mutable() -> None:
    schema = (
        torch.ops.cuda_nvfp4_decoder_attention.cuda_kv_cache_append_
        .default._schema
    )
    assert schema.is_mutable
    assert str(schema) == (
        "cuda_nvfp4_decoder_attention::cuda_kv_cache_append_("
        "Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor new_k, Tensor new_v, "
        "int past_length) -> ()"
    )
    arguments = {argument.name: argument for argument in schema.arguments}
    assert arguments["k_cache"].alias_info is not None
    assert arguments["k_cache"].alias_info.is_write
    assert arguments["v_cache"].alias_info is not None
    assert arguments["v_cache"].alias_info.is_write
    assert arguments["new_k"].alias_info is None
    assert arguments["new_v"].alias_info is None


def test_non_default_current_stream_orders_writes_and_consumer() -> None:
    batch_size, token_count, kv_head_count, head_dim = 2, 3, 2, 8
    past_length, capacity = 3, 12
    target_k, target_v = _generated_new_values(
        (batch_size, token_count, kv_head_count, head_dim),
        72_011,
    )
    k_cache = torch.empty(
        (batch_size, kv_head_count, capacity, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.empty_like(k_cache)
    new_k = torch.empty_like(target_k)
    new_v = torch.empty_like(target_v)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        k_cache.fill_(-31.0)
        v_cache.fill_(47.0)
        new_k.copy_(target_k)
        new_v.copy_(target_v)
        result = _call(k_cache, v_cache, new_k, new_v, past_length)
        consumed_k = k_cache[
            :, :, past_length : past_length + token_count, :
        ].clone()
        consumed_v = v_cache[
            :, :, past_length : past_length + token_count, :
        ].clone()
    stream.synchronize()

    assert result is None
    _assert_bitwise_equal(consumed_k, target_k.permute(0, 2, 1, 3))
    _assert_bitwise_equal(consumed_v, target_v.permute(0, 2, 1, 3))
    _assert_bitwise_equal(
        k_cache[:, :, :past_length, :],
        torch.full_like(k_cache[:, :, :past_length, :], -31.0),
    )
    _assert_bitwise_equal(
        v_cache[:, :, past_length + token_count :, :],
        torch.full_like(v_cache[:, :, past_length + token_count :, :], 47.0),
    )


@pytest.mark.parametrize("name", ("k_cache", "v_cache", "new_k", "new_v"))
def test_raw_operator_rejects_cpu_tensors(name: str) -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    values = {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "new_k": new_k,
        "new_v": new_v,
    }
    values[name] = values[name].cpu()
    with pytest.raises(RuntimeError, match=f"{name} must be a CUDA tensor"):
        _raw_call(
            values["k_cache"],
            values["v_cache"],
            values["new_k"],
            values["new_v"],
            1,
        )


@pytest.mark.parametrize("name", ("k_cache", "v_cache", "new_k", "new_v"))
def test_raw_operator_rejects_wrong_dtype(name: str) -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    values = {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "new_k": new_k,
        "new_v": new_v,
    }
    values[name] = values[name].float()
    with pytest.raises(RuntimeError, match=f"{name} must have dtype torch.bfloat16"):
        _raw_call(
            values["k_cache"],
            values["v_cache"],
            values["new_k"],
            values["new_v"],
            1,
        )


@pytest.mark.parametrize("name", ("k_cache", "v_cache", "new_k", "new_v"))
def test_raw_operator_rejects_wrong_rank(name: str) -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    values = {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "new_k": new_k,
        "new_v": new_v,
    }
    values[name] = values[name].reshape(-1, values[name].shape[-1])
    with pytest.raises(RuntimeError, match=f"{name} must have rank 4"):
        _raw_call(
            values["k_cache"],
            values["v_cache"],
            values["new_k"],
            values["new_v"],
            1,
        )


@pytest.mark.parametrize("name", ("k_cache", "v_cache", "new_k", "new_v"))
def test_raw_operator_rejects_noncontiguous_tensors(name: str) -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    values = {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "new_k": new_k,
        "new_v": new_v,
    }
    shape = values[name].shape
    backing = torch.empty(
        (*shape[:-1], shape[-1] * 2),
        dtype=torch.bfloat16,
        device="cuda",
    )
    values[name] = backing[..., ::2]
    assert values[name].shape == shape
    assert not values[name].is_contiguous()
    with pytest.raises(RuntimeError, match=f"{name} must be contiguous"):
        _raw_call(
            values["k_cache"],
            values["v_cache"],
            values["new_k"],
            values["new_v"],
            1,
        )


@pytest.mark.parametrize("name", ("k_cache", "v_cache", "new_k", "new_v"))
def test_raw_operator_rejects_empty_dimensions(name: str) -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    values = {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "new_k": new_k,
        "new_v": new_v,
    }
    shape = list(values[name].shape)
    shape[0] = 0
    values[name] = torch.empty(
        shape, dtype=torch.bfloat16, device="cuda"
    )
    with pytest.raises(RuntimeError, match=f"{name} dimensions must all be positive"):
        _raw_call(
            values["k_cache"],
            values["v_cache"],
            values["new_k"],
            values["new_v"],
            1,
        )


def test_raw_operator_rejects_cache_shape_mismatch() -> None:
    k_cache, _, new_k, new_v = _valid_inputs()
    v_cache = torch.empty(
        (1, 2, 6, 4), dtype=torch.bfloat16, device="cuda"
    )
    with pytest.raises(RuntimeError, match="v_cache must have the same shape"):
        _raw_call(k_cache, v_cache, new_k, new_v, 1)


def test_raw_operator_rejects_new_shape_mismatch() -> None:
    k_cache, v_cache, new_k, _ = _valid_inputs()
    new_v = torch.empty(
        (1, 3, 2, 4), dtype=torch.bfloat16, device="cuda"
    )
    with pytest.raises(RuntimeError, match="new_v must have the same shape"):
        _raw_call(k_cache, v_cache, new_k, new_v, 1)


def test_raw_operator_rejects_batch_mismatch() -> None:
    k_cache, v_cache, _, _ = _valid_inputs()
    new_k = torch.empty(
        (2, 2, 2, 4), dtype=torch.bfloat16, device="cuda"
    )
    new_v = torch.empty_like(new_k)
    with pytest.raises(RuntimeError, match="batch size must match"):
        _raw_call(k_cache, v_cache, new_k, new_v, 1)


def test_raw_operator_rejects_kv_head_mismatch() -> None:
    k_cache, v_cache, _, _ = _valid_inputs()
    new_k = torch.empty(
        (1, 2, 3, 4), dtype=torch.bfloat16, device="cuda"
    )
    new_v = torch.empty_like(new_k)
    with pytest.raises(RuntimeError, match="head count must match"):
        _raw_call(k_cache, v_cache, new_k, new_v, 1)


def test_raw_operator_rejects_head_dimension_mismatch() -> None:
    k_cache, v_cache, _, _ = _valid_inputs()
    new_k = torch.empty(
        (1, 2, 2, 8), dtype=torch.bfloat16, device="cuda"
    )
    new_v = torch.empty_like(new_k)
    with pytest.raises(RuntimeError, match="head dimension must match"):
        _raw_call(k_cache, v_cache, new_k, new_v, 1)


def test_raw_operator_rejects_negative_past_length() -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    with pytest.raises(RuntimeError, match="past_length must be nonnegative"):
        _raw_call(k_cache, v_cache, new_k, new_v, -1)


def test_raw_operator_rejects_past_plus_tokens_int64_overflow() -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    with pytest.raises(RuntimeError, match="overflows int64"):
        _raw_call(
            k_cache,
            v_cache,
            new_k,
            new_v,
            torch.iinfo(torch.int64).max,
        )


def test_raw_operator_rejects_insufficient_capacity() -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    with pytest.raises(RuntimeError, match="exceeds cache capacity"):
        _raw_call(k_cache, v_cache, new_k, new_v, 4)


@pytest.mark.parametrize("past_length", (True, 1.0, "1"))
def test_python_wrapper_rejects_non_integer_past_length(
    past_length: object,
) -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    with pytest.raises(TypeError, match="past_length must be an integer"):
        _call(
            k_cache,
            v_cache,
            new_k,
            new_v,
            past_length,  # type: ignore[arg-type]
        )


def test_raw_operator_rejects_past_length_outside_int64() -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    with pytest.raises(RuntimeError, match="Unable to cast"):
        _raw_call(k_cache, v_cache, new_k, new_v, 2**63)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("name", ("v_cache", "new_k", "new_v"))
def test_raw_operator_rejects_cross_device_inputs_when_testable(name: str) -> None:
    k_cache, v_cache, new_k, new_v = _valid_inputs()
    values = {
        "v_cache": v_cache,
        "new_k": new_k,
        "new_v": new_v,
    }
    values[name] = values[name].to("cuda:1")
    with pytest.raises(RuntimeError, match="same CUDA device"):
        _raw_call(
            k_cache,
            values["v_cache"],
            values["new_k"],
            values["new_v"],
            1,
        )
