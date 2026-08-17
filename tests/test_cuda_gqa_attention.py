"""Correctness and integration tests for the Milestone 5A CUDA GQA baseline."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from reference import gqa_attention_reference


CUDA_READY = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
pytestmark = pytest.mark.skipif(
    not CUDA_READY,
    reason="requires a CUDA device with PyTorch BF16 support",
)

if CUDA_READY:
    import cuda_primitives
else:
    cuda_primitives = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ContextMetrics:
    maximum_absolute_error: float
    mean_absolute_error: float
    exact_bf16_fraction: float
    maximum_bf16_adjacency_distance: int


# Context crosses a BF16 storage boundary. Normal-range discrepancies must be
# no farther than one adjacent BF16 value. Near cancellation, adjacency is not
# a useful magnitude metric, so a much tighter 2^-20 absolute floor applies.
ACCEPTED_MAXIMUM_ABSOLUTE_ERROR = 2.0**-7
ACCEPTED_MEAN_ABSOLUTE_ERROR = 2.0**-16
ADJACENCY_ABSOLUTE_FLOOR = 2.0**-20


def _call(
    q: torch.Tensor,
    present_k: torch.Tensor,
    present_v: torch.Tensor,
    past_length: int,
) -> torch.Tensor:
    assert cuda_primitives is not None
    return cuda_primitives.cuda_gqa_attention(
        q,
        present_k,
        present_v,
        past_length,
    )


def _raw_call(
    q: torch.Tensor,
    present_k: torch.Tensor,
    present_v: torch.Tensor,
    past_length: int,
) -> torch.Tensor:
    return torch.ops.cuda_nvfp4_decoder_attention.cuda_gqa_attention(
        q,
        present_k,
        present_v,
        past_length,
    )


def _bf16_ordered_keys(values: torch.Tensor) -> torch.Tensor:
    """Map finite BF16 values to monotonic integers, merging signed zeros."""

    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    magnitude = bits & 0x7FFF
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> ContextMetrics:
    assert actual.dtype == torch.bfloat16
    assert expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()

    error = (actual.float() - expected.float()).abs()
    adjacency = (
        _bf16_ordered_keys(actual) - _bf16_ordered_keys(expected)
    ).abs()
    return ContextMetrics(
        maximum_absolute_error=float(error.max().item()),
        mean_absolute_error=float(error.mean().item()),
        exact_bf16_fraction=float((actual == expected).float().mean().item()),
        maximum_bf16_adjacency_distance=int(adjacency.max().item()),
    )


def _assert_matches_reference(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> ContextMetrics:
    metrics = _metrics(actual, expected)
    error = (actual.float() - expected.float()).abs()
    adjacency = (
        _bf16_ordered_keys(actual) - _bf16_ordered_keys(expected)
    ).abs()
    assert metrics.maximum_absolute_error <= ACCEPTED_MAXIMUM_ABSOLUTE_ERROR, metrics
    assert metrics.mean_absolute_error <= ACCEPTED_MEAN_ABSOLUTE_ERROR, metrics
    assert torch.all(
        (adjacency <= 1) | (error <= ADJACENCY_ABSOLUTE_FLOOR)
    ), metrics
    return metrics


def _deterministic_case(
    *,
    batch_size: int = 1,
    token_count: int,
    query_head_count: int,
    kv_head_count: int,
    head_dim: int,
    past_length: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    context_length = past_length + token_count
    q = (
        torch.randn(
            (batch_size, token_count, query_head_count, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16)
    present_k = (
        torch.randn(
            (batch_size, kv_head_count, context_length, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16)
    present_v = (
        torch.randn(
            (batch_size, kv_head_count, context_length, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16)
    return q.cuda(), present_k.cuda(), present_v.cuda()


def _reference_context(
    q: torch.Tensor,
    present_k: torch.Tensor,
    present_v: torch.Tensor,
    past_length: int,
) -> torch.Tensor:
    _, _, context = gqa_attention_reference(
        q,
        present_k,
        present_v,
        past_length,
        return_attention=False,
    )
    return context


def _uniform_visible_hand_oracle(
    present_v: torch.Tensor,
    *,
    token_count: int,
    query_head_count: int,
    past_length: int,
) -> torch.Tensor:
    """Independent scalar oracle for zero-Q/K uniform visible attention."""

    present_v_cpu = present_v.cpu()
    batch_size, kv_head_count, _, head_dim = present_v_cpu.shape
    group_size = query_head_count // kv_head_count
    expected = torch.empty(
        (batch_size, token_count, query_head_count, head_dim),
        dtype=torch.bfloat16,
    )
    for batch in range(batch_size):
        for token in range(token_count):
            visible_count = past_length + token + 1
            for query_head in range(query_head_count):
                kv_head = query_head // group_size
                for d in range(head_dim):
                    total = sum(
                        float(present_v_cpu[batch, kv_head, j, d].float())
                        for j in range(visible_count)
                    )
                    expected[batch, token, query_head, d] = (
                        total / visible_count
                    )
    return expected


def _valid_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.ones((1, 1, 4, 8), dtype=torch.bfloat16, device="cuda")
    present_k = torch.ones(
        (1, 2, 2, 8), dtype=torch.bfloat16, device="cuda"
    )
    present_v = torch.ones_like(present_k)
    return q, present_k, present_v


def test_hand_computable_gqa_mapping_is_exact() -> None:
    q = torch.zeros((1, 1, 4, 2), dtype=torch.bfloat16, device="cuda")
    present_k = torch.tensor(
        [[[[1.0, -1.0]], [[7.0, 9.0]]]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    present_v = torch.tensor(
        [[[[1.0, 2.0]], [[11.0, 12.0]]]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = torch.tensor(
        [[[[1.0, 2.0], [1.0, 2.0], [11.0, 12.0], [11.0, 12.0]]]],
        dtype=torch.bfloat16,
        device="cuda",
    )

    actual = _call(q, present_k, present_v, past_length=0)

    assert torch.equal(actual, expected)
    assert torch.equal(
        _reference_context(q, present_k, present_v, 0),
        expected,
    )


def test_t_greater_than_one_causal_visibility_is_exact() -> None:
    past_length = 2
    token_count = 3
    query_head_count = 4
    kv_head_count = 2
    head_dim = 8
    q = torch.zeros(
        (1, token_count, query_head_count, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    present_k = torch.zeros(
        (1, kv_head_count, past_length + token_count, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    present_v = torch.empty_like(present_k)
    kv0_values = (3.0, 6.0, 9.0, 1_200.0, 24_000.0)
    kv1_values = (30.0, 60.0, 90.0, 12_000.0, 240_000.0)
    for key, value in enumerate(kv0_values):
        present_v[0, 0, key].fill_(value)
    for key, value in enumerate(kv1_values):
        present_v[0, 1, key].fill_(value)
    expected = _uniform_visible_hand_oracle(
        present_v,
        token_count=token_count,
        query_head_count=query_head_count,
        past_length=past_length,
    ).cuda()

    actual = _call(q, present_k, present_v, past_length)

    assert torch.equal(actual, expected)
    # For token zero, the future values 1,200/24,000 and 12,000/240,000 are
    # excluded; even tiny leakage would make this inspection-friendly row huge.
    assert torch.equal(actual[0, 0, 0], torch.full((head_dim,), 6.0, device="cuda", dtype=torch.bfloat16))
    assert torch.equal(actual[0, 0, 2], torch.full((head_dim,), 60.0, device="cuda", dtype=torch.bfloat16))


def test_current_token_can_attend_to_its_own_cache_slot() -> None:
    past_length = 1
    token_count = 2
    q = torch.zeros((1, token_count, 2, 8), dtype=torch.bfloat16, device="cuda")
    present_k = torch.zeros((1, 1, 3, 8), dtype=torch.bfloat16, device="cuda")
    present_v = torch.zeros_like(present_k)
    present_v[0, 0, 1].fill_(2.0)
    present_v[0, 0, 2].fill_(60.0)
    expected = _uniform_visible_hand_oracle(
        present_v,
        token_count=token_count,
        query_head_count=2,
        past_length=past_length,
    ).cuda()

    actual = _call(q, present_k, present_v, past_length)

    assert torch.equal(actual, expected)
    assert torch.equal(actual[0, 0], torch.ones_like(actual[0, 0]))
    assert torch.all(actual[0, 1] > 20.0)


def test_single_visible_key_returns_directly_mapped_v() -> None:
    q, present_k, present_v = _deterministic_case(
        token_count=1,
        query_head_count=8,
        kv_head_count=2,
        head_dim=32,
        past_length=0,
        seed=50_003,
    )
    expected = torch.empty_like(q)
    for query_head in range(8):
        expected[0, 0, query_head] = present_v[0, query_head // 4, 0]

    actual = _call(q, present_k, present_v, 0)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("query_head_count", "kv_head_count"),
    ((4, 4), (4, 2), (4, 1), (24, 6), (24, 3)),
    ids=("ratio-1", "ratio-2", "ratio-4", "canonical-24-to-6", "ratio-8"),
)
def test_multiple_gqa_ratios_match_direct_mapping_reference(
    query_head_count: int,
    kv_head_count: int,
) -> None:
    q, present_k, present_v = _deterministic_case(
        token_count=2,
        query_head_count=query_head_count,
        kv_head_count=kv_head_count,
        head_dim=16,
        past_length=1,
        seed=51_001 + query_head_count * 101 + kv_head_count,
    )
    expected = _reference_context(q, present_k, present_v, 1)

    actual = _call(q, present_k, present_v, 1)

    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize("head_dim", (8, 16, 32, 128, 320))
def test_head_dimensions_including_above_256_match_reference(head_dim: int) -> None:
    q, present_k, present_v = _deterministic_case(
        token_count=2,
        query_head_count=4,
        kv_head_count=2,
        head_dim=head_dim,
        past_length=3,
        seed=52_003 + head_dim,
    )
    expected = _reference_context(q, present_k, present_v, 3)

    actual = _call(q, present_k, present_v, 3)

    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize("past_length", (0, 128, 512, 2_048, 8_192))
def test_canonical_decode_lengths_match_reference(past_length: int) -> None:
    q, present_k, present_v = _deterministic_case(
        token_count=1,
        query_head_count=24,
        kv_head_count=6,
        head_dim=128,
        past_length=past_length,
        seed=53_009 + past_length,
    )
    expected = _reference_context(q, present_k, present_v, past_length)

    actual = _call(q, present_k, present_v, past_length)
    metrics = _assert_matches_reference(actual, expected)
    torch.cuda.synchronize()

    print(
        f"canonical past_length={past_length} "
        f"max_abs={metrics.maximum_absolute_error:.9g} "
        f"mean_abs={metrics.mean_absolute_error:.9g} "
        f"exact_fraction={metrics.exact_bf16_fraction:.9g} "
        f"max_bf16_distance={metrics.maximum_bf16_adjacency_distance}"
    )


@pytest.mark.parametrize("past_length", (0, 8))
def test_canonical_head_chunk_cases_match_reference(past_length: int) -> None:
    q, present_k, present_v = _deterministic_case(
        token_count=4,
        query_head_count=24,
        kv_head_count=6,
        head_dim=128,
        past_length=past_length,
        seed=54_011 + past_length,
    )
    expected = _reference_context(q, present_k, present_v, past_length)

    actual = _call(q, present_k, present_v, past_length)

    _assert_matches_reference(actual, expected)


def test_changing_past_length_changes_visibility_without_changing_mapping() -> None:
    q0 = torch.zeros((1, 1, 4, 8), dtype=torch.bfloat16, device="cuda")
    k0 = torch.zeros((1, 2, 1, 8), dtype=torch.bfloat16, device="cuda")
    v0 = torch.empty_like(k0)
    v0[0, 0, 0].fill_(10.0)
    v0[0, 1, 0].fill_(20.0)

    q1 = q0.clone()
    k1 = torch.zeros((1, 2, 2, 8), dtype=torch.bfloat16, device="cuda")
    v1 = torch.empty_like(k1)
    v1[0, 0, 0].fill_(0.0)
    v1[0, 0, 1].fill_(10.0)
    v1[0, 1, 0].fill_(100.0)
    v1[0, 1, 1].fill_(20.0)

    no_past = _call(q0, k0, v0, 0)
    one_past = _call(q1, k1, v1, 1)

    assert torch.equal(no_past[0, 0, 0], torch.full((8,), 10.0, dtype=torch.bfloat16, device="cuda"))
    assert torch.equal(no_past[0, 0, 2], torch.full((8,), 20.0, dtype=torch.bfloat16, device="cuda"))
    assert torch.equal(one_past[0, 0, 0], torch.full((8,), 5.0, dtype=torch.bfloat16, device="cuda"))
    assert torch.equal(one_past[0, 0, 2], torch.full((8,), 60.0, dtype=torch.bfloat16, device="cuda"))


def test_equal_scores_produce_observable_uniform_average() -> None:
    q = torch.zeros((1, 1, 2, 8), dtype=torch.bfloat16, device="cuda")
    present_k = torch.randn((1, 1, 4, 8), device="cuda").to(torch.bfloat16)
    present_v = torch.zeros_like(present_k)
    present_v[0, 0, 3].fill_(4.0)

    actual = _call(q, present_k, present_v, past_length=3)

    assert torch.equal(actual, torch.ones_like(actual))


def test_one_strongly_preferred_visible_key_dominates_context() -> None:
    q = torch.ones((1, 1, 2, 8), dtype=torch.bfloat16, device="cuda")
    present_k = torch.empty((1, 1, 2, 8), dtype=torch.bfloat16, device="cuda")
    present_k[0, 0, 0].fill_(-8.0)
    present_k[0, 0, 1].fill_(8.0)
    present_v = torch.empty_like(present_k)
    present_v[0, 0, 0].fill_(-10.0)
    present_v[0, 0, 1].fill_(10.0)

    actual = _call(q, present_k, present_v, past_length=1)

    assert torch.equal(actual, torch.full_like(actual, 10.0))


def test_future_key_with_dominating_score_and_value_is_masked() -> None:
    q = torch.ones((1, 2, 2, 8), dtype=torch.bfloat16, device="cuda")
    present_k = torch.zeros((1, 1, 2, 8), dtype=torch.bfloat16, device="cuda")
    present_k[0, 0, 1].fill_(32.0)
    present_v = torch.empty_like(present_k)
    present_v[0, 0, 0].fill_(3.0)
    present_v[0, 0, 1].fill_(20_000.0)

    actual = _call(q, present_k, present_v, past_length=0)

    assert torch.equal(actual[0, 0], torch.full_like(actual[0, 0], 3.0))
    assert torch.equal(actual[0, 1], torch.full_like(actual[0, 1], 20_000.0))


def test_moderate_dynamic_range_softmax_matches_reference() -> None:
    q = torch.zeros((1, 1, 4, 8), dtype=torch.bfloat16, device="cuda")
    q[..., 0] = 1.0
    present_k = torch.zeros((1, 2, 5, 8), dtype=torch.bfloat16, device="cuda")
    present_v = torch.zeros_like(present_k)
    for kv_head in range(2):
        present_k[0, kv_head, :, 0] = torch.tensor(
            [-4.0, -2.0, 0.0, 2.0, 4.0],
            dtype=torch.bfloat16,
            device="cuda",
        )
        present_v[0, kv_head, :, 0] = torch.tensor(
            [-3.0, 1.0, 2.0, 5.0, 9.0],
            dtype=torch.bfloat16,
            device="cuda",
        ) + 10 * kv_head
    expected = _reference_context(q, present_k, present_v, 4)

    actual = _call(q, present_k, present_v, 4)

    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize("zero_operand", ("q", "k"))
def test_zero_q_or_zero_k_matches_uniform_reference(zero_operand: str) -> None:
    q, present_k, present_v = _deterministic_case(
        token_count=3,
        query_head_count=4,
        kv_head_count=2,
        head_dim=16,
        past_length=2,
        seed=55_001,
    )
    if zero_operand == "q":
        q.zero_()
    else:
        present_k.zero_()
    expected = _reference_context(q, present_k, present_v, 2)

    actual = _call(q, present_k, present_v, 2)

    _assert_matches_reference(actual, expected)


def test_zero_v_produces_exact_zero_context() -> None:
    q, present_k, present_v = _deterministic_case(
        token_count=3,
        query_head_count=4,
        kv_head_count=1,
        head_dim=32,
        past_length=3,
        seed=56_003,
    )
    present_v.zero_()

    actual = _call(q, present_k, present_v, 3)

    assert torch.equal(actual, torch.zeros_like(actual))


def test_uniform_v_is_preserved_by_normalized_probabilities() -> None:
    q, present_k, present_v = _deterministic_case(
        token_count=4,
        query_head_count=4,
        kv_head_count=2,
        head_dim=16,
        past_length=4,
        seed=57_007,
    )
    for kv_head, value in enumerate((1.5, -2.0)):
        present_v[0, kv_head].fill_(value)
    expected = torch.empty_like(q)
    for query_head in range(4):
        expected[0, :, query_head].fill_((1.5, -2.0)[query_head // 2])

    actual = _call(q, present_k, present_v, 4)

    assert torch.equal(actual, expected)


def test_inputs_are_not_mutated_and_output_has_fresh_contiguous_storage() -> None:
    q, present_k, present_v = _deterministic_case(
        token_count=2,
        query_head_count=4,
        kv_head_count=2,
        head_dim=16,
        past_length=2,
        seed=58_009,
    )
    original_q = q.clone()
    original_k = present_k.clone()
    original_v = present_v.clone()

    actual = _call(q, present_k, present_v, 2)
    torch.cuda.synchronize()

    assert torch.equal(q, original_q)
    assert torch.equal(present_k, original_k)
    assert torch.equal(present_v, original_v)
    assert actual.shape == (1, 2, 4, 16)
    assert actual.dtype == torch.bfloat16
    assert actual.device == q.device
    assert actual.is_contiguous()
    assert actual.data_ptr() not in {
        q.data_ptr(),
        present_k.data_ptr(),
        present_v.data_ptr(),
    }


def test_non_default_current_stream_orders_inputs_all_kernels_and_consumer() -> None:
    target_q, target_k, target_v = _deterministic_case(
        token_count=4,
        query_head_count=24,
        kv_head_count=6,
        head_dim=128,
        past_length=8,
        seed=59_011,
    )
    expected = _reference_context(target_q, target_k, target_v, 8)
    q = torch.zeros_like(target_q)
    present_k = torch.zeros_like(target_k)
    present_v = torch.zeros_like(target_v)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        q.copy_(target_q)
        present_k.copy_(target_k)
        present_v.copy_(target_v)
        actual = _call(q, present_k, present_v, 8)
        downstream = actual.float().square().sum()
    stream.synchronize()

    _assert_matches_reference(actual, expected)
    torch.testing.assert_close(
        downstream.cpu(),
        expected.float().square().sum().cpu(),
        rtol=0.0,
        atol=1.0e-5,
    )


@pytest.mark.parametrize("name", ("q", "present_k", "present_v"))
def test_raw_operator_rejects_cpu_tensors(name: str) -> None:
    q, present_k, present_v = _valid_inputs()
    values = {"q": q, "present_k": present_k, "present_v": present_v}
    values[name] = values[name].cpu()
    with pytest.raises(RuntimeError, match=f"{name} must be a CUDA tensor"):
        _raw_call(values["q"], values["present_k"], values["present_v"], 1)


@pytest.mark.parametrize("name", ("q", "present_k", "present_v"))
def test_raw_operator_rejects_wrong_dtype(name: str) -> None:
    q, present_k, present_v = _valid_inputs()
    values = {"q": q, "present_k": present_k, "present_v": present_v}
    values[name] = values[name].float()
    with pytest.raises(RuntimeError, match=f"{name} must have dtype torch.bfloat16"):
        _raw_call(values["q"], values["present_k"], values["present_v"], 1)


@pytest.mark.parametrize("name", ("q", "present_k", "present_v"))
def test_raw_operator_rejects_noncontiguous_tensors(name: str) -> None:
    q, present_k, present_v = _valid_inputs()
    values = {"q": q, "present_k": present_k, "present_v": present_v}
    if name == "q":
        values[name] = torch.ones(
            (1, 4, 2, 8), dtype=torch.bfloat16, device="cuda"
        ).transpose(1, 2)
    else:
        values[name] = torch.ones(
            (1, 2, 8, 2), dtype=torch.bfloat16, device="cuda"
        ).transpose(2, 3)
    assert not values[name].is_contiguous()
    with pytest.raises(RuntimeError, match=f"{name} must be contiguous"):
        _raw_call(values["q"], values["present_k"], values["present_v"], 1)


@pytest.mark.parametrize("name", ("q", "present_k", "present_v"))
def test_raw_operator_rejects_wrong_rank(name: str) -> None:
    q, present_k, present_v = _valid_inputs()
    values = {"q": q, "present_k": present_k, "present_v": present_v}
    values[name] = values[name].reshape(-1, values[name].shape[-1])
    with pytest.raises(RuntimeError, match=f"{name} must have rank 4"):
        _raw_call(values["q"], values["present_k"], values["present_v"], 1)


@pytest.mark.parametrize("name", ("q", "present_k", "present_v"))
def test_raw_operator_rejects_empty_dimensions(name: str) -> None:
    q, present_k, present_v = _valid_inputs()
    values = {"q": q, "present_k": present_k, "present_v": present_v}
    if name == "q":
        values[name] = torch.empty(
            (1, 0, 4, 8), dtype=torch.bfloat16, device="cuda"
        )
    else:
        values[name] = torch.empty(
            (1, 2, 0, 8), dtype=torch.bfloat16, device="cuda"
        )
    with pytest.raises(RuntimeError, match=f"{name} dimensions must all be positive"):
        _raw_call(values["q"], values["present_k"], values["present_v"], 1)


def test_raw_operator_rejects_different_cache_shapes() -> None:
    q, present_k, _ = _valid_inputs()
    present_v = torch.ones(
        (1, 2, 3, 8), dtype=torch.bfloat16, device="cuda"
    )
    with pytest.raises(RuntimeError, match="present_v must have the same shape"):
        _raw_call(q, present_k, present_v, 1)


def test_raw_operator_rejects_batch_mismatch() -> None:
    q, present_k, present_v = _valid_inputs()
    present_k = present_k.expand(2, -1, -1, -1).contiguous()
    present_v = present_v.expand(2, -1, -1, -1).contiguous()
    with pytest.raises(RuntimeError, match="batch sizes must match"):
        _raw_call(q, present_k, present_v, 1)


def test_raw_operator_rejects_head_dimension_mismatch() -> None:
    q, _, _ = _valid_inputs()
    present_k = torch.ones(
        (1, 2, 2, 16), dtype=torch.bfloat16, device="cuda"
    )
    present_v = torch.ones_like(present_k)
    with pytest.raises(RuntimeError, match="head dimensions must match"):
        _raw_call(q, present_k, present_v, 1)


def test_raw_operator_rejects_nondivisible_gqa() -> None:
    q = torch.ones((1, 1, 5, 8), dtype=torch.bfloat16, device="cuda")
    present_k = torch.ones(
        (1, 2, 2, 8), dtype=torch.bfloat16, device="cuda"
    )
    present_v = torch.ones_like(present_k)
    with pytest.raises(RuntimeError, match="must be divisible"):
        _raw_call(q, present_k, present_v, 1)


def test_raw_operator_rejects_negative_past_length() -> None:
    q, present_k, present_v = _valid_inputs()
    with pytest.raises(RuntimeError, match="past_length must be nonnegative"):
        _raw_call(q, present_k, present_v, -1)


def test_raw_operator_rejects_past_plus_tokens_int64_overflow() -> None:
    q, present_k, present_v = _valid_inputs()
    with pytest.raises(RuntimeError, match="overflows int64"):
        _raw_call(q, present_k, present_v, torch.iinfo(torch.int64).max)


def test_raw_operator_rejects_context_length_not_equal_to_p_plus_t() -> None:
    q, present_k, present_v = _valid_inputs()
    with pytest.raises(RuntimeError, match="cache context length must equal"):
        _raw_call(q, present_k, present_v, 0)


@pytest.mark.parametrize("past_length", (True, 1.0, "1"))
def test_python_wrapper_rejects_non_integer_past_length(
    past_length: object,
) -> None:
    q, present_k, present_v = _valid_inputs()
    with pytest.raises(TypeError, match="past_length must be an integer"):
        _call(q, present_k, present_v, past_length)  # type: ignore[arg-type]


def test_raw_operator_rejects_past_length_outside_int64() -> None:
    q, present_k, present_v = _valid_inputs()
    with pytest.raises(RuntimeError, match="Unable to cast"):
        _raw_call(q, present_k, present_v, 2**63)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("name", ("present_k", "present_v"))
def test_raw_operator_rejects_device_mismatch_when_testable(name: str) -> None:
    q, present_k, present_v = _valid_inputs()
    values = {"present_k": present_k, "present_v": present_v}
    values[name] = values[name].to("cuda:1")
    with pytest.raises(RuntimeError, match="same CUDA device"):
        _raw_call(q, values["present_k"], values["present_v"], 1)
