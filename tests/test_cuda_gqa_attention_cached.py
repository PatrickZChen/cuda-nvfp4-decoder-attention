"""Correctness tests for the M5C capacity-aware CUDA GQA primitive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


ACCEPTED_MAXIMUM_ABSOLUTE_ERROR = 2.0**-7
ACCEPTED_MEAN_ABSOLUTE_ERROR = 2.0**-16
ADJACENCY_ABSOLUTE_FLOOR = 2.0**-20


def _call(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    past_length: int,
) -> torch.Tensor:
    assert cuda_primitives is not None
    return cuda_primitives.cuda_gqa_attention_cached(
        q,
        k_cache,
        v_cache,
        past_length,
    )


def _raw_call(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    past_length: int,
) -> torch.Tensor:
    return torch.ops.cuda_nvfp4_decoder_attention.cuda_gqa_attention_cached(
        q,
        k_cache,
        v_cache,
        past_length,
    )


def _bf16_ordered_keys(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    magnitude = bits & 0x7FFF
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> ContextMetrics:
    assert actual.dtype == expected.dtype == torch.bfloat16
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


def _generated_case(
    *,
    batch_size: int,
    token_count: int,
    query_head_count: int,
    kv_head_count: int,
    head_dim: int,
    past_length: int,
    capacity: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logical_context_length = past_length + token_count
    assert logical_context_length <= capacity
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = (
        torch.randn(
            (batch_size, token_count, query_head_count, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16)
    k_prefix = (
        torch.randn(
            (batch_size, kv_head_count, logical_context_length, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16)
    v_prefix = (
        torch.randn(
            (batch_size, kv_head_count, logical_context_length, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16)
    k_cache = torch.full(
        (batch_size, kv_head_count, capacity, head_dim),
        91.0,
        dtype=torch.bfloat16,
    )
    v_cache = torch.full_like(k_cache, -87.0)
    k_cache[:, :, :logical_context_length].copy_(k_prefix)
    v_cache[:, :, :logical_context_length].copy_(v_prefix)
    return q.cuda(), k_cache.cuda(), v_cache.cuda()


def _compact_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    past_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logical_context_length = past_length + q.size(1)
    # Compaction is deliberately confined to the test oracle.
    compact_k = k_cache[:, :, :logical_context_length, :].clone()
    compact_v = v_cache[:, :, :logical_context_length, :].clone()
    _, _, reference = gqa_attention_reference(
        q,
        compact_k,
        compact_v,
        past_length,
        return_attention=False,
    )
    assert cuda_primitives is not None
    compact_cuda = cuda_primitives.cuda_gqa_attention(
        q,
        compact_k,
        compact_v,
        past_length,
    )
    return compact_k, compact_v, reference, compact_cuda


def _assert_cached_compact_and_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    past_length: int,
) -> ContextMetrics:
    _, _, expected, compact_actual = _compact_reference(
        q,
        k_cache,
        v_cache,
        past_length,
    )
    cached_actual = _call(q, k_cache, v_cache, past_length)
    assert torch.equal(cached_actual, compact_actual)
    return _assert_matches_reference(cached_actual, expected)


def _valid_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.ones((1, 2, 4, 8), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.ones((1, 2, 7, 8), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.ones_like(k_cache)
    return q, k_cache, v_cache


def test_hand_physical_v_stride_uses_capacity_not_logical_length() -> None:
    past_length, token_count, capacity = 1, 1, 7
    q = torch.zeros((1, token_count, 4, 8), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.full(
        (1, 2, capacity, 8), 123.0, dtype=torch.bfloat16, device="cuda"
    )
    k_cache[:, :, :2].zero_()
    v_cache = torch.full_like(k_cache, -12_000.0)
    v_cache[0, 0, 0].fill_(2.0)
    v_cache[0, 0, 1].fill_(4.0)
    v_cache[0, 1, 0].fill_(20.0)
    v_cache[0, 1, 1].fill_(40.0)
    expected = torch.empty_like(q)
    expected[0, 0, 0:2].fill_(3.0)
    expected[0, 0, 2:4].fill_(30.0)

    actual = _call(q, k_cache, v_cache, past_length)

    assert torch.equal(actual, expected)
    # An S-based head stride would read head-0 sentinel slots 2:4 here.
    assert not torch.any(actual == torch.tensor(-12_000.0, device="cuda"))


def test_hand_physical_k_stride_uses_capacity_not_logical_length() -> None:
    past_length, token_count, capacity = 1, 1, 7
    q = torch.zeros((1, token_count, 4, 8), dtype=torch.bfloat16, device="cuda")
    q[0, 0, 2:4].fill_(1.0)
    k_cache = torch.zeros(
        (1, 2, capacity, 8), dtype=torch.bfloat16, device="cuda"
    )
    # Correct KV head 1 strongly selects key 1.
    k_cache[0, 1, 0].fill_(-8.0)
    k_cache[0, 1, 1].fill_(8.0)
    # The wrong S-based head base lands at head-0 slots 2:4 and reverses it.
    k_cache[0, 0, 2].fill_(8.0)
    k_cache[0, 0, 3].fill_(-8.0)
    v_cache = torch.zeros_like(k_cache)
    v_cache[0, 1, 1].fill_(10.0)
    # Make a wrong V stride read the same pair so this test isolates QK stride.
    v_cache[0, 0, 2].fill_(0.0)
    v_cache[0, 0, 3].fill_(10.0)

    actual = _call(q, k_cache, v_cache, past_length)

    assert torch.equal(actual[0, 0, 2:4], torch.full_like(actual[0, 0, 2:4], 10.0))
    _assert_cached_compact_and_reference(q, k_cache, v_cache, past_length)


def test_batch_two_capacity_and_head_strides_match_independent_oracle() -> None:
    batch_size, token_count, query_heads, kv_heads, head_dim = 2, 3, 4, 2, 8
    past_length, capacity = 2, 19
    logical_context_length = past_length + token_count
    q = torch.zeros(
        (batch_size, token_count, query_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_cache = torch.full(
        (batch_size, kv_heads, capacity, head_dim),
        9_000.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_cache[:, :, :logical_context_length].zero_()
    v_cache = torch.full_like(k_cache, -7_000.0)
    for batch in range(batch_size):
        for kv_head in range(kv_heads):
            for position in range(logical_context_length):
                value = 100.0 * batch + 10.0 * kv_head + position
                v_cache[batch, kv_head, position].fill_(value)

    expected = torch.empty_like(q)
    group_size = query_heads // kv_heads
    for batch in range(batch_size):
        for token in range(token_count):
            visible = past_length + token + 1
            for query_head in range(query_heads):
                kv_head = query_head // group_size
                values = [
                    100.0 * batch + 10.0 * kv_head + position
                    for position in range(visible)
                ]
                expected[batch, token, query_head].fill_(sum(values) / visible)

    actual = _call(q, k_cache, v_cache, past_length)

    assert torch.equal(actual, expected)
    assert not torch.equal(actual[0], actual[1])
    _assert_cached_compact_and_reference(q, k_cache, v_cache, past_length)


CASES = (
    # label, B, T, Hq, Hkv, D, P, C
    ("p0-t1-ratio1-spare", 1, 1, 1, 1, 8, 0, 7),
    ("p0-t4-ratio2", 1, 4, 2, 1, 32, 0, 11),
    ("p3-t2-ratio4-cwide", 1, 2, 4, 1, 8, 3, 97),
    ("p128-t1-ratio2", 1, 1, 4, 2, 32, 128, 512),
    ("p128-t4-ratio4-d128", 1, 4, 4, 1, 128, 128, 1024),
    ("b2-t2-ratio4", 2, 2, 8, 2, 32, 3, 257),
    ("canonical-p2048-c8192", 1, 1, 24, 6, 128, 2048, 8192),
    ("canonical-small-s-c8192", 1, 1, 24, 6, 128, 3, 8192),
)


@pytest.mark.parametrize(
    ("label", "batch_size", "token_count", "query_heads", "kv_heads", "head_dim", "past_length", "capacity"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_capacity_matrix_matches_m5a_and_reference(
    label: str,
    batch_size: int,
    token_count: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    past_length: int,
    capacity: int,
) -> None:
    q, k_cache, v_cache = _generated_case(
        batch_size=batch_size,
        token_count=token_count,
        query_head_count=query_heads,
        kv_head_count=kv_heads,
        head_dim=head_dim,
        past_length=past_length,
        capacity=capacity,
        seed=90_001 + len(label) * 101 + past_length,
    )
    original_k = k_cache.clone()
    original_v = v_cache.clone()

    metrics = _assert_cached_compact_and_reference(
        q,
        k_cache,
        v_cache,
        past_length,
    )

    assert torch.equal(k_cache, original_k)
    assert torch.equal(v_cache, original_v)
    print(
        f"cached-gqa case={label} max_abs={metrics.maximum_absolute_error:.9g} "
        f"mean_abs={metrics.mean_absolute_error:.9g} "
        f"exact_fraction={metrics.exact_bf16_fraction:.9g} "
        f"max_bf16_distance={metrics.maximum_bf16_adjacency_distance}"
    )


def test_future_chunk_slots_are_physically_present_but_causally_hidden() -> None:
    past_length, token_count, capacity = 1, 3, 17
    q = torch.ones((1, token_count, 4, 8), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros((1, 2, capacity, 8), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    k_cache[:, :, 2].fill_(32.0)
    k_cache[:, :, 3].fill_(64.0)
    v_cache[:, :, 0].fill_(2.0)
    v_cache[:, :, 1].fill_(4.0)
    v_cache[:, :, 2].fill_(20_000.0)
    v_cache[:, :, 3].fill_(30_000.0)

    actual = _call(q, k_cache, v_cache, past_length)

    assert torch.equal(actual[:, 0], torch.full_like(actual[:, 0], 3.0))
    assert torch.all(actual[:, 1] > 19_000.0)
    assert torch.all(actual[:, 2] > 29_000.0)


def test_output_is_fresh_contiguous_and_inputs_are_not_mutated() -> None:
    q, k_cache, v_cache = _generated_case(
        batch_size=2,
        token_count=2,
        query_head_count=4,
        kv_head_count=2,
        head_dim=32,
        past_length=3,
        capacity=67,
        seed=91_007,
    )
    original_q = q.clone()
    original_k = k_cache.clone()
    original_v = v_cache.clone()

    actual = _call(q, k_cache, v_cache, 3)
    torch.cuda.synchronize()

    assert torch.equal(q, original_q)
    assert torch.equal(k_cache, original_k)
    assert torch.equal(v_cache, original_v)
    assert actual.shape == q.shape
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert actual.data_ptr() not in {q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr()}


def test_non_default_current_stream_orders_preparation_all_stages_and_consumer() -> None:
    target_q, target_k, target_v = _generated_case(
        batch_size=2,
        token_count=3,
        query_head_count=8,
        kv_head_count=2,
        head_dim=32,
        past_length=5,
        capacity=257,
        seed=92_011,
    )
    _, _, expected, compact_actual = _compact_reference(
        target_q,
        target_k,
        target_v,
        5,
    )
    q = torch.zeros_like(target_q)
    k_cache = torch.zeros_like(target_k)
    v_cache = torch.zeros_like(target_v)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        q.copy_(target_q)
        k_cache.copy_(target_k)
        v_cache.copy_(target_v)
        actual = _call(q, k_cache, v_cache, 5)
        downstream = actual.float().square().sum()
    stream.synchronize()

    _assert_matches_reference(actual, expected)
    assert torch.equal(downstream, compact_actual.float().square().sum())


@pytest.mark.parametrize("name", ("q", "k_cache", "v_cache"))
def test_raw_operator_rejects_cpu_dtype_rank_contiguity_and_empty(name: str) -> None:
    q, k_cache, v_cache = _valid_inputs()
    base = {"q": q, "k_cache": k_cache, "v_cache": v_cache}

    cpu = dict(base)
    cpu[name] = cpu[name].cpu()
    with pytest.raises(RuntimeError, match=f"{name} must be a CUDA tensor"):
        _raw_call(cpu["q"], cpu["k_cache"], cpu["v_cache"], 1)

    wrong_dtype = dict(base)
    wrong_dtype[name] = wrong_dtype[name].float()
    with pytest.raises(RuntimeError, match=f"{name} must have dtype torch.bfloat16"):
        _raw_call(
            wrong_dtype["q"],
            wrong_dtype["k_cache"],
            wrong_dtype["v_cache"],
            1,
        )

    wrong_rank = dict(base)
    wrong_rank[name] = wrong_rank[name].reshape(-1, wrong_rank[name].shape[-1])
    with pytest.raises(RuntimeError, match=f"{name} must have rank 4"):
        _raw_call(wrong_rank["q"], wrong_rank["k_cache"], wrong_rank["v_cache"], 1)

    noncontiguous = dict(base)
    backing = torch.empty(
        (*noncontiguous[name].shape[:-1], noncontiguous[name].shape[-1] * 2),
        dtype=torch.bfloat16,
        device="cuda",
    )
    noncontiguous[name] = backing[..., ::2]
    assert not noncontiguous[name].is_contiguous()
    with pytest.raises(RuntimeError, match=f"{name} must be contiguous"):
        _raw_call(
            noncontiguous["q"],
            noncontiguous["k_cache"],
            noncontiguous["v_cache"],
            1,
        )

    empty = dict(base)
    shape = list(empty[name].shape)
    shape[0] = 0
    empty[name] = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match=f"{name} dimensions must all be positive"):
        _raw_call(empty["q"], empty["k_cache"], empty["v_cache"], 1)


def test_raw_operator_rejects_shape_relationships_and_invalid_length() -> None:
    q, k_cache, v_cache = _valid_inputs()
    with pytest.raises(RuntimeError, match="v_cache must have the same shape"):
        _raw_call(q, k_cache, torch.ones((1, 2, 8, 8), dtype=torch.bfloat16, device="cuda"), 1)
    with pytest.raises(RuntimeError, match="batch sizes must match"):
        _raw_call(q, k_cache.expand(2, -1, -1, -1).contiguous(), v_cache.expand(2, -1, -1, -1).contiguous(), 1)
    with pytest.raises(RuntimeError, match="head dimensions must match"):
        bad = torch.ones((1, 2, 7, 16), dtype=torch.bfloat16, device="cuda")
        _raw_call(q, bad, bad.clone(), 1)
    with pytest.raises(RuntimeError, match="must be divisible"):
        bad_q = torch.ones((1, 2, 5, 8), dtype=torch.bfloat16, device="cuda")
        _raw_call(bad_q, k_cache, v_cache, 1)
    with pytest.raises(RuntimeError, match="past_length must be nonnegative"):
        _raw_call(q, k_cache, v_cache, -1)
    with pytest.raises(RuntimeError, match="overflows int64"):
        _raw_call(q, k_cache, v_cache, torch.iinfo(torch.int64).max)
    with pytest.raises(RuntimeError, match="exceeds cache capacity"):
        _raw_call(q, k_cache, v_cache, 6)


@pytest.mark.parametrize("past_length", (True, 1.0, "1"))
def test_python_wrapper_rejects_non_integer_past_length(past_length: object) -> None:
    q, k_cache, v_cache = _valid_inputs()
    with pytest.raises(TypeError, match="past_length must be an integer"):
        _call(q, k_cache, v_cache, past_length)  # type: ignore[arg-type]


def test_production_cached_attention_contains_no_prefix_materialization() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    sources = (
        (repository_root / "src" / "gqa_attention_cached.cu").read_text(),
        (repository_root / "src" / "bindings.cpp").read_text(),
        (repository_root / "cuda_primitives.py").read_text(),
    )
    cached_source = sources[0]
    production_source = "\n".join(sources)
    for forbidden in ("torch.cat", "at::cat", ".contiguous()", "clone("):
        assert forbidden not in production_source
    assert "launch_gqa_attention_cuda" not in cached_source
    assert "cache_capacity + key_index" in cached_source
    assert "cache_capacity * head_dim" in cached_source
