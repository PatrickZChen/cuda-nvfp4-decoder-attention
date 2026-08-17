"""Correctness and integration tests for the Milestone 3B CUDA RoPE."""

from __future__ import annotations

import math

import pytest
import torch

from reference import apply_rope_reference, rms_norm_reference


CUDA_READY = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
pytestmark = pytest.mark.skipif(
    not CUDA_READY,
    reason="requires a CUDA device with PyTorch BF16 support",
)

if CUDA_READY:
    import cuda_primitives
else:
    cuda_primitives = None  # type: ignore[assignment]


# The deterministic RTX 4080 / SM89 matrix was observed to be BF16 bit-exact
# against the same-device frozen reference. A nonzero result is investigated.
ACCEPTED_MAX_ABSOLUTE_ERROR = 0.0


ROPE_CASES = (
    ((1, 1, 3, 2), 1),
    ((1, 2, 2, 4), 7),
    ((2, 3, 3, 8), 11),
    ((1, 2, 4, 64), 128),
    ((1, 4, 2, 128), 512),
    ((1, 1, 24, 128), 128),
    ((2, 1, 24, 128), 512),
    ((1, 4, 24, 128), 2_048),
    ((1, 1, 6, 128), 128),
    ((2, 1, 6, 128), 512),
    ((1, 4, 6, 128), 8_192),
)


def _call(
    x: torch.Tensor,
    past_length: int,
    rope_theta: float = 10_000.0,
) -> torch.Tensor:
    assert cuda_primitives is not None
    return cuda_primitives.cuda_apply_rope(x, past_length, rope_theta)


def _deterministic_input(
    shape: tuple[int, int, int, int],
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=generator) * 0.75).to(torch.bfloat16)


def _assert_matches_reference(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    error = (actual.cpu().float() - expected.cpu().float()).abs()
    maximum = float(error.max().item())
    mean = float(error.mean().item())
    assert maximum <= ACCEPTED_MAX_ABSOLUTE_ERROR, (
        f"maximum absolute error {maximum}, mean absolute error {mean}"
    )
    return maximum, mean


def test_position_zero_is_storage_exact_identity() -> None:
    x_cpu = _deterministic_input((2, 1, 5, 8), seed=3_001)
    x_cpu[0, 0, 0, 0] = 0.0
    x_cpu[0, 0, 0, 1] = -0.0
    x = x_cpu.cuda()

    actual = _call(x, past_length=0)
    actual_bits = actual.cpu().view(torch.int16)
    input_bits = x_cpu.view(torch.int16)

    assert torch.equal(actual_bits, input_bits)
    assert actual.data_ptr() != x.data_ptr()
    assert actual.shape == x.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device == x.device
    assert actual.is_contiguous()


def test_one_pair_matches_independent_hand_calculation() -> None:
    x_cpu = torch.tensor([[[[3.0, 4.0]]]], dtype=torch.bfloat16)
    angle = 1.0
    expected = torch.tensor(
        [[[[
            3.0 * math.cos(angle) - 4.0 * math.sin(angle),
            3.0 * math.sin(angle) + 4.0 * math.cos(angle),
        ]]]],
        dtype=torch.bfloat16,
    )

    actual = _call(x_cpu.cuda(), past_length=1)

    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(
        apply_rope_reference(x_cpu, 1, 10_000.0),
        expected,
    )


def test_adjacent_pair_layout_is_not_split_half() -> None:
    x_cpu = torch.tensor([[[[1.0, 0.0, 0.0, 1.0]]]], dtype=torch.bfloat16)
    first_angle = 1.0
    second_angle = 1.0 / math.sqrt(10_000.0)
    expected = torch.tensor(
        [[[[
            math.cos(first_angle),
            math.sin(first_angle),
            -math.sin(second_angle),
            math.cos(second_angle),
        ]]]],
        dtype=torch.bfloat16,
    )

    actual = _call(x_cpu.cuda(), past_length=1)

    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(
        apply_rope_reference(x_cpu, 1, 10_000.0),
        expected,
    )


def test_past_offset_tokens_heads_and_batches_are_indexed_independently() -> None:
    row = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.bfloat16)
    x_cpu = row.reshape(1, 1, 1, 4).expand(2, 3, 4, 4).contiguous()
    x = x_cpu.cuda()
    expected = apply_rope_reference(x, 5, 10_000.0)

    actual = _call(x, past_length=5)

    _assert_matches_reference(actual, expected)
    for token in range(x.shape[1]):
        assert torch.equal(actual[0, token, 0], actual[1, token, 3])
    assert not torch.equal(actual[:, 0], actual[:, 1])
    assert not torch.equal(actual[:, 1], actual[:, 2])


@pytest.mark.parametrize(
    ("shape", "past_length"),
    ROPE_CASES,
    ids=lambda value: (
        "x".join(map(str, value)) if isinstance(value, tuple) else f"past-{value}"
    ),
)
def test_deterministic_shape_matrix_matches_frozen_reference(
    shape: tuple[int, int, int, int],
    past_length: int,
) -> None:
    x_cpu = _deterministic_input(shape, seed=4_003 + math.prod(shape) + past_length)
    x = x_cpu.cuda()
    expected = apply_rope_reference(x, past_length, 10_000.0)

    actual = _call(x, past_length)
    torch.cuda.synchronize()

    _assert_matches_reference(actual, expected)
    assert actual.shape == x.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device == x.device
    assert actual.is_contiguous()
    assert actual.data_ptr() != x.data_ptr()


def test_custom_rope_theta_matches_frozen_reference() -> None:
    x_cpu = _deterministic_input((2, 3, 4, 8), seed=5_009)
    x = x_cpu.cuda()
    expected = apply_rope_reference(x, 17, 1_000.0)

    actual = _call(x, 17, 1_000.0)

    _assert_matches_reference(actual, expected)


def test_non_default_current_stream_orders_input_kernel_and_consumer() -> None:
    target_cpu = _deterministic_input((2, 4, 6, 128), seed=6_013)
    target = target_cpu.cuda()
    x = torch.zeros_like(target)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        x.copy_(target)
        actual = _call(x, past_length=512)
        downstream = actual.float().sum()
    stream.synchronize()

    expected = apply_rope_reference(target, 512, 10_000.0)
    _assert_matches_reference(actual, expected)
    torch.testing.assert_close(
        downstream.cpu(),
        expected.float().sum().cpu(),
        rtol=0.0,
        atol=1.0e-4,
    )


@pytest.mark.parametrize(
    ("name", "head_count", "seed"),
    (("q", 24, 7_021), ("k", 6, 7_027)),
)
@pytest.mark.parametrize(
    ("batch_size", "token_count", "past_length"),
    ((1, 1, 0), (2, 1, 512), (1, 4, 2_048)),
)
def test_modular_rmsnorm_then_rope_matches_frozen_reference(
    name: str,
    head_count: int,
    seed: int,
    batch_size: int,
    token_count: int,
    past_length: int,
) -> None:
    del name
    shape = (batch_size, token_count, head_count, 128)
    projected_cpu = _deterministic_input(
        shape,
        seed=seed + batch_size * 101 + token_count * 103 + past_length,
    )
    norm_weight_cpu = torch.linspace(
        0.5,
        1.5,
        128,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    projected = projected_cpu.cuda()
    norm_weight = norm_weight_cpu.cuda()

    expected_normalized = rms_norm_reference(projected, norm_weight, 1.0e-6)
    expected = apply_rope_reference(
        expected_normalized,
        past_length,
        10_000.0,
    )
    assert cuda_primitives is not None
    actual_normalized = cuda_primitives.cuda_rms_norm(
        projected,
        norm_weight,
        1.0e-6,
    )
    actual = cuda_primitives.cuda_apply_rope(
        actual_normalized,
        past_length,
        10_000.0,
    )

    assert torch.equal(actual_normalized, expected_normalized)
    _assert_matches_reference(actual, expected)


def test_cpu_input_is_rejected() -> None:
    x = torch.ones((1, 1, 1, 2), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="x must be a CUDA tensor"):
        _call(x, 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_wrong_input_dtype_is_rejected(dtype: torch.dtype) -> None:
    x = torch.ones((1, 1, 1, 2), dtype=dtype, device="cuda")
    with pytest.raises(RuntimeError, match="x must have dtype torch.bfloat16"):
        _call(x, 0)


def test_noncontiguous_input_is_rejected() -> None:
    x = torch.ones((1, 2, 8, 3), dtype=torch.bfloat16, device="cuda").transpose(
        2,
        3,
    )
    assert tuple(x.shape) == (1, 2, 3, 8)
    assert not x.is_contiguous()
    with pytest.raises(RuntimeError, match="x must be contiguous"):
        _call(x, 0)


@pytest.mark.parametrize("shape", [(1, 1, 2), (1, 1, 1, 2, 1)])
def test_wrong_rank_is_rejected(shape: tuple[int, ...]) -> None:
    x = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="x must have rank 4"):
        _call(x, 0)


@pytest.mark.parametrize(
    "shape",
    ((0, 1, 1, 2), (1, 0, 1, 2), (1, 1, 0, 2), (1, 1, 1, 0)),
)
def test_empty_dimensions_are_rejected(shape: tuple[int, int, int, int]) -> None:
    x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="dimensions must all be nonempty"):
        _call(x, 0)


def test_head_dimension_below_two_is_rejected() -> None:
    x = torch.ones((1, 1, 1, 1), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="head dimension must be at least 2"):
        _call(x, 0)


def test_odd_head_dimension_is_rejected() -> None:
    x = torch.ones((1, 1, 1, 3), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="head dimension must be even"):
        _call(x, 0)


def test_negative_past_length_is_rejected() -> None:
    x = torch.ones((1, 1, 1, 2), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="past_length must be nonnegative"):
        _call(x, -1)


def test_absolute_position_integer_overflow_is_rejected() -> None:
    x = torch.ones((1, 2, 1, 2), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="must fit in int64"):
        _call(x, torch.iinfo(torch.int64).max)


def test_past_length_outside_int64_is_rejected_by_operator_boundary() -> None:
    x = torch.ones((1, 1, 1, 2), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="Unable to cast"):
        _call(x, 2**63)


@pytest.mark.parametrize("rope_theta", [0.0, -1.0, math.inf, math.nan])
def test_invalid_rope_theta_is_rejected(rope_theta: float) -> None:
    x = torch.ones((1, 1, 1, 2), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="rope_theta must be finite and positive"):
        _call(x, 0, rope_theta)


@pytest.mark.parametrize(
    "rope_theta",
    [1.0e-50, float.fromhex("0x1.fffffffffffffp+1023")],
)
def test_rope_theta_not_representable_as_positive_finite_fp32_is_rejected(
    rope_theta: float,
) -> None:
    x = torch.ones((1, 1, 1, 2), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        RuntimeError,
        match="representable as a finite positive FP32 value",
    ):
        _call(x, 0, rope_theta)
