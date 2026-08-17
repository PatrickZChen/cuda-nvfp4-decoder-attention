"""Correctness and integration tests for the Milestone 3A CUDA RMSNorm."""

from __future__ import annotations

import math

import pytest
import torch

from reference import rms_norm_reference


CUDA_READY = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
pytestmark = pytest.mark.skipif(
    not CUDA_READY,
    reason="requires a CUDA device with PyTorch BF16 support",
)

if CUDA_READY:
    import cuda_primitives
else:
    cuda_primitives = None  # type: ignore[assignment]


# The deterministic RTX 4080 / SM89 matrix was observed to be BF16 bit-exact.
# A nonzero result is therefore investigated instead of hidden by a tolerance.
ACCEPTED_MAX_ABSOLUTE_ERROR = 0.0


SHAPES = (
    (1, 1, 4),
    (2, 1, 8),
    (2, 3, 64),
    (2, 1, 128),
    (1, 1, 3072),
    (2, 1, 3072),
    (1, 1, 24, 128),
    (2, 1, 24, 128),
    (1, 1, 6, 128),
)


def _call(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    assert cuda_primitives is not None
    return cuda_primitives.cuda_rms_norm(x, weight, eps)


def _deterministic_case(
    shape: tuple[int, ...],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(shape, generator=generator) * 0.75).to(torch.bfloat16)
    weight = torch.linspace(
        0.5,
        1.5,
        shape[-1],
        dtype=torch.float32,
    ).to(torch.bfloat16)
    return x, weight


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


def test_tiny_hand_checkable_case() -> None:
    x_cpu = torch.tensor(
        [[[3.0, 4.0, 0.0, -1.0]]],
        dtype=torch.bfloat16,
    )
    weight_cpu = torch.tensor(
        [2.0, 0.5, 1.0, -1.0],
        dtype=torch.bfloat16,
    )
    eps = 1.0e-6
    inverse_rms = 1.0 / math.sqrt((9.0 + 16.0 + 0.0 + 1.0) / 4.0 + eps)
    hand_expected = torch.tensor(
        [[[6.0 * inverse_rms, 2.0 * inverse_rms, 0.0, inverse_rms]]],
        dtype=torch.bfloat16,
    )
    reference = rms_norm_reference(x_cpu, weight_cpu, eps)

    assert torch.equal(reference, hand_expected)
    actual = _call(x_cpu.cuda(), weight_cpu.cuda(), eps)
    assert torch.equal(actual.cpu(), hand_expected)


def test_multiple_rows_are_normalized_independently() -> None:
    x_cpu = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [8.0, -4.0, 2.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    weight_cpu = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16)
    x = x_cpu.cuda()
    weight = weight_cpu.cuda()
    actual = _call(x, weight)

    _assert_matches_reference(
        actual,
        rms_norm_reference(x_cpu, weight_cpu, 1.0e-6),
    )
    for row in range(x.shape[0]):
        isolated = _call(x[row : row + 1], weight)
        assert torch.equal(actual[row : row + 1], isolated)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: "x".join(map(str, shape)))
def test_deterministic_shape_matrix_matches_frozen_reference(
    shape: tuple[int, ...],
) -> None:
    x_cpu, weight_cpu = _deterministic_case(shape, seed=1_003 + math.prod(shape))
    x = x_cpu.cuda()
    weight = weight_cpu.cuda()
    expected = rms_norm_reference(x, weight, 1.0e-6)
    actual = _call(x, weight)
    torch.cuda.synchronize()

    maximum, mean = _assert_matches_reference(actual, expected)
    assert maximum == 0.0
    assert mean == 0.0
    assert actual.shape == x.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device == x.device
    assert actual.is_contiguous()
    assert actual.data_ptr() != x.data_ptr()


def test_zeros_and_mixed_signs_match_reference() -> None:
    x_cpu = torch.tensor(
        [
            [[0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0]],
            [[-8.0, 0.5, -0.5, 0.0, 3.0, -3.0, 6.0, -6.0]],
        ],
        dtype=torch.bfloat16,
    )
    weight_cpu = torch.tensor(
        [1.0, -1.0, 0.5, 2.0, -0.75, 1.5, -2.0, 0.0],
        dtype=torch.bfloat16,
    )
    expected = rms_norm_reference(x_cpu, weight_cpu, 1.0e-6)
    actual = _call(x_cpu.cuda(), weight_cpu.cuda())

    _assert_matches_reference(actual, expected)


def test_non_default_current_stream_orders_input_kernel_and_consumer() -> None:
    target_cpu, weight_cpu = _deterministic_case((2, 1, 24, 128), seed=4_091)
    target = target_cpu.cuda()
    x = torch.zeros_like(target)
    weight = weight_cpu.cuda()
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        x.copy_(target)
        actual = _call(x, weight)
        downstream = actual.float().sum()
    stream.synchronize()

    expected = rms_norm_reference(target_cpu, weight_cpu, 1.0e-6)
    _assert_matches_reference(actual, expected)
    torch.testing.assert_close(
        downstream.cpu(),
        expected.float().sum(),
        rtol=0.0,
        atol=1.0e-5,
    )


def test_cpu_inputs_are_rejected() -> None:
    x = torch.ones((1, 1, 8), dtype=torch.bfloat16)
    weight = torch.ones(8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        _call(x, weight)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_wrong_input_dtype_is_rejected(dtype: torch.dtype) -> None:
    x = torch.ones((1, 1, 8), dtype=dtype, device="cuda")
    weight = torch.ones(8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="x must have dtype torch.bfloat16"):
        _call(x, weight)


def test_wrong_weight_dtype_is_rejected() -> None:
    x = torch.ones((1, 1, 8), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(8, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="weight must have dtype torch.bfloat16"):
        _call(x, weight)


def test_wrong_weight_rank_is_rejected() -> None:
    x = torch.ones((1, 1, 8), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((1, 8), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="weight must be rank 1"):
        _call(x, weight)


def test_mismatched_final_dimension_is_rejected() -> None:
    x = torch.ones((1, 1, 8), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(7, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="weight length must match"):
        _call(x, weight)


def test_noncontiguous_input_is_rejected() -> None:
    x = torch.ones(
        (2, 8, 3),
        dtype=torch.bfloat16,
        device="cuda",
    ).transpose(1, 2)
    weight = torch.ones(8, dtype=torch.bfloat16, device="cuda")
    assert not x.is_contiguous()
    with pytest.raises(RuntimeError, match="x must be contiguous"):
        _call(x, weight)


def test_noncontiguous_weight_is_rejected() -> None:
    x = torch.ones((2, 3, 8), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(16, dtype=torch.bfloat16, device="cuda")[::2]
    assert not weight.is_contiguous()
    with pytest.raises(RuntimeError, match="weight must be contiguous"):
        _call(x, weight)


@pytest.mark.parametrize("eps", [0.0, -1.0e-6, math.inf, math.nan])
def test_invalid_epsilon_is_rejected(eps: float) -> None:
    x = torch.ones((1, 1, 8), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="eps must be finite and positive"):
        _call(x, weight, eps)


@pytest.mark.parametrize("eps", [1.0e-50, float.fromhex("0x1.fffffffffffffp+1023")])
def test_epsilon_not_representable_as_positive_finite_fp32_is_rejected(
    eps: float,
) -> None:
    x = torch.ones((1, 1, 8), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        RuntimeError,
        match="eps must be representable as a finite positive FP32 value",
    ):
        _call(x, weight, eps)


def test_empty_input_is_rejected() -> None:
    x = torch.empty((1, 0, 8), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="x must be nonempty"):
        _call(x, weight)


def test_empty_weight_is_rejected() -> None:
    x = torch.ones((1, 1, 8), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="weight must be nonempty"):
        _call(x, weight)


@pytest.mark.parametrize(
    "shape",
    [(), (1, 1, 1, 1, 8)],
    ids=["scalar", "rank-five"],
)
def test_unsupported_input_rank_is_rejected(shape: tuple[int, ...]) -> None:
    x = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(
        1 if not shape else shape[-1],
        dtype=torch.bfloat16,
        device="cuda",
    )
    with pytest.raises(RuntimeError, match=r"x must have rank in \[1, 4\]"):
        _call(x, weight)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_device_mismatch_is_rejected_when_testable() -> None:
    x = torch.ones((1, 1, 8), dtype=torch.bfloat16, device="cuda:0")
    weight = torch.ones(8, dtype=torch.bfloat16, device="cuda:1")
    with pytest.raises(RuntimeError, match="same CUDA device"):
        _call(x, weight)
