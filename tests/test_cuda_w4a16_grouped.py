"""Correctness tests for the isolated M4C grouped-decode W4A16 candidate."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from reference import (
    NVFP4Tensor,
    quantize_nvfp4_reference,
    w4a16_linear_reference,
)


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
class ProjectionMetrics:
    maximum_absolute_error: float
    mean_absolute_error: float
    exact_bf16_fraction: float
    maximum_bf16_adjacency_distance: int


E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

REPRESENTATIVE_CASES = (
    ((16,), 1),
    ((3, 32), 3),
    ((1, 2, 128), 8),
    ((1, 1, 2, 512), 17),
)


def _call(x: torch.Tensor, weight: NVFP4Tensor) -> torch.Tensor:
    assert cuda_primitives is not None
    return cuda_primitives.cuda_w4a16_linear_grouped_decode(x, weight)


def _raw_call(
    x: torch.Tensor,
    packed_values: torch.Tensor,
    block_scales: torch.Tensor,
    global_decode_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.cuda_nvfp4_decoder_attention.cuda_w4a16_linear_grouped_decode(
        x,
        packed_values,
        block_scales,
        global_decode_scale,
    )


def _move_quantized(
    quantized: NVFP4Tensor,
    device: str | torch.device = "cuda",
) -> NVFP4Tensor:
    return NVFP4Tensor(
        packed_values=quantized.packed_values.to(device),
        block_scales=quantized.block_scales.to(device),
        global_decode_scale=quantized.global_decode_scale.to(device),
        logical_shape=quantized.logical_shape,
    )


def _storage(
    packed_values: list[list[int]],
    block_scales: list[list[int]],
    global_decode_scale: float,
    *,
    device: str | torch.device = "cpu",
) -> NVFP4Tensor:
    packed = torch.tensor(packed_values, dtype=torch.uint8, device=device)
    scales = torch.tensor(block_scales, dtype=torch.uint8, device=device)
    return NVFP4Tensor(
        packed_values=packed,
        block_scales=scales,
        global_decode_scale=torch.tensor(
            global_decode_scale,
            dtype=torch.float32,
            device=device,
        ),
        logical_shape=(packed.shape[0], packed.shape[1] * 2),
    )


def _bf16_ordered_keys(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    magnitude = bits & 0x7FFF
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> ProjectionMetrics:
    assert actual.dtype == torch.bfloat16
    assert expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()

    error = (actual.float() - expected.float()).abs()
    adjacency = (
        _bf16_ordered_keys(actual) - _bf16_ordered_keys(expected)
    ).abs()
    return ProjectionMetrics(
        maximum_absolute_error=float(error.max().item()),
        mean_absolute_error=float(error.mean().item()),
        exact_bf16_fraction=float((actual == expected).float().mean().item()),
        maximum_bf16_adjacency_distance=int(adjacency.max().item()),
    )


def _assert_at_most_one_adjacent_bf16(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> ProjectionMetrics:
    metrics = _metrics(actual, expected)
    assert metrics.maximum_bf16_adjacency_distance <= 1, metrics
    return metrics


def _deterministic_source(rows: int, columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((rows, columns), generator=generator) * 0.5


def _deterministic_activation(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=generator) * 0.75).to(torch.bfloat16)


def _manual_ue4m3_decode(code: int) -> float:
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def _valid_raw_storage(
    *,
    rows: int = 2,
    columns: int = 32,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.ones(columns, dtype=torch.bfloat16, device=device)
    packed = torch.full(
        (rows, columns // 2),
        0x22,
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.full(
        (rows, columns // 16),
        0x38,
        dtype=torch.uint8,
        device=device,
    )
    gamma = torch.tensor(1.0, dtype=torch.float32, device=device)
    return x, packed, scales, gamma


def test_grouped_hand_computable_k16_projection_is_exact() -> None:
    weight_cpu = _storage(
        [[0x42] * 8, [0xA2] * 8],
        [[0x38], [0x38]],
        1.0,
    )
    x_cpu = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.arange(1, 17, dtype=torch.float32).to(torch.bfloat16),
        )
    )
    expected = torch.tensor(
        [[24.0, 0.0], [208.0, -8.0]],
        dtype=torch.bfloat16,
    )

    assert torch.equal(w4a16_linear_reference(x_cpu, weight_cpu), expected)
    actual = _call(x_cpu.cuda(), _move_quantized(weight_cpu))

    assert torch.equal(actual.cpu(), expected)
    assert actual.dtype == torch.bfloat16
    assert actual.shape == (2, 2)
    assert actual.is_contiguous()


def test_grouped_weight_orientation_is_x_times_w_transpose() -> None:
    weight = _move_quantized(
        _storage(
            [
                [0x22] * 8,
                [0x44] * 8,
                [0xA2] * 8,
            ],
            [[0x38], [0x38], [0x38]],
            1.0,
        )
    )
    x = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.arange(1, 17, dtype=torch.float32).to(torch.bfloat16),
        )
    ).cuda()
    expected = torch.tensor(
        [[16.0, 32.0, 0.0], [136.0, 272.0, -8.0]],
        dtype=torch.bfloat16,
        device="cuda",
    )

    actual = _call(x, weight)

    assert actual.shape == (2, 3)
    assert torch.equal(actual, expected)


def test_grouped_low_and_high_nibbles_are_independently_consumed() -> None:
    weight = _move_quantized(_storage([[0x42] * 8], [[0x38]], 1.0))
    x = torch.zeros((2, 16), dtype=torch.bfloat16, device="cuda")
    x[0, 0::2] = 1.0
    x[1, 1::2] = 1.0

    actual = _call(x, weight)

    assert torch.equal(
        actual,
        torch.tensor([[8.0], [16.0]], dtype=torch.bfloat16, device="cuda"),
    )


def test_grouped_row_local_microscale_indexing_is_exact() -> None:
    weight = _move_quantized(
        _storage(
            [[0x22] * 16, [0x22] * 16],
            [[0x38, 0x40], [0x48, 0x50]],
            1.0,
        )
    )
    x = torch.zeros((2, 32), dtype=torch.bfloat16, device="cuda")
    x[0, :16] = 1.0
    x[1, 16:] = 1.0

    actual = _call(x, weight)

    assert torch.equal(
        actual,
        torch.tensor(
            [[16.0, 64.0], [32.0, 128.0]],
            dtype=torch.bfloat16,
            device="cuda",
        ),
    )


def test_grouped_every_e2m1_code_is_decoded() -> None:
    weight = _move_quantized(
        _storage(
            [[code | (code << 4)] * 8 for code in range(16)],
            [[0x38]] * 16,
            1.0,
        )
    )
    x = torch.zeros(16, dtype=torch.bfloat16, device="cuda")
    x[0] = 1.0

    actual = _call(x, weight)

    assert torch.equal(
        actual,
        torch.tensor(E2M1_VALUES, dtype=torch.bfloat16, device="cuda"),
    )


def test_grouped_every_finite_canonical_ue4m3_code_is_decoded() -> None:
    finite_code_count = 0x7F
    weight = NVFP4Tensor(
        packed_values=torch.full(
            (finite_code_count, 8),
            0x22,
            dtype=torch.uint8,
            device="cuda",
        ),
        block_scales=torch.arange(
            finite_code_count,
            dtype=torch.uint8,
            device="cuda",
        ).reshape(finite_code_count, 1),
        global_decode_scale=torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        logical_shape=(finite_code_count, 16),
    )
    x = torch.zeros(16, dtype=torch.bfloat16, device="cuda")
    x[0] = 1.0
    expected = torch.tensor(
        [_manual_ue4m3_decode(code) for code in range(finite_code_count)],
        dtype=torch.bfloat16,
        device="cuda",
    )

    actual = _call(x, weight)

    assert torch.equal(actual, expected)


def test_grouped_zero_activation_zero_weight_and_zero_scale_blocks() -> None:
    source = torch.linspace(-2.0, 2.0, 64).reshape(2, 32)
    weight = _move_quantized(quantize_nvfp4_reference(source))
    zero_x = torch.zeros((3, 32), dtype=torch.bfloat16, device="cuda")
    assert torch.equal(_call(zero_x, weight), w4a16_linear_reference(zero_x, weight))

    zero_weight = _move_quantized(
        quantize_nvfp4_reference(torch.zeros((2, 32), dtype=torch.float32))
    )
    x = _deterministic_activation((3, 32), 31_001).cuda()
    assert torch.equal(
        _call(x, zero_weight),
        w4a16_linear_reference(x, zero_weight),
    )

    zero_scale_weight = _move_quantized(
        _storage(
            [[0x42] * 16, [0xA2] * 16],
            [[0x00, 0x38], [0x38, 0x00]],
            1.0,
        )
    )
    assert torch.equal(
        _call(x, zero_scale_weight),
        w4a16_linear_reference(x, zero_scale_weight),
    )


def test_grouped_signed_mixed_e2m1_values_are_exact() -> None:
    weight = _move_quantized(
        _storage(
            [[0xA2, 0xD4, 0xF6, 0x9B, 0x2A, 0x4D, 0x6F, 0xB9]],
            [[0x38]],
            1.0,
        )
    )
    x = torch.arange(1, 17, dtype=torch.float32).to(torch.bfloat16).cuda()

    actual = _call(x, weight)
    expected = w4a16_linear_reference(x, weight)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("activation_shape", "output_features"),
    REPRESENTATIVE_CASES,
    ids=lambda value: (
        "x".join(map(str, value)) if isinstance(value, tuple) else f"n{value}"
    ),
)
def test_grouped_k_sizes_ranks_and_multiple_m_n_match_reference(
    activation_shape: tuple[int, ...],
    output_features: int,
) -> None:
    reduction_size = activation_shape[-1]
    seed = 32_003 + reduction_size * 13 + output_features
    source = _deterministic_source(output_features, reduction_size, seed)
    weight = _move_quantized(quantize_nvfp4_reference(source))
    x = _deterministic_activation(activation_shape, seed + 17).cuda()

    expected = w4a16_linear_reference(x, weight)
    actual = _call(x, weight)
    metrics = _assert_at_most_one_adjacent_bf16(actual, expected)

    print(
        f"grouped-representative shape={activation_shape} n={output_features} "
        f"max_abs={metrics.maximum_absolute_error:.9g} "
        f"mean_abs={metrics.mean_absolute_error:.9g} "
        f"exact_fraction={metrics.exact_bf16_fraction:.9g} "
        f"max_bf16_distance={metrics.maximum_bf16_adjacency_distance}"
    )
    assert actual.shape == (*activation_shape[:-1], output_features)
    assert actual.dtype == torch.bfloat16
    assert actual.device == x.device


def test_grouped_multiple_rows_with_k3072_match_reference() -> None:
    source = _deterministic_source(32, 3072, 33_001)
    weight = _move_quantized(quantize_nvfp4_reference(source))
    x = _deterministic_activation((1, 2, 3072), 33_007).cuda()

    expected = w4a16_linear_reference(x, weight)
    actual = _call(x, weight)
    metrics = _assert_at_most_one_adjacent_bf16(actual, expected)

    print(
        "grouped-long-reduction m=2 n=32 k=3072 "
        f"max_abs={metrics.maximum_absolute_error:.9g} "
        f"mean_abs={metrics.mean_absolute_error:.9g} "
        f"exact_fraction={metrics.exact_bf16_fraction:.9g} "
        f"max_bf16_distance={metrics.maximum_bf16_adjacency_distance}"
    )


@pytest.mark.parametrize(
    ("label", "output_features"),
    (("q", 3072), ("kv", 768)),
)
def test_grouped_canonical_decoder_projection_metrics(
    label: str,
    output_features: int,
) -> None:
    reduction_size = 3072
    seed = 34_001 + output_features
    source_cpu = _deterministic_source(output_features, reduction_size, seed)
    weight = quantize_nvfp4_reference(source_cpu.cuda())
    del source_cpu
    x = _deterministic_activation((1, 1, reduction_size), seed + 31).cuda()

    expected = w4a16_linear_reference(x, weight)
    actual = _call(x, weight)
    baseline = cuda_primitives.cuda_w4a16_linear(x, weight)
    metrics = _assert_at_most_one_adjacent_bf16(actual, expected)
    baseline_metrics = _metrics(actual, baseline)
    torch.cuda.synchronize()

    print(
        f"grouped-canonical-{label} shape={tuple(actual.shape)} "
        f"max_abs={metrics.maximum_absolute_error:.9g} "
        f"mean_abs={metrics.mean_absolute_error:.9g} "
        f"exact_fraction={metrics.exact_bf16_fraction:.9g} "
        f"max_bf16_distance={metrics.maximum_bf16_adjacency_distance} "
        "candidate_vs_baseline_max_bf16_distance="
        f"{baseline_metrics.maximum_bf16_adjacency_distance}"
    )
    assert actual.shape == (1, 1, output_features)


def test_grouped_non_default_stream_orders_inputs_kernel_and_consumer() -> None:
    source = _deterministic_source(8, 128, 35_003)
    target_weight = _move_quantized(quantize_nvfp4_reference(source))
    target_x = _deterministic_activation((2, 128), 35_009).cuda()
    expected = w4a16_linear_reference(target_x, target_weight)

    x = torch.zeros_like(target_x)
    prepared_weight = NVFP4Tensor(
        packed_values=torch.zeros_like(target_weight.packed_values),
        block_scales=torch.zeros_like(target_weight.block_scales),
        global_decode_scale=torch.zeros_like(target_weight.global_decode_scale),
        logical_shape=target_weight.logical_shape,
    )
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        x.copy_(target_x)
        prepared_weight.packed_values.copy_(target_weight.packed_values)
        prepared_weight.block_scales.copy_(target_weight.block_scales)
        prepared_weight.global_decode_scale.copy_(
            target_weight.global_decode_scale
        )
        actual = _call(x, prepared_weight)
        downstream = actual.float() + 1.0
    stream.synchronize()

    metrics = _assert_at_most_one_adjacent_bf16(actual, expected)
    print(
        f"grouped-current-stream max_abs={metrics.maximum_absolute_error:.9g} "
        f"mean_abs={metrics.mean_absolute_error:.9g} "
        f"exact_fraction={metrics.exact_bf16_fraction:.9g} "
        f"max_bf16_distance={metrics.maximum_bf16_adjacency_distance}"
    )
    assert torch.equal(downstream, actual.float() + 1.0)


def test_grouped_python_wrapper_requires_nvfp4_tensor() -> None:
    assert cuda_primitives is not None
    x, packed, _, _ = _valid_raw_storage()
    with pytest.raises(TypeError, match="NVFP4Tensor"):
        cuda_primitives.cuda_w4a16_linear_grouped_decode(  # type: ignore[arg-type]
            x,
            packed,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("cpu", "x must be a CUDA tensor"),
        ("dtype", "x must have dtype torch.bfloat16"),
        ("rank-zero", r"rank in \[1, 4\]"),
        ("rank-five", r"rank in \[1, 4\]"),
        ("empty", "x must be nonempty"),
        ("noncontiguous", "x must be contiguous"),
        ("k-mismatch", "x final dimension must match"),
    ),
)
def test_grouped_raw_operator_rejects_invalid_activation_structure(
    case: str,
    message: str,
) -> None:
    _, packed, scales, gamma = _valid_raw_storage()
    if case == "cpu":
        x = torch.ones(32, dtype=torch.bfloat16)
    elif case == "dtype":
        x = torch.ones(32, dtype=torch.float32, device="cuda")
    elif case == "rank-zero":
        x = torch.tensor(1.0, dtype=torch.bfloat16, device="cuda")
    elif case == "rank-five":
        x = torch.ones((1, 1, 1, 1, 32), dtype=torch.bfloat16, device="cuda")
    elif case == "empty":
        x = torch.empty((0, 32), dtype=torch.bfloat16, device="cuda")
    elif case == "noncontiguous":
        x = torch.ones((2, 32, 3), dtype=torch.bfloat16, device="cuda").transpose(
            1, 2
        )
        assert x.shape[-1] == 32 and not x.is_contiguous()
    elif case == "k-mismatch":
        x = torch.ones(16, dtype=torch.bfloat16, device="cuda")
    else:
        raise AssertionError(f"unexpected case {case}")

    with pytest.raises(RuntimeError, match=message):
        _raw_call(x, packed, scales, gamma)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("cpu", "packed_values must be a CUDA tensor"),
        ("dtype", "packed_values must have dtype torch.uint8"),
        ("rank", "packed_values must have rank 2"),
        ("noncontiguous", "packed_values must be contiguous"),
        ("empty-n", "nonempty N and packed-byte dimensions"),
        ("k-below-16", "logical K must be at least 16"),
        ("k-not-divisible", "logical K must be divisible by 16"),
    ),
)
def test_grouped_raw_operator_rejects_invalid_packed_weight_structure(
    case: str,
    message: str,
) -> None:
    x, _, scales, gamma = _valid_raw_storage()
    if case == "cpu":
        packed = torch.zeros((2, 16), dtype=torch.uint8)
    elif case == "dtype":
        packed = torch.zeros((2, 16), dtype=torch.float32, device="cuda")
    elif case == "rank":
        packed = torch.zeros(16, dtype=torch.uint8, device="cuda")
    elif case == "noncontiguous":
        packed = torch.zeros((16, 2), dtype=torch.uint8, device="cuda").transpose(
            0, 1
        )
        assert not packed.is_contiguous()
    elif case == "empty-n":
        packed = torch.zeros((0, 16), dtype=torch.uint8, device="cuda")
        scales = torch.zeros((0, 2), dtype=torch.uint8, device="cuda")
    elif case == "k-below-16":
        x = torch.ones(8, dtype=torch.bfloat16, device="cuda")
        packed = torch.zeros((2, 4), dtype=torch.uint8, device="cuda")
        scales = torch.zeros((2, 0), dtype=torch.uint8, device="cuda")
    elif case == "k-not-divisible":
        x = torch.ones(18, dtype=torch.bfloat16, device="cuda")
        packed = torch.zeros((2, 9), dtype=torch.uint8, device="cuda")
        scales = torch.zeros((2, 1), dtype=torch.uint8, device="cuda")
    else:
        raise AssertionError(f"unexpected case {case}")

    with pytest.raises(RuntimeError, match=message):
        _raw_call(x, packed, scales, gamma)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("cpu", "block_scales must be a CUDA tensor"),
        ("dtype", "block_scales must have dtype torch.uint8"),
        ("rank", "block_scales must have rank 2"),
        ("noncontiguous", "block_scales must be contiguous"),
        ("rows", "row count must match"),
        ("columns", r"shape \[N, K/16\]"),
    ),
)
def test_grouped_raw_operator_rejects_invalid_block_scale_structure(
    case: str,
    message: str,
) -> None:
    x, packed, _, gamma = _valid_raw_storage()
    if case == "cpu":
        scales = torch.zeros((2, 2), dtype=torch.uint8)
    elif case == "dtype":
        scales = torch.zeros((2, 2), dtype=torch.float32, device="cuda")
    elif case == "rank":
        scales = torch.zeros(4, dtype=torch.uint8, device="cuda")
    elif case == "noncontiguous":
        scales = torch.zeros((2, 2), dtype=torch.uint8, device="cuda").transpose(
            0, 1
        )
        assert not scales.is_contiguous()
    elif case == "rows":
        scales = torch.zeros((1, 2), dtype=torch.uint8, device="cuda")
    elif case == "columns":
        scales = torch.zeros((2, 1), dtype=torch.uint8, device="cuda")
    else:
        raise AssertionError(f"unexpected case {case}")

    with pytest.raises(RuntimeError, match=message):
        _raw_call(x, packed, scales, gamma)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("cpu", "global_decode_scale must be a CUDA tensor"),
        ("dtype", "global_decode_scale must have dtype torch.float32"),
        ("shape", "scalar tensor with shape"),
    ),
)
def test_grouped_raw_operator_rejects_invalid_global_decode_scale_structure(
    case: str,
    message: str,
) -> None:
    x, packed, scales, _ = _valid_raw_storage()
    if case == "cpu":
        gamma = torch.tensor(1.0, dtype=torch.float32)
    elif case == "dtype":
        gamma = torch.tensor(1.0, dtype=torch.float64, device="cuda")
    elif case == "shape":
        gamma = torch.ones(1, dtype=torch.float32, device="cuda")
    else:
        raise AssertionError(f"unexpected case {case}")

    with pytest.raises(RuntimeError, match=message):
        _raw_call(x, packed, scales, gamma)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
@pytest.mark.parametrize("field", ("packed", "scales", "gamma"))
def test_grouped_raw_operator_rejects_device_mismatch_when_testable(
    field: str,
) -> None:
    x, packed, scales, gamma = _valid_raw_storage(device="cuda:0")
    if field == "packed":
        packed = packed.to("cuda:1")
    elif field == "scales":
        scales = scales.to("cuda:1")
    elif field == "gamma":
        gamma = gamma.to("cuda:1")
    else:
        raise AssertionError(f"unexpected field {field}")

    with pytest.raises(RuntimeError, match="same CUDA device"):
        _raw_call(x, packed, scales, gamma)
