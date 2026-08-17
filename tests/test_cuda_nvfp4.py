"""Correctness and integration tests for Milestone 3C CUDA NVFP4 decode."""

from __future__ import annotations

import pytest
import torch

from reference import (
    NVFP4Tensor,
    dequantize_nvfp4_reference,
    quantize_nvfp4_reference,
)


CUDA_READY = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not CUDA_READY,
    reason="requires a CUDA device",
)

if CUDA_READY:
    import cuda_primitives
else:
    cuda_primitives = None  # type: ignore[assignment]


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

E2M1_PACKED_PATTERN = (
    0x10,
    0x32,
    0x54,
    0x76,
    0x98,
    0xBA,
    0xDC,
    0xFE,
)

REPRESENTATIVE_SHAPES = (
    (1, 16),
    (2, 32),
    (4, 64),
    (8, 128),
    (8, 3072),
    (24, 3072),
    (6, 3072),
)


def _manual_ue4m3_decode(code: int) -> float:
    """Independent scalar transcription of the frozen UE4M3 equation."""

    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def _call_unpack(packed_values: torch.Tensor) -> torch.Tensor:
    assert cuda_primitives is not None
    return cuda_primitives.cuda_unpack_e2m1_codes(packed_values)


def _call_dequant(quantized: NVFP4Tensor) -> torch.Tensor:
    assert cuda_primitives is not None
    return cuda_primitives.cuda_dequantize_nvfp4(quantized)


def _raw_dequant(
    packed_values: torch.Tensor,
    block_scales: torch.Tensor,
    global_decode_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.cuda_nvfp4_decoder_attention.cuda_dequantize_nvfp4(
        packed_values,
        block_scales,
        global_decode_scale,
    )


def _cuda_storage(
    packed_values: torch.Tensor,
    block_scales: torch.Tensor,
    global_decode_scale: torch.Tensor,
    logical_shape: tuple[int, int],
) -> NVFP4Tensor:
    return NVFP4Tensor(
        packed_values=packed_values.cuda(),
        block_scales=block_scales.cuda(),
        global_decode_scale=global_decode_scale.cuda(),
        logical_shape=logical_shape,
    )


def _move_quantized_to_cuda(quantized: NVFP4Tensor) -> NVFP4Tensor:
    return _cuda_storage(
        quantized.packed_values,
        quantized.block_scales,
        quantized.global_decode_scale,
        quantized.logical_shape,
    )


def _valid_raw_storage(
    *,
    rows: int = 1,
    columns: int = 16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((rows, columns // 2), dtype=torch.uint8, device=device),
        torch.zeros((rows, columns // 16), dtype=torch.uint8, device=device),
        torch.tensor(1.0, dtype=torch.float32, device=device),
    )


def _assert_exact_cuda_reference(
    quantized: NVFP4Tensor,
) -> tuple[float, float]:
    expected = dequantize_nvfp4_reference(quantized)
    actual = _call_dequant(quantized)
    error = (actual - expected).abs()
    maximum = float(error.max().item())
    mean = float(error.mean().item())
    assert maximum == 0.0, f"maximum={maximum}, mean={mean}"
    assert mean == 0.0
    assert torch.equal(actual, expected)
    return maximum, mean


def test_complete_e2m1_unpack_and_decode_tables_are_exact() -> None:
    packed = torch.tensor(
        [E2M1_PACKED_PATTERN],
        dtype=torch.uint8,
        device="cuda",
    )
    expected_codes = torch.arange(16, dtype=torch.uint8, device="cuda").reshape(
        1,
        16,
    )

    unpacked = _call_unpack(packed)

    assert torch.equal(unpacked, expected_codes)
    assert unpacked.shape == (1, 16)
    assert unpacked.dtype == torch.uint8
    assert unpacked.device == packed.device
    assert unpacked.is_contiguous()

    quantized = NVFP4Tensor(
        packed_values=packed,
        block_scales=torch.tensor([[0x38]], dtype=torch.uint8, device="cuda"),
        global_decode_scale=torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        logical_shape=(1, 16),
    )
    expected_values = torch.tensor(
        [E2M1_VALUES],
        dtype=torch.float32,
        device="cuda",
    )
    actual = _call_dequant(quantized)

    assert torch.equal(actual, expected_values)
    assert not torch.signbit(actual[0, 0])
    assert torch.signbit(actual[0, 8])
    assert actual.dtype == torch.float32
    assert actual.shape == (1, 16)
    assert actual.device == packed.device
    assert actual.is_contiguous()


def test_every_finite_ue4m3_byte_decodes_exactly_in_cuda() -> None:
    block_count = 0x7F
    packed = torch.full(
        (block_count, 8),
        0x22,
        dtype=torch.uint8,
        device="cuda",
    )
    scales = torch.arange(
        block_count,
        dtype=torch.uint8,
        device="cuda",
    ).reshape(block_count, 1)
    quantized = NVFP4Tensor(
        packed_values=packed,
        block_scales=scales,
        global_decode_scale=torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        logical_shape=(block_count, 16),
    )
    expected = torch.tensor(
        [_manual_ue4m3_decode(code) for code in range(block_count)],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(1).expand(block_count, 16)

    actual = _call_dequant(quantized)

    assert torch.equal(actual, expected)
    key_codes = (0x00, 0x01, 0x07, 0x08, 0x7E)
    key_values = (0.0, 2.0**-9, 7.0 * 2.0**-9, 2.0**-6, 448.0)
    for code, value in zip(key_codes, key_values, strict=True):
        assert actual[code, 0].item() == value


def test_hand_written_portable_golden_vector_is_exact() -> None:
    packed = torch.tensor([E2M1_PACKED_PATTERN], dtype=torch.uint8)
    scales = torch.tensor([[0x7E]], dtype=torch.uint8)
    gamma = torch.tensor(6.0, dtype=torch.float32) * torch.tensor(
        1.0 / 2688.0,
        dtype=torch.float32,
    )
    quantized = _cuda_storage(packed, scales, gamma, (1, 16))
    expected = torch.tensor(
        [
            [
                0.0,
                0.5,
                1.0,
                1.5000001192092896,
                2.0,
                3.000000238418579,
                4.0,
                6.000000476837158,
                -0.0,
                -0.5,
                -1.0,
                -1.5000001192092896,
                -2.0,
                -3.000000238418579,
                -4.0,
                -6.000000476837158,
            ]
        ],
        dtype=torch.float32,
        device="cuda",
    )

    actual = _call_dequant(quantized)

    assert torch.equal(actual, expected)
    assert torch.signbit(actual[0, 8])


def test_row_local_blocks_use_their_own_scale_and_never_cross_rows() -> None:
    quantized = NVFP4Tensor(
        packed_values=torch.full(
            (2, 16),
            0x22,
            dtype=torch.uint8,
            device="cuda",
        ),
        block_scales=torch.tensor(
            [[0x38, 0x40], [0x48, 0x50]],
            dtype=torch.uint8,
            device="cuda",
        ),
        global_decode_scale=torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        logical_shape=(2, 32),
    )
    expected = torch.tensor(
        [
            [1.0] * 16 + [2.0] * 16,
            [4.0] * 16 + [8.0] * 16,
        ],
        dtype=torch.float32,
        device="cuda",
    )

    assert torch.equal(_call_dequant(quantized), expected)


@pytest.mark.parametrize(
    "shape",
    REPRESENTATIVE_SHAPES,
    ids=lambda shape: "x".join(map(str, shape)),
)
def test_m2b_quantize_to_cuda_dequant_matches_same_device_oracle_exactly(
    shape: tuple[int, int],
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(
        12_001 + shape[0] * 101 + shape[1]
    )
    source = (torch.randn(shape, generator=generator) * 0.75).cuda()
    quantized = quantize_nvfp4_reference(source)

    maximum, mean = _assert_exact_cuda_reference(quantized)

    assert maximum == 0.0
    assert mean == 0.0


def test_all_zero_storage_matches_m2b_exactly() -> None:
    source = torch.zeros((2, 32), dtype=torch.float32)
    source[:, 1::2] = -0.0
    quantized = _move_quantized_to_cuda(quantize_nvfp4_reference(source))

    _assert_exact_cuda_reference(quantized)
    actual = _call_dequant(quantized)
    assert not torch.signbit(actual).any()


def test_zero_block_inside_nonzero_tensor_matches_m2b_exactly() -> None:
    source = torch.zeros((1, 32), dtype=torch.float32)
    source[0, 16:] = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    quantized = _move_quantized_to_cuda(quantize_nvfp4_reference(source))

    _assert_exact_cuda_reference(quantized)
    assert quantized.block_scales[0, 0].item() == 0x00
    assert torch.equal(_call_dequant(quantized)[0, :16], torch.zeros(16, device="cuda"))


def test_local_ue4m3_scale_underflow_matches_m2b_exactly() -> None:
    source = torch.cat(
        (torch.full((16,), 2.0**-20), torch.ones(16))
    ).reshape(1, 32)
    quantized = _move_quantized_to_cuda(quantize_nvfp4_reference(source))

    _assert_exact_cuda_reference(quantized)
    assert quantized.block_scales[0, 0].item() == 0x00
    assert torch.equal(_call_dequant(quantized)[0, :16], torch.zeros(16, device="cuda"))


def test_global_decode_scale_underflow_matches_m2b_exactly() -> None:
    minimum_subnormal = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    source = torch.full((1, 16), minimum_subnormal, dtype=torch.float32)
    quantized = _move_quantized_to_cuda(quantize_nvfp4_reference(source))

    assert quantized.global_decode_scale.item() == 0.0
    _assert_exact_cuda_reference(quantized)
    assert torch.equal(_call_dequant(quantized), torch.zeros((1, 16), device="cuda"))


def test_synthetic_zero_scale_with_nonzero_payload_follows_raw_equation() -> None:
    quantized = NVFP4Tensor(
        packed_values=torch.tensor(
            [E2M1_PACKED_PATTERN],
            dtype=torch.uint8,
            device="cuda",
        ),
        block_scales=torch.tensor([[0x00]], dtype=torch.uint8, device="cuda"),
        global_decode_scale=torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        logical_shape=(1, 16),
    )

    expected = dequantize_nvfp4_reference(quantized)
    actual = _call_dequant(quantized)

    assert torch.equal(actual, expected)
    assert torch.equal(actual, torch.zeros_like(actual))
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    assert not torch.signbit(actual[0, :8]).any()
    assert torch.signbit(actual[0, 8:]).all()


def test_non_default_current_stream_orders_storage_kernel_and_consumer() -> None:
    source = torch.linspace(-3.0, 3.0, 128, dtype=torch.float32).reshape(2, 64)
    target = _move_quantized_to_cuda(quantize_nvfp4_reference(source))
    expected = dequantize_nvfp4_reference(target)

    packed = torch.zeros_like(target.packed_values)
    scales = torch.zeros_like(target.block_scales)
    gamma = torch.zeros_like(target.global_decode_scale)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        packed.copy_(target.packed_values)
        scales.copy_(target.block_scales)
        gamma.copy_(target.global_decode_scale)
        actual = _raw_dequant(packed, scales, gamma)
        downstream = actual + 1.0
    stream.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(downstream, expected + 1.0)


def test_python_dequant_wrapper_requires_nvfp4_tensor() -> None:
    packed, _, _ = _valid_raw_storage()
    assert cuda_primitives is not None
    with pytest.raises(TypeError, match="NVFP4Tensor"):
        cuda_primitives.cuda_dequantize_nvfp4(packed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("cpu", "CUDA tensor"),
        ("dtype", "torch.uint8"),
        ("rank", "rank 2"),
        ("empty", "nonempty"),
        ("noncontiguous", "contiguous"),
    ),
    ids=("cpu", "dtype", "rank", "empty", "noncontiguous"),
)
def test_unpack_rejects_invalid_structure(
    case: str,
    message: str,
) -> None:
    if case == "cpu":
        packed = torch.zeros((1, 8), dtype=torch.uint8)
    elif case == "dtype":
        packed = torch.zeros((1, 8), dtype=torch.float32, device="cuda")
    elif case == "rank":
        packed = torch.zeros(8, dtype=torch.uint8, device="cuda")
    elif case == "empty":
        packed = torch.zeros((1, 0), dtype=torch.uint8, device="cuda")
    elif case == "noncontiguous":
        packed = torch.zeros(
            (8, 2),
            dtype=torch.uint8,
            device="cuda",
        ).transpose(0, 1)
    else:
        raise AssertionError(f"unexpected case {case}")
    with pytest.raises(RuntimeError, match=message):
        _call_unpack(packed)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("cpu-packed", "packed_values must be a CUDA tensor"),
        ("packed-dtype", "packed_values must have dtype torch.uint8"),
        ("packed-rank", "packed_values must have rank 2"),
        ("packed-empty", "packed_values must have nonempty"),
        ("packed-noncontiguous", "packed_values must be contiguous"),
        ("k-below-16", "logical K must be at least 16"),
        ("k-not-divisible-by-16", "logical K must be divisible by 16"),
    ),
    ids=(
        "cpu-packed",
        "packed-dtype",
        "packed-rank",
        "packed-empty",
        "packed-noncontiguous",
        "k-below-16",
        "k-not-divisible-by-16",
    ),
)
def test_dequant_rejects_invalid_packed_structure(
    case: str,
    message: str,
) -> None:
    if case == "cpu-packed":
        packed = torch.zeros((1, 8), dtype=torch.uint8)
        scales = torch.zeros((1, 1), dtype=torch.uint8, device="cuda")
    elif case == "packed-dtype":
        packed = torch.zeros((1, 8), dtype=torch.float32, device="cuda")
        scales = torch.zeros((1, 1), dtype=torch.uint8, device="cuda")
    elif case == "packed-rank":
        packed = torch.zeros(8, dtype=torch.uint8, device="cuda")
        scales = torch.zeros((1, 1), dtype=torch.uint8, device="cuda")
    elif case == "packed-empty":
        packed = torch.zeros((1, 0), dtype=torch.uint8, device="cuda")
        scales = torch.zeros((1, 0), dtype=torch.uint8, device="cuda")
    elif case == "packed-noncontiguous":
        packed = torch.zeros(
            (8, 2),
            dtype=torch.uint8,
            device="cuda",
        ).transpose(0, 1)
        scales = torch.zeros((2, 2), dtype=torch.uint8, device="cuda")
    elif case == "k-below-16":
        packed = torch.zeros((1, 4), dtype=torch.uint8, device="cuda")
        scales = torch.zeros((1, 0), dtype=torch.uint8, device="cuda")
    elif case == "k-not-divisible-by-16":
        packed = torch.zeros((1, 9), dtype=torch.uint8, device="cuda")
        scales = torch.zeros((1, 1), dtype=torch.uint8, device="cuda")
    else:
        raise AssertionError(f"unexpected case {case}")
    gamma = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match=message):
        _raw_dequant(packed, scales, gamma)


def test_dequant_rejects_cpu_block_scales() -> None:
    packed, _, gamma = _valid_raw_storage()
    scales = torch.zeros((1, 1), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="block_scales must be a CUDA tensor"):
        _raw_dequant(packed, scales, gamma)


def test_dequant_rejects_wrong_block_scale_dtype() -> None:
    packed, _, gamma = _valid_raw_storage()
    scales = torch.zeros((1, 1), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="block_scales must have dtype torch.uint8"):
        _raw_dequant(packed, scales, gamma)


def test_dequant_rejects_wrong_block_scale_rank() -> None:
    packed, _, gamma = _valid_raw_storage()
    scales = torch.zeros(1, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="block_scales must have rank 2"):
        _raw_dequant(packed, scales, gamma)


def test_dequant_rejects_noncontiguous_block_scales() -> None:
    packed, _, gamma = _valid_raw_storage(rows=2, columns=32)
    scales = torch.zeros((2, 2), dtype=torch.uint8, device="cuda").transpose(0, 1)
    assert not scales.is_contiguous()
    with pytest.raises(RuntimeError, match="block_scales must be contiguous"):
        _raw_dequant(packed, scales, gamma)


def test_dequant_rejects_mismatched_block_scale_rows() -> None:
    packed, _, gamma = _valid_raw_storage(rows=2)
    scales = torch.zeros((1, 1), dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="row count must match"):
        _raw_dequant(packed, scales, gamma)


def test_dequant_rejects_mismatched_block_scale_shape() -> None:
    packed, _, gamma = _valid_raw_storage(columns=32)
    scales = torch.zeros((1, 1), dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match=r"shape \[N, K/16\]"):
        _raw_dequant(packed, scales, gamma)


def test_dequant_rejects_cpu_global_decode_scale() -> None:
    packed, scales, _ = _valid_raw_storage()
    gamma = torch.tensor(1.0, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="global_decode_scale must be a CUDA tensor"):
        _raw_dequant(packed, scales, gamma)


def test_dequant_rejects_wrong_global_decode_scale_dtype() -> None:
    packed, scales, _ = _valid_raw_storage()
    gamma = torch.tensor(1.0, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError, match="global_decode_scale must have dtype torch.float32"):
        _raw_dequant(packed, scales, gamma)


def test_dequant_rejects_nonscalar_global_decode_scale() -> None:
    packed, scales, _ = _valid_raw_storage()
    gamma = torch.ones(1, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="scalar tensor with shape"):
        _raw_dequant(packed, scales, gamma)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_dequant_rejects_block_scale_device_mismatch_when_testable() -> None:
    packed, _, gamma = _valid_raw_storage(device="cuda:0")
    scales = torch.zeros((1, 1), dtype=torch.uint8, device="cuda:1")
    with pytest.raises(RuntimeError, match="same CUDA device"):
        _raw_dequant(packed, scales, gamma)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_dequant_rejects_global_scale_device_mismatch_when_testable() -> None:
    packed, scales, _ = _valid_raw_storage(device="cuda:0")
    gamma = torch.tensor(1.0, dtype=torch.float32, device="cuda:1")
    with pytest.raises(RuntimeError, match="same CUDA device"):
        _raw_dequant(packed, scales, gamma)
