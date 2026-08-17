"""Milestone 2B tests for the portable NVFP4 numerical reference."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from reference import (
    NVFP4ErrorMetrics,
    NVFP4Tensor,
    analyze_nvfp4_error,
    compute_nvfp4_global_scales,
    decode_e2m1,
    decode_ue4m3,
    dequantize_nvfp4_reference,
    encode_e2m1,
    encode_ue4m3,
    pack_e2m1_codes,
    quantize_nvfp4_reference,
    unpack_e2m1_codes,
)


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


def _manual_ue4m3_decode(code: int) -> float:
    """Scalar oracle copied from the frozen format equation, not production."""

    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def _valid_storage(
    *,
    rows: int = 1,
    columns: int = 16,
) -> dict[str, object]:
    return {
        "packed_values": torch.zeros(
            (rows, columns // 2),
            dtype=torch.uint8,
        ),
        "block_scales": torch.zeros(
            (rows, columns // 16),
            dtype=torch.uint8,
        ),
        "global_decode_scale": torch.tensor(1.0, dtype=torch.float32),
        "logical_shape": (rows, columns),
    }


# A, B: exact code map and both zero signs.
def test_every_e2m1_nibble_decodes_to_the_frozen_value() -> None:
    codes = torch.arange(16, dtype=torch.uint8)
    expected = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    decoded = decode_e2m1(codes)

    assert decoded.dtype == torch.float32
    assert torch.equal(decoded, expected)
    assert not torch.signbit(decoded[0])
    assert torch.signbit(decoded[8])


# C: standalone zero signs and negative values rounded to zero are explicit.
def test_e2m1_encoding_preserves_signed_zero() -> None:
    values = torch.tensor([0.0, -0.0, 0.1, -0.1], dtype=torch.float32)
    expected_codes = torch.tensor([0x0, 0x8, 0x0, 0x8], dtype=torch.uint8)
    codes = encode_e2m1(values)

    assert torch.equal(codes, expected_codes)
    decoded = decode_e2m1(codes)
    assert torch.equal(
        torch.signbit(decoded),
        torch.tensor([False, True, False, True]),
    )


# D: all positive and negative nearest-even midpoint decisions.
def test_every_e2m1_midpoint_uses_ties_to_even() -> None:
    midpoints = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    positive_codes = [0x0, 0x2, 0x2, 0x4, 0x4, 0x6, 0x6]
    negative_codes = [0x8, 0xA, 0xA, 0xC, 0xC, 0xE, 0xE]
    values = torch.tensor(midpoints + [-value for value in midpoints])
    expected = torch.tensor(positive_codes + negative_codes, dtype=torch.uint8)

    assert torch.equal(encode_e2m1(values), expected)


# E: exact maxima and finite same-sign saturation.
def test_e2m1_exact_maxima_and_finite_saturation() -> None:
    values = torch.tensor(
        [
            -torch.finfo(torch.float32).max,
            -6.01,
            -6.0,
            6.0,
            6.01,
            torch.finfo(torch.float32).max,
        ],
        dtype=torch.float32,
    )
    expected = torch.tensor([0xF, 0xF, 0xF, 0x7, 0x7, 0x7], dtype=torch.uint8)
    assert torch.equal(encode_e2m1(values), expected)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_e2m1_standalone_encoder_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError):
        encode_e2m1(torch.tensor([value], dtype=torch.float32))


# F: every canonical UE4M3 byte and its key format boundaries.
def test_every_canonical_ue4m3_byte_decodes_exactly() -> None:
    codes = torch.arange(0x7F, dtype=torch.uint8)
    expected = torch.tensor(
        [_manual_ue4m3_decode(code) for code in range(0x7F)],
        dtype=torch.float32,
    )
    decoded = decode_ue4m3(codes)

    assert torch.equal(decoded, expected)
    key_codes = torch.tensor([0x00, 0x01, 0x07, 0x08, 0x7E], dtype=torch.uint8)
    key_values = torch.tensor(
        [0.0, 2.0**-9, 7.0 * 2.0**-9, 2.0**-6, 448.0],
        dtype=torch.float32,
    )
    assert torch.equal(decode_ue4m3(key_codes), key_values)


@pytest.mark.parametrize("invalid_byte", [0x7F, 0x80, 0xFE, 0xFF])
def test_ue4m3_rejects_nan_and_noncanonical_msb_bytes(invalid_byte: int) -> None:
    with pytest.raises(ValueError):
        decode_ue4m3(torch.tensor([invalid_byte], dtype=torch.uint8))


# G: exact finite-code round trip and midpoint RNE across the whole code range.
def test_every_finite_ue4m3_value_round_trips_to_its_byte() -> None:
    codes = torch.arange(0x7F, dtype=torch.uint8)
    assert torch.equal(encode_ue4m3(decode_ue4m3(codes)), codes)


def test_every_ue4m3_midpoint_uses_ties_to_even() -> None:
    values = [_manual_ue4m3_decode(code) for code in range(0x7F)]
    midpoints = [
        (values[lower] + values[lower + 1]) / 2.0
        for lower in range(0x7E)
    ]
    expected_codes = [
        lower if lower % 2 == 0 else lower + 1
        for lower in range(0x7E)
    ]

    actual = encode_ue4m3(torch.tensor(midpoints, dtype=torch.float32))
    expected = torch.tensor(expected_codes, dtype=torch.uint8)
    assert torch.equal(actual, expected)

    zero_subnormal_midpoint = torch.tensor(2.0**-10, dtype=torch.float32)
    assert encode_ue4m3(zero_subnormal_midpoint).item() == 0x00


def test_ue4m3_minima_maximum_and_finite_saturation() -> None:
    candidates = torch.tensor(
        [
            0.0,
            2.0**-10,
            2.0**-9,
            2.0**-6,
            448.0,
            449.0,
            torch.finfo(torch.float32).max,
        ],
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [0x00, 0x00, 0x01, 0x08, 0x7E, 0x7E, 0x7E],
        dtype=torch.uint8,
    )
    assert torch.equal(encode_ue4m3(candidates), expected)


@pytest.mark.parametrize("candidate", [-1.0, math.nan, math.inf, -math.inf])
def test_ue4m3_encoder_rejects_invalid_candidates(candidate: float) -> None:
    with pytest.raises(ValueError):
        encode_ue4m3(torch.tensor([candidate], dtype=torch.float32))


# H: the A == 0 global branch and canonical all-zero representation.
def test_all_zero_tensor_has_canonical_storage_and_exact_reconstruction() -> None:
    source = torch.zeros((2, 32), dtype=torch.float32)
    source[:, 1::2] = -0.0
    alpha, gamma = compute_nvfp4_global_scales(source)
    quantized = quantize_nvfp4_reference(source)
    reconstructed = dequantize_nvfp4_reference(quantized)

    assert alpha.item() == 1.0
    assert gamma.item() == 0.0
    assert quantized.global_decode_scale.item() == 0.0
    assert torch.count_nonzero(quantized.block_scales).item() == 0
    assert torch.count_nonzero(quantized.packed_values).item() == 0
    assert torch.count_nonzero(unpack_e2m1_codes(quantized.packed_values)).item() == 0
    assert torch.equal(reconstructed, torch.zeros_like(source))
    assert not torch.signbit(reconstructed).any()

    metrics = analyze_nvfp4_error(source, quantized)
    assert metrics.cosine_similarity == 1.0
    assert metrics.zero_fraction == 1.0
    assert metrics.saturation_fraction == 0.0


# I: an interior zero block cannot divide and always gets +0 payloads.
def test_zero_block_inside_nonzero_tensor_is_canonical() -> None:
    source = torch.zeros((1, 32), dtype=torch.float32)
    source[0, 16:] = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    quantized = quantize_nvfp4_reference(source)
    codes = unpack_e2m1_codes(quantized.packed_values)
    reconstructed = dequantize_nvfp4_reference(quantized)

    assert quantized.block_scales[0, 0].item() == 0x00
    assert torch.equal(codes[0, :16], torch.zeros(16, dtype=torch.uint8))
    assert torch.equal(reconstructed[0, :16], torch.zeros(16))
    assert quantized.block_scales[0, 1].item() != 0x00


# J: all expected values below are independently specified from the contract.
def test_hand_computable_block_matches_scale_codes_bytes_and_reconstruction() -> None:
    source = torch.tensor([E2M1_VALUES], dtype=torch.float32)
    expected_codes = torch.arange(16, dtype=torch.uint8).reshape(1, 16)
    expected_packed = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]],
        dtype=torch.uint8,
    )
    reciprocal_range = torch.tensor(1.0 / 2688.0, dtype=torch.float32)
    expected_gamma = torch.tensor(6.0, dtype=torch.float32) * reciprocal_range
    expected_reconstruction = torch.tensor(
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
    )

    alpha, gamma = compute_nvfp4_global_scales(source)
    quantized = quantize_nvfp4_reference(source)

    assert alpha.item() == 448.0
    assert torch.equal(gamma, expected_gamma)
    assert torch.equal(quantized.global_decode_scale, expected_gamma)
    assert torch.equal(
        quantized.block_scales,
        torch.tensor([[0x7E]], dtype=torch.uint8),
    )
    assert torch.equal(unpack_e2m1_codes(quantized.packed_values), expected_codes)
    assert torch.equal(quantized.packed_values, expected_packed)
    assert torch.equal(
        dequantize_nvfp4_reference(quantized),
        expected_reconstruction,
    )


# K: blocks in one row retain independent stored scales.
def test_multiple_blocks_in_a_row_have_independent_scales() -> None:
    source = torch.cat(
        (torch.full((16,), 6.0), torch.full((16,), 3.0))
    ).reshape(1, 32)
    quantized = quantize_nvfp4_reference(source)

    assert torch.equal(
        quantized.block_scales,
        torch.tensor([[0x7E, 0x76]], dtype=torch.uint8),
    )
    assert torch.equal(
        unpack_e2m1_codes(quantized.packed_values),
        torch.full((1, 32), 0x7, dtype=torch.uint8),
    )


# L: the same values in different rows cannot share a block.
def test_blocks_stop_at_row_boundaries() -> None:
    source = torch.stack(
        (torch.full((16,), 6.0), torch.full((16,), 3.0)),
        dim=0,
    )
    quantized = quantize_nvfp4_reference(source)

    assert quantized.block_scales.shape == (2, 1)
    assert torch.equal(
        quantized.block_scales,
        torch.tensor([[0x7E], [0x76]], dtype=torch.uint8),
    )
    codes = unpack_e2m1_codes(quantized.packed_values)
    assert torch.equal(codes, torch.full((2, 16), 0x7, dtype=torch.uint8))


# M, N: even-low/odd-high order, multiple rows, and exact nibble round trip.
def test_pack_order_and_multrow_round_trip_are_exact() -> None:
    codes = torch.tensor(
        [
            list(range(16)),
            list(reversed(range(16))),
        ],
        dtype=torch.uint8,
    )
    packed = pack_e2m1_codes(codes)
    expected_first = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
        dtype=torch.uint8,
    )
    expected_second = torch.tensor(
        [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01],
        dtype=torch.uint8,
    )

    assert torch.equal(packed[0], expected_first)
    assert torch.equal(packed[1], expected_second)
    assert torch.equal(unpack_e2m1_codes(packed), codes)


# O: directional ordinary, zero, cap, and decode-underflow branches.
def test_global_encode_and_decode_scale_branches_are_exact() -> None:
    ordinary = torch.zeros((1, 16), dtype=torch.float32)
    ordinary[0, 0] = 21.0
    alpha, gamma = compute_nvfp4_global_scales(ordinary)
    assert alpha.item() == 128.0
    assert gamma.item() == 1.0 / 128.0
    assert (alpha * gamma).item() == 1.0

    zero_alpha, zero_gamma = compute_nvfp4_global_scales(torch.zeros_like(ordinary))
    assert zero_alpha.item() == 1.0
    assert zero_gamma.item() == 0.0

    minimum_subnormal = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    tiny = torch.full((1, 16), minimum_subnormal, dtype=torch.float32)
    capped_alpha, underflowed_gamma = compute_nvfp4_global_scales(tiny)
    assert capped_alpha.item() == torch.finfo(torch.float32).max
    assert underflowed_gamma.item() == 0.0


# P: local reconstruction reflects rounded stored beta, not its raw candidate.
def test_reconstruction_uses_decoded_stored_block_scale() -> None:
    local_max = torch.tensor(1.03125 * 6.0 / 448.0, dtype=torch.float32)
    source = torch.cat(
        (torch.full((16,), 6.0), torch.full((16,), local_max))
    ).reshape(1, 32)
    quantized = quantize_nvfp4_reference(source)
    reconstructed = dequantize_nvfp4_reference(quantized)

    assert quantized.block_scales[0, 1].item() == 0x38
    stored_beta = torch.tensor(1.0, dtype=torch.float32)
    raw_candidate = torch.tensor(1.03125, dtype=torch.float32)
    decoded_maximum = torch.tensor(6.0, dtype=torch.float32)
    expected = (decoded_maximum * stored_beta) * quantized.global_decode_scale
    raw_candidate_result = (
        decoded_maximum * raw_candidate
    ) * quantized.global_decode_scale

    assert torch.all(reconstructed[0, 16:] == expected)
    assert not torch.any(reconstructed[0, 16:] == raw_candidate_result)


def _statistical_source(distribution: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        {
            "normal": 1001,
            "uniform": 1002,
            "outlier-heavy": 1003,
            "tiny": 1004,
            "large": 1005,
        }[distribution]
    )
    if distribution == "normal":
        return torch.randn((4, 64), generator=generator)
    if distribution == "uniform":
        return torch.rand((4, 64), generator=generator) * 4.0 - 2.0
    if distribution == "outlier-heavy":
        source = torch.randn((4, 64), generator=generator) * 0.1
        source[0, 0] = 100.0
        source[1, 17] = -80.0
        source[3, 63] = 55.0
        return source
    if distribution == "tiny":
        return torch.randn((4, 64), generator=generator) * 1.0e-20
    if distribution == "large":
        return torch.randn((4, 64), generator=generator) * 1.0e30
    raise AssertionError(f"unexpected distribution {distribution}")


# Q, R, S, plus the required deterministic tiny/large statistical stresses.
@pytest.mark.parametrize(
    "distribution",
    ["normal", "uniform", "outlier-heavy", "tiny", "large"],
)
def test_seeded_distributions_report_finite_quality_metrics(
    distribution: str,
) -> None:
    source = _statistical_source(distribution)
    quantized = quantize_nvfp4_reference(source)
    metrics = analyze_nvfp4_error(source, quantized)

    assert isinstance(metrics, NVFP4ErrorMetrics)
    for value in (
        metrics.maximum_absolute_error,
        metrics.mean_absolute_error,
        metrics.rmse,
        metrics.cosine_similarity,
        metrics.zero_fraction,
        metrics.saturation_fraction,
        metrics.maximum_code_fraction,
        metrics.scale_underflow_block_fraction,
    ):
        assert math.isfinite(value)
    assert metrics.maximum_absolute_error >= 0.0
    assert metrics.mean_absolute_error >= 0.0
    assert metrics.rmse >= 0.0
    assert metrics.cosine_similarity > 0.9
    for fraction in (
        metrics.zero_fraction,
        metrics.saturation_fraction,
        metrics.maximum_code_fraction,
        metrics.scale_underflow_block_fraction,
    ):
        assert 0.0 <= fraction <= 1.0


# T: local UE4M3 and global FP32 underflow have no minimum clamp.
def test_local_scale_and_global_decode_underflow_are_explicit() -> None:
    local_underflow_source = torch.cat(
        (
            torch.full((16,), 2.0**-20),
            torch.ones(16),
        )
    ).reshape(1, 32)
    local_quantized = quantize_nvfp4_reference(local_underflow_source)
    local_codes = unpack_e2m1_codes(local_quantized.packed_values)
    local_metrics = analyze_nvfp4_error(
        local_underflow_source,
        local_quantized,
    )

    assert local_quantized.block_scales[0, 0].item() == 0x00
    assert torch.equal(local_codes[0, :16], torch.zeros(16, dtype=torch.uint8))
    assert torch.equal(
        dequantize_nvfp4_reference(local_quantized)[0, :16],
        torch.zeros(16),
    )
    assert local_metrics.scale_underflow_block_fraction == 0.5

    minimum_subnormal = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    global_underflow_source = torch.full(
        (1, 16), minimum_subnormal, dtype=torch.float32
    )
    global_quantized = quantize_nvfp4_reference(global_underflow_source)
    assert global_quantized.global_decode_scale.item() == 0.0
    assert torch.equal(
        dequantize_nvfp4_reference(global_quantized),
        torch.zeros_like(global_underflow_source),
    )


# U: exact |y| == 6 is not clipping; strict |y| > 6 is counted separately.
def test_strict_saturation_and_maximum_code_fractions_are_distinct() -> None:
    clipped_local_max = torch.tensor(1.03125 * 6.0 / 448.0, dtype=torch.float32)
    source = torch.cat(
        (
            torch.full((16,), 6.0),
            torch.full((16,), clipped_local_max),
        )
    ).reshape(1, 32)
    metrics = analyze_nvfp4_error(source, quantize_nvfp4_reference(source))

    assert metrics.maximum_code_fraction == 1.0
    assert metrics.saturation_fraction == 0.5


# V: FP64 reductions and zero-vector cosine conventions.
def test_error_metric_reductions_match_independent_fp64_formulas() -> None:
    source = torch.linspace(-2.0, 2.0, 32, dtype=torch.float32).reshape(2, 16)
    quantized = quantize_nvfp4_reference(source)
    reconstructed = dequantize_nvfp4_reference(quantized)
    metrics = analyze_nvfp4_error(source, quantized)

    source64 = source.double()
    reconstructed64 = reconstructed.double()
    difference = reconstructed64 - source64
    expected_maximum = difference.abs().max().item()
    expected_mean = difference.abs().mean().item()
    expected_rmse = difference.square().mean().sqrt().item()
    expected_cosine = (
        (source64 * reconstructed64).sum()
        / (source64.square().sum().sqrt() * reconstructed64.square().sum().sqrt())
    ).item()
    codes = unpack_e2m1_codes(quantized.packed_values)
    expected_zero_fraction = (
        torch.count_nonzero((codes == 0x0) | (codes == 0x8)).item()
        / source.numel()
    )

    assert metrics.maximum_absolute_error == expected_maximum
    assert metrics.mean_absolute_error == expected_mean
    assert metrics.rmse == expected_rmse
    assert metrics.cosine_similarity == expected_cosine
    assert metrics.zero_fraction == expected_zero_fraction


def test_cosine_similarity_zero_vector_conventions_are_exact() -> None:
    zero_source = torch.zeros((1, 16), dtype=torch.float32)
    zero_quantized = NVFP4Tensor(**_valid_storage())
    assert analyze_nvfp4_error(zero_source, zero_quantized).cosine_similarity == 1.0

    nonzero_source = torch.ones((1, 16), dtype=torch.float32)
    assert analyze_nvfp4_error(
        nonzero_source,
        zero_quantized,
    ).cosine_similarity == 0.0

    nonzero_quantized = NVFP4Tensor(
        packed_values=torch.full((1, 8), 0x22, dtype=torch.uint8),
        block_scales=torch.tensor([[0x38]], dtype=torch.uint8),
        global_decode_scale=torch.tensor(1.0, dtype=torch.float32),
        logical_shape=(1, 16),
    )
    assert analyze_nvfp4_error(
        zero_source,
        nonzero_quantized,
    ).cosine_similarity == 0.0


# W: deterministic storage on repeated same-device runs.
def test_quantization_is_byte_for_byte_deterministic() -> None:
    source = _statistical_source("normal")
    first = quantize_nvfp4_reference(source)
    second = quantize_nvfp4_reference(source)

    assert torch.equal(first.packed_values, second.packed_values)
    assert torch.equal(first.block_scales, second.block_scales)
    assert torch.equal(first.global_decode_scale, second.global_decode_scale)
    assert torch.equal(
        dequantize_nvfp4_reference(first),
        dequantize_nvfp4_reference(second),
    )


# X: source rank, dimensions, dtype, and contiguity validation.
@pytest.mark.parametrize(
    "source",
    [
        torch.zeros(16, dtype=torch.float32),
        torch.zeros((1, 1, 16), dtype=torch.float32),
        torch.zeros((0, 16), dtype=torch.float32),
        torch.zeros((1, 0), dtype=torch.float32),
        torch.zeros((1, 8), dtype=torch.float32),
        torch.zeros((1, 17), dtype=torch.float32),
        torch.zeros((1, 16), dtype=torch.float16),
        torch.zeros((1, 16), dtype=torch.float64),
        torch.zeros((1, 16), dtype=torch.int32),
        torch.zeros((16, 2), dtype=torch.float32).transpose(0, 1),
    ],
    ids=[
        "rank-one",
        "rank-three",
        "empty-n",
        "empty-k",
        "k-too-small",
        "k-not-divisible",
        "fp16",
        "fp64",
        "integer",
        "noncontiguous",
    ],
)
def test_invalid_quantization_sources_are_rejected(source: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        quantize_nvfp4_reference(source)


def test_malformed_nvfp4_storage_is_rejected() -> None:
    with pytest.raises(TypeError):
        NVFP4Tensor(**{**_valid_storage(), "packed_values": torch.zeros((1, 8))})
    with pytest.raises(TypeError):
        NVFP4Tensor(**{**_valid_storage(), "block_scales": torch.zeros((1, 1))})
    with pytest.raises(TypeError):
        NVFP4Tensor(
            **{
                **_valid_storage(),
                "global_decode_scale": torch.tensor(1.0, dtype=torch.float64),
            }
        )
    with pytest.raises(ValueError):
        NVFP4Tensor(
            **{
                **_valid_storage(),
                "packed_values": torch.zeros((1, 7), dtype=torch.uint8),
            }
        )
    with pytest.raises(ValueError):
        NVFP4Tensor(
            **{
                **_valid_storage(),
                "block_scales": torch.zeros((1, 2), dtype=torch.uint8),
            }
        )
    with pytest.raises(ValueError):
        NVFP4Tensor(
            **{
                **_valid_storage(),
                "global_decode_scale": torch.ones(1, dtype=torch.float32),
            }
        )

    noncontiguous_values = torch.zeros((16, 2), dtype=torch.uint8).transpose(0, 1)
    assert not noncontiguous_values.is_contiguous()
    with pytest.raises(ValueError):
        NVFP4Tensor(
            **{
                **_valid_storage(rows=2, columns=32),
                "packed_values": noncontiguous_values,
            }
        )

    noncontiguous_scales = torch.zeros((2, 2), dtype=torch.uint8).transpose(0, 1)
    assert not noncontiguous_scales.is_contiguous()
    with pytest.raises(ValueError):
        NVFP4Tensor(
            **{
                **_valid_storage(rows=2, columns=32),
                "block_scales": noncontiguous_scales,
            }
        )

    for invalid_byte in (0x7F, 0x80, 0xFF):
        with pytest.raises(ValueError):
            NVFP4Tensor(
                **{
                    **_valid_storage(),
                    "block_scales": torch.tensor(
                        [[invalid_byte]],
                        dtype=torch.uint8,
                    ),
                }
            )

    for invalid_scale in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            NVFP4Tensor(
                **{
                    **_valid_storage(),
                    "global_decode_scale": torch.tensor(
                        invalid_scale,
                        dtype=torch.float32,
                    ),
                }
            )

    for logical_shape in (
        [1, 16],
        (True, 16),
        (0, 16),
        (1, 8),
        (1, 17),
        (2, 16),
    ):
        with pytest.raises((TypeError, ValueError)):
            NVFP4Tensor(**{**_valid_storage(), "logical_shape": logical_shape})


def test_nvfp4_storage_device_mismatch_is_rejected() -> None:
    meta_scales = torch.empty((1, 1), dtype=torch.uint8, device="meta")
    with pytest.raises(ValueError):
        NVFP4Tensor(**{**_valid_storage(), "block_scales": meta_scales})


def test_mutated_scale_storage_is_revalidated_before_dequantization() -> None:
    quantized = NVFP4Tensor(**_valid_storage())
    quantized.block_scales.fill_(0x7F)
    with pytest.raises(ValueError):
        dequantize_nvfp4_reference(quantized)


def test_nvfp4_data_objects_are_frozen() -> None:
    quantized = NVFP4Tensor(**_valid_storage())
    with pytest.raises((FrozenInstanceError, AttributeError)):
        quantized.logical_shape = (1, 32)  # type: ignore[misc]


# Y: nonfinite matrix inputs are rejected before the amax reduction.
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_quantization_sources_are_rejected(value: float) -> None:
    source = torch.zeros((1, 16), dtype=torch.float32)
    source[0, 5] = value
    with pytest.raises(ValueError):
        quantize_nvfp4_reference(source)


def test_bfloat16_source_is_supported_and_promoted_portably() -> None:
    source = _statistical_source("uniform").to(torch.bfloat16)
    quantized = quantize_nvfp4_reference(source)
    reconstructed = dequantize_nvfp4_reference(quantized)

    assert reconstructed.dtype == torch.float32
    assert reconstructed.shape == source.shape
    assert reconstructed.device == source.device
    assert torch.isfinite(reconstructed).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA device",
)
def test_cpu_cuda_quantization_has_exact_portable_storage_parity() -> None:
    cpu_source = _statistical_source("normal")
    cpu_quantized = quantize_nvfp4_reference(cpu_source)
    cuda_quantized = quantize_nvfp4_reference(cpu_source.to("cuda"))
    torch.cuda.synchronize()

    assert cuda_quantized.packed_values.device.type == "cuda"
    assert cuda_quantized.block_scales.device.type == "cuda"
    assert cuda_quantized.global_decode_scale.device.type == "cuda"
    assert torch.equal(
        cuda_quantized.packed_values.cpu(),
        cpu_quantized.packed_values,
    )
    assert torch.equal(
        cuda_quantized.block_scales.cpu(),
        cpu_quantized.block_scales,
    )
    assert torch.equal(
        cuda_quantized.global_decode_scale.cpu(),
        cpu_quantized.global_decode_scale,
    )
    assert torch.equal(
        dequantize_nvfp4_reference(cuda_quantized).cpu(),
        dequantize_nvfp4_reference(cpu_quantized),
    )
