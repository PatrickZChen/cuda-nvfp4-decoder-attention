"""Run custom CUDA NVFP4 unpack/dequant kernels under validation tools."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cuda_primitives  # noqa: E402
from reference import (  # noqa: E402
    NVFP4Tensor,
    dequantize_nvfp4_reference,
    quantize_nvfp4_reference,
)


E2M1_PACKED_PATTERN = (0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE)
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
CASES = (
    (2, 32),
    (4, 64),
    (8, 128),
    (8, 3072),
    (24, 3072),
    (6, 3072),
)


def _manual_ue4m3_decode(code: int) -> float:
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    return float(error.max().item()), float(error.mean().item())


def _require_exact(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    maximum, mean = _errors(actual, expected)
    print(f"case={label} max_abs={maximum:.9g} mean_abs={mean:.9g}")
    if maximum != 0.0 or mean != 0.0 or not torch.equal(actual, expected):
        raise AssertionError(
            f"{label} diverged from its independent/reference result: "
            f"maximum={maximum}, mean={mean}"
        )
    return maximum, mean


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    print(f"device={torch.cuda.get_device_name()}")
    print(f"capability={torch.cuda.get_device_capability()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")

    packed = torch.tensor(
        [E2M1_PACKED_PATTERN],
        dtype=torch.uint8,
        device="cuda",
    )
    expected_codes = torch.arange(16, dtype=torch.uint8, device="cuda").reshape(
        1,
        16,
    )
    unpacked = cuda_primitives.cuda_unpack_e2m1_codes(packed)
    _require_exact("e2m1-unpack", unpacked, expected_codes)

    e2m1_quantized = NVFP4Tensor(
        packed_values=packed,
        block_scales=torch.tensor([[0x38]], dtype=torch.uint8, device="cuda"),
        global_decode_scale=torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        logical_shape=(1, 16),
    )
    expected_e2m1 = torch.tensor(
        [E2M1_VALUES],
        dtype=torch.float32,
        device="cuda",
    )
    decoded_e2m1 = cuda_primitives.cuda_dequantize_nvfp4(e2m1_quantized)
    _require_exact("e2m1-decode", decoded_e2m1, expected_e2m1)
    if not torch.signbit(decoded_e2m1[0, 8]).item():
        raise AssertionError("E2M1 code 0x8 did not decode to negative zero")

    finite_scale_count = 0x7F
    all_scales_quantized = NVFP4Tensor(
        packed_values=torch.full(
            (finite_scale_count, 8),
            0x22,
            dtype=torch.uint8,
            device="cuda",
        ),
        block_scales=torch.arange(
            finite_scale_count,
            dtype=torch.uint8,
            device="cuda",
        ).reshape(finite_scale_count, 1),
        global_decode_scale=torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        logical_shape=(finite_scale_count, 16),
    )
    expected_scales = torch.tensor(
        [_manual_ue4m3_decode(code) for code in range(finite_scale_count)],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(1).expand(finite_scale_count, 16)
    decoded_scales = cuda_primitives.cuda_dequantize_nvfp4(all_scales_quantized)
    _require_exact("ue4m3-all-finite-codes", decoded_scales, expected_scales)

    zero_scale_quantized = NVFP4Tensor(
        packed_values=packed,
        block_scales=torch.zeros((1, 1), dtype=torch.uint8, device="cuda"),
        global_decode_scale=torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        logical_shape=(1, 16),
    )
    zero_scale_actual = cuda_primitives.cuda_dequantize_nvfp4(
        zero_scale_quantized
    )
    zero_scale_expected = dequantize_nvfp4_reference(zero_scale_quantized)
    _require_exact("synthetic-zero-scale", zero_scale_actual, zero_scale_expected)

    maximum_observed = 0.0
    absolute_error_sum = 0.0
    element_count = 0
    for rows, columns in CASES:
        generator = torch.Generator(device="cpu").manual_seed(
            15_001 + rows * 101 + columns
        )
        source = (
            torch.randn((rows, columns), generator=generator) * 0.75
        ).cuda()
        quantized = quantize_nvfp4_reference(source)
        expected = dequantize_nvfp4_reference(quantized)
        actual = cuda_primitives.cuda_dequantize_nvfp4(quantized)
        torch.cuda.synchronize()

        maximum, mean = _require_exact(
            f"m2b-{rows}x{columns}",
            actual,
            expected,
        )
        maximum_observed = max(maximum_observed, maximum)
        absolute_error_sum += mean * actual.numel()
        element_count += actual.numel()

    aggregate_mean = absolute_error_sum / element_count
    print(f"maximum_observed_abs_error={maximum_observed:.9g}")
    print(f"aggregate_mean_abs_error={aggregate_mean:.9g}")


if __name__ == "__main__":
    main()
