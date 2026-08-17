"""Portable PyTorch reference for the repository's 1D NVFP4 contract.

The implementation is deliberately expressed with ordinary FP32 PyTorch
operations.  It does not depend on a native FP4 dtype, hardware conversion
instruction, or hardware-specific packed layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


NVFP4_BLOCK_SIZE = 16
E2M1_MAX = 6.0
UE4M3_MAX = 448.0
NVFP4_GLOBAL_RANGE = E2M1_MAX * UE4M3_MAX

_FLOAT32_MAX = torch.finfo(torch.float32).max
_E2M1_DECODE_TABLE = (
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


def _ue4m3_value_for_code(code: int) -> float:
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return math.ldexp(float(mantissa), -9)
    return math.ldexp(1.0 + mantissa / 8.0, exponent - 7)


_UE4M3_DECODE_TABLE = tuple(
    _ue4m3_value_for_code(code) for code in range(0x7F)
)


@dataclass(frozen=True)
class NVFP4Tensor:
    """Validated portable storage for one logical NVFP4 matrix."""

    packed_values: Tensor
    block_scales: Tensor
    global_decode_scale: Tensor
    logical_shape: tuple[int, int]

    def __post_init__(self) -> None:
        _validate_nvfp4_tensor_fields(
            self.packed_values,
            self.block_scales,
            self.global_decode_scale,
            self.logical_shape,
        )


@dataclass(frozen=True)
class NVFP4ErrorMetrics:
    """Numerical quality and representation statistics for an NVFP4 matrix."""

    maximum_absolute_error: float
    mean_absolute_error: float
    rmse: float
    cosine_similarity: float
    zero_fraction: float
    saturation_fraction: float
    maximum_code_fraction: float
    scale_underflow_block_fraction: float


def _require_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_portable_device(name: str, tensor: Tensor) -> None:
    if tensor.device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"{name} must be on a CPU or CUDA device, got {tensor.device}"
        )


def _contains_true(condition: Tensor) -> bool:
    return bool(torch.any(condition).item())


def _validate_source_matrix(source: object, *, name: str = "source") -> Tensor:
    source = _require_tensor(name, source)
    _require_portable_device(name, source)
    if source.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            f"{name} must have dtype torch.float32 or torch.bfloat16, "
            f"got {source.dtype}"
        )
    if source.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got rank {source.ndim}")

    rows, columns = source.shape
    if rows < 1:
        raise ValueError(f"{name} must have at least one row, got {rows}")
    if columns < NVFP4_BLOCK_SIZE:
        raise ValueError(
            f"{name} K dimension must be at least {NVFP4_BLOCK_SIZE}, got {columns}"
        )
    if columns % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"{name} K dimension must be divisible by {NVFP4_BLOCK_SIZE}, "
            f"got {columns}"
        )
    if not source.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if not bool(torch.isfinite(source).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return source


def _validate_logical_shape(logical_shape: object) -> tuple[int, int]:
    if not isinstance(logical_shape, tuple) or len(logical_shape) != 2:
        raise TypeError("logical_shape must be a tuple of two integers")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in logical_shape
    ):
        raise TypeError("logical_shape must be a tuple of two integers")

    rows, columns = logical_shape
    if rows < 1:
        raise ValueError(f"logical_shape N must be at least 1, got {rows}")
    if columns < NVFP4_BLOCK_SIZE:
        raise ValueError(
            "logical_shape K must be at least "
            f"{NVFP4_BLOCK_SIZE}, got {columns}"
        )
    if columns % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            "logical_shape K must be divisible by "
            f"{NVFP4_BLOCK_SIZE}, got {columns}"
        )
    return rows, columns


def _validate_nvfp4_tensor_fields(
    packed_values: object,
    block_scales: object,
    global_decode_scale: object,
    logical_shape: object,
) -> None:
    rows, columns = _validate_logical_shape(logical_shape)
    packed_values = _require_tensor("packed_values", packed_values)
    block_scales = _require_tensor("block_scales", block_scales)
    global_decode_scale = _require_tensor(
        "global_decode_scale", global_decode_scale
    )

    _require_portable_device("packed_values", packed_values)
    if block_scales.device != packed_values.device:
        raise ValueError(
            "block_scales must be on the same device as packed_values "
            f"({block_scales.device} != {packed_values.device})"
        )
    if global_decode_scale.device != packed_values.device:
        raise ValueError(
            "global_decode_scale must be on the same device as packed_values "
            f"({global_decode_scale.device} != {packed_values.device})"
        )

    if packed_values.dtype != torch.uint8:
        raise TypeError(
            "packed_values must have dtype torch.uint8, "
            f"got {packed_values.dtype}"
        )
    if block_scales.dtype != torch.uint8:
        raise TypeError(
            f"block_scales must have dtype torch.uint8, got {block_scales.dtype}"
        )
    if global_decode_scale.dtype != torch.float32:
        raise TypeError(
            "global_decode_scale must have dtype torch.float32, "
            f"got {global_decode_scale.dtype}"
        )

    expected_values_shape = (rows, columns // 2)
    expected_scales_shape = (rows, columns // NVFP4_BLOCK_SIZE)
    if tuple(packed_values.shape) != expected_values_shape:
        raise ValueError(
            f"packed_values must have shape {expected_values_shape}, "
            f"got {tuple(packed_values.shape)}"
        )
    if tuple(block_scales.shape) != expected_scales_shape:
        raise ValueError(
            f"block_scales must have shape {expected_scales_shape}, "
            f"got {tuple(block_scales.shape)}"
        )
    if tuple(global_decode_scale.shape) != ():
        raise ValueError(
            "global_decode_scale must be a scalar tensor with shape (), "
            f"got {tuple(global_decode_scale.shape)}"
        )
    if not packed_values.is_contiguous():
        raise ValueError("packed_values must be contiguous")
    if not block_scales.is_contiguous():
        raise ValueError("block_scales must be contiguous")
    if _contains_true(block_scales > 0x7E):
        raise ValueError(
            "block_scales must contain only canonical finite UE4M3 bytes "
            "in 0x00..0x7e"
        )

    decode_scale = float(global_decode_scale.item())
    if not math.isfinite(decode_scale) or decode_scale < 0.0:
        raise ValueError(
            "global_decode_scale must be finite and nonnegative, "
            f"got {decode_scale}"
        )


def _validate_nvfp4_tensor(value: object) -> NVFP4Tensor:
    if not isinstance(value, NVFP4Tensor):
        raise TypeError("quantized must be an NVFP4Tensor")
    # Tensor contents remain mutable even though the data object is frozen.
    # Revalidate at API boundaries so mutated scale storage is never trusted.
    _validate_nvfp4_tensor_fields(
        value.packed_values,
        value.block_scales,
        value.global_decode_scale,
        value.logical_shape,
    )
    return value


def _validate_floating_values(
    name: str,
    values: object,
    *,
    fp32_only: bool,
) -> Tensor:
    values = _require_tensor(name, values)
    _require_portable_device(name, values)
    allowed_dtypes = (torch.float32,) if fp32_only else (
        torch.float32,
        torch.bfloat16,
    )
    if values.dtype not in allowed_dtypes:
        allowed = "torch.float32" if fp32_only else "torch.float32 or torch.bfloat16"
        raise TypeError(f"{name} must have dtype {allowed}, got {values.dtype}")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return values


def decode_e2m1(codes: Tensor) -> Tensor:
    """Decode E2M1 nibbles to FP32, preserving the distinct signed zeros."""

    codes = _require_tensor("codes", codes)
    _require_portable_device("codes", codes)
    if codes.dtype != torch.uint8:
        raise TypeError(f"codes must have dtype torch.uint8, got {codes.dtype}")
    if _contains_true(codes > 0x0F):
        raise ValueError("E2M1 codes must be nibbles in 0x0..0xf")

    table = torch.tensor(
        _E2M1_DECODE_TABLE,
        dtype=torch.float32,
        device=codes.device,
    )
    return table[codes.to(torch.int64)]


def encode_e2m1(values: Tensor) -> Tensor:
    """Encode finite FP32/BF16 values with E2M1 RNE and finite saturation."""

    values = _validate_floating_values("values", values, fp32_only=False)
    values_fp32 = values.to(torch.float32)
    magnitudes = torch.abs(values_fp32)

    # Each comparison spells out one RNE interval.  Inclusive comparisons are
    # exactly the midpoint cases whose even endpoint is the lower code.
    magnitude_codes = torch.full_like(magnitudes, 0x7, dtype=torch.uint8)
    magnitude_codes = torch.where(
        magnitudes <= 5.0,
        torch.full_like(magnitude_codes, 0x6),
        magnitude_codes,
    )
    magnitude_codes = torch.where(
        magnitudes < 3.5,
        torch.full_like(magnitude_codes, 0x5),
        magnitude_codes,
    )
    magnitude_codes = torch.where(
        magnitudes <= 2.5,
        torch.full_like(magnitude_codes, 0x4),
        magnitude_codes,
    )
    magnitude_codes = torch.where(
        magnitudes < 1.75,
        torch.full_like(magnitude_codes, 0x3),
        magnitude_codes,
    )
    magnitude_codes = torch.where(
        magnitudes <= 1.25,
        torch.full_like(magnitude_codes, 0x2),
        magnitude_codes,
    )
    magnitude_codes = torch.where(
        magnitudes < 0.75,
        torch.full_like(magnitude_codes, 0x1),
        magnitude_codes,
    )
    magnitude_codes = torch.where(
        magnitudes <= 0.25,
        torch.zeros_like(magnitude_codes),
        magnitude_codes,
    )

    sign_codes = torch.where(
        torch.signbit(values_fp32),
        torch.full_like(magnitude_codes, 0x8),
        torch.zeros_like(magnitude_codes),
    )
    return magnitude_codes | sign_codes


def decode_ue4m3(scale_bytes: Tensor) -> Tensor:
    """Decode canonical finite UE4M3 bytes to FP32."""

    scale_bytes = _require_tensor("scale_bytes", scale_bytes)
    _require_portable_device("scale_bytes", scale_bytes)
    if scale_bytes.dtype != torch.uint8:
        raise TypeError(
            f"scale_bytes must have dtype torch.uint8, got {scale_bytes.dtype}"
        )
    if _contains_true(scale_bytes > 0x7E):
        raise ValueError(
            "UE4M3 scale bytes must be canonical finite bytes in 0x00..0x7e; "
            "0x7f and bytes with bit 7 set are invalid"
        )

    table = torch.tensor(
        _UE4M3_DECODE_TABLE,
        dtype=torch.float32,
        device=scale_bytes.device,
    )
    return table[scale_bytes.to(torch.int64)]


def encode_ue4m3(candidates: Tensor) -> Tensor:
    """Encode finite nonnegative FP32 candidates with UE4M3 RNE saturation."""

    candidates = _validate_floating_values(
        "candidates", candidates, fp32_only=True
    )
    if _contains_true(candidates < 0.0):
        raise ValueError("UE4M3 candidates must be nonnegative")

    levels = torch.tensor(
        _UE4M3_DECODE_TABLE,
        dtype=torch.float32,
        device=candidates.device,
    )
    flat_candidates = candidates.reshape(-1)
    insertion = torch.searchsorted(levels, flat_candidates, right=False)
    upper_codes = torch.clamp(insertion, min=0, max=0x7E)
    lower_codes = torch.clamp(insertion - 1, min=0, max=0x7E)
    upper_values = levels[upper_codes]
    lower_values = levels[lower_codes]
    upper_distance = torch.abs(upper_values - flat_candidates)
    lower_distance = torch.abs(flat_candidates - lower_values)

    tie_to_upper_even = (upper_distance == lower_distance) & (
        (upper_codes & 1) == 0
    )
    choose_upper = (upper_distance < lower_distance) | tie_to_upper_even
    encoded = torch.where(choose_upper, upper_codes, lower_codes).to(torch.uint8)
    return encoded.reshape(candidates.shape)


def pack_e2m1_codes(codes: Tensor) -> Tensor:
    """Pack even elements low and odd elements high along a rank-2 K axis."""

    codes = _require_tensor("codes", codes)
    _require_portable_device("codes", codes)
    if codes.dtype != torch.uint8:
        raise TypeError(f"codes must have dtype torch.uint8, got {codes.dtype}")
    if codes.ndim != 2:
        raise ValueError(f"codes must be rank 2, got rank {codes.ndim}")
    if codes.shape[0] < 1 or codes.shape[1] < 2 or codes.shape[1] % 2 != 0:
        raise ValueError(
            "codes must have at least one row and a positive even K dimension"
        )
    if _contains_true(codes > 0x0F):
        raise ValueError("E2M1 codes must be nibbles in 0x0..0xf")

    low_nibbles = codes[:, 0::2]
    high_nibbles = codes[:, 1::2] << 4
    return (low_nibbles | high_nibbles).contiguous()


def unpack_e2m1_codes(packed_values: Tensor) -> Tensor:
    """Unpack repository bytes into even-low, odd-high logical nibbles."""

    packed_values = _require_tensor("packed_values", packed_values)
    _require_portable_device("packed_values", packed_values)
    if packed_values.dtype != torch.uint8:
        raise TypeError(
            f"packed_values must have dtype torch.uint8, got {packed_values.dtype}"
        )
    if packed_values.ndim != 2:
        raise ValueError(
            f"packed_values must be rank 2, got rank {packed_values.ndim}"
        )
    if packed_values.shape[0] < 1 or packed_values.shape[1] < 1:
        raise ValueError("packed_values must have nonempty N and byte dimensions")

    low_nibbles = packed_values & 0x0F
    high_nibbles = packed_values >> 4
    return torch.stack((low_nibbles, high_nibbles), dim=-1).reshape(
        packed_values.shape[0], -1
    ).contiguous()


def _fp32_scalar(value: float, device: torch.device) -> Tensor:
    return torch.tensor(value, dtype=torch.float32, device=device)


def _compute_global_scales_from_fp32(source_fp32: Tensor) -> tuple[Tensor, Tensor]:
    global_amax = torch.amax(torch.abs(source_fp32))
    if float(global_amax.item()) == 0.0:
        global_encode_scale = _fp32_scalar(1.0, source_fp32.device)
    else:
        range_value = _fp32_scalar(NVFP4_GLOBAL_RANGE, source_fp32.device)
        uncapped_encode_scale = range_value / global_amax
        global_encode_scale = torch.minimum(
            uncapped_encode_scale,
            _fp32_scalar(_FLOAT32_MAX, source_fp32.device),
        )

    reciprocal_range = _fp32_scalar(
        1.0 / NVFP4_GLOBAL_RANGE,
        source_fp32.device,
    )
    global_decode_scale = global_amax * reciprocal_range
    return global_encode_scale, global_decode_scale


def compute_nvfp4_global_scales(source: Tensor) -> tuple[Tensor, Tensor]:
    """Return directional ``(global_encode_scale, global_decode_scale)``."""

    source = _validate_source_matrix(source)
    return _compute_global_scales_from_fp32(source.to(torch.float32))


def _conversion_factors(global_encode_scale: Tensor, beta: Tensor) -> Tensor:
    factors = torch.zeros_like(beta, dtype=torch.float32)
    positive_scale = beta > 0.0
    if _contains_true(positive_scale):
        one = _fp32_scalar(1.0, beta.device)
        global_encode_inverse = one / global_encode_scale
        denominators = beta[positive_scale] * global_encode_inverse
        reciprocals = one / denominators
        factors[positive_scale] = torch.minimum(
            reciprocals,
            _fp32_scalar(_FLOAT32_MAX, beta.device),
        )
    return factors


def _block_scale_state(
    source_fp32: Tensor,
    global_encode_scale: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    rows, columns = source_fp32.shape
    blocks = source_fp32.reshape(rows, columns // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    block_amax = torch.amax(torch.abs(blocks), dim=-1)
    one_sixth = _fp32_scalar(1.0 / E2M1_MAX, source_fp32.device)
    candidate_multiplier = global_encode_scale * one_sixth
    scaled_block_candidates = block_amax * candidate_multiplier
    block_scale_bytes = encode_ue4m3(scaled_block_candidates)
    decoded_block_scales = decode_ue4m3(block_scale_bytes)
    return blocks, block_amax, block_scale_bytes, decoded_block_scales


def quantize_nvfp4_reference(source: Tensor) -> NVFP4Tensor:
    """Quantize a contiguous FP32/BF16 matrix into portable 1D NVFP4."""

    source = _validate_source_matrix(source)
    source_fp32 = source.to(torch.float32)
    rows, columns = source_fp32.shape
    global_encode_scale, global_decode_scale = _compute_global_scales_from_fp32(
        source_fp32
    )
    blocks, _, block_scale_bytes, decoded_block_scales = _block_scale_state(
        source_fp32,
        global_encode_scale,
    )
    conversion_factors = _conversion_factors(
        global_encode_scale,
        decoded_block_scales,
    )

    # Scale-zero blocks never enter a reciprocal or E2M1 conversion path.
    # Their preinitialized payload is the repository's canonical +0 code.
    block_codes = torch.zeros_like(blocks, dtype=torch.uint8)
    positive_scale = decoded_block_scales > 0.0
    if _contains_true(positive_scale):
        pre_e2m1 = (
            blocks[positive_scale]
            * conversion_factors[positive_scale].unsqueeze(-1)
        )
        block_codes[positive_scale] = encode_e2m1(pre_e2m1)

    logical_codes = block_codes.reshape(rows, columns)
    packed_values = pack_e2m1_codes(logical_codes)
    return NVFP4Tensor(
        packed_values=packed_values,
        block_scales=block_scale_bytes.contiguous(),
        global_decode_scale=global_decode_scale,
        logical_shape=(rows, columns),
    )


def dequantize_nvfp4_reference(quantized: NVFP4Tensor) -> Tensor:
    """Reconstruct a portable NVFP4 matrix as FP32 in the frozen order."""

    quantized = _validate_nvfp4_tensor(quantized)
    rows, columns = quantized.logical_shape
    logical_codes = unpack_e2m1_codes(quantized.packed_values)
    decoded_values = decode_e2m1(logical_codes).reshape(
        rows,
        columns // NVFP4_BLOCK_SIZE,
        NVFP4_BLOCK_SIZE,
    )
    decoded_block_scales = decode_ue4m3(quantized.block_scales)

    block_scaled = decoded_values * decoded_block_scales.unsqueeze(-1)
    reconstructed = block_scaled * quantized.global_decode_scale
    return reconstructed.reshape(rows, columns).contiguous()


def analyze_nvfp4_error(
    source: Tensor,
    quantized: NVFP4Tensor,
) -> NVFP4ErrorMetrics:
    """Compute the frozen FP64-reduction error and representation metrics."""

    source = _validate_source_matrix(source)
    quantized = _validate_nvfp4_tensor(quantized)
    if tuple(source.shape) != quantized.logical_shape:
        raise ValueError(
            "source shape must match quantized.logical_shape "
            f"({tuple(source.shape)} != {quantized.logical_shape})"
        )
    if source.device != quantized.packed_values.device:
        raise ValueError(
            "source and quantized storage must be on the same device "
            f"({source.device} != {quantized.packed_values.device})"
        )

    source_fp32 = source.to(torch.float32)
    reconstructed_fp32 = dequantize_nvfp4_reference(quantized)
    source_fp64 = source_fp32.to(torch.float64)
    reconstructed_fp64 = reconstructed_fp32.to(torch.float64)
    difference = reconstructed_fp64 - source_fp64
    absolute_error = torch.abs(difference)
    element_count = source.numel()

    maximum_absolute_error = float(torch.amax(absolute_error).item())
    mean_absolute_error = float(torch.mean(absolute_error, dtype=torch.float64).item())
    rmse = float(
        torch.sqrt(torch.mean(difference * difference, dtype=torch.float64)).item()
    )

    source_norm_squared = torch.sum(source_fp64 * source_fp64, dtype=torch.float64)
    reconstructed_norm_squared = torch.sum(
        reconstructed_fp64 * reconstructed_fp64,
        dtype=torch.float64,
    )
    source_is_zero = float(source_norm_squared.item()) == 0.0
    reconstructed_is_zero = float(reconstructed_norm_squared.item()) == 0.0
    if source_is_zero and reconstructed_is_zero:
        cosine_similarity = 1.0
    elif source_is_zero or reconstructed_is_zero:
        cosine_similarity = 0.0
    else:
        dot_product = torch.sum(
            source_fp64 * reconstructed_fp64,
            dtype=torch.float64,
        )
        denominator = torch.sqrt(
            source_norm_squared * reconstructed_norm_squared
        )
        cosine_similarity = float((dot_product / denominator).item())

    logical_codes = unpack_e2m1_codes(quantized.packed_values)
    zero_count = torch.count_nonzero((logical_codes == 0x0) | (logical_codes == 0x8))
    zero_fraction = float(zero_count.to(torch.float64).item() / element_count)

    decoded_values = decode_e2m1(logical_codes)
    maximum_code_count = torch.count_nonzero(torch.abs(decoded_values) == E2M1_MAX)
    maximum_code_fraction = float(
        maximum_code_count.to(torch.float64).item() / element_count
    )

    global_encode_scale, _ = _compute_global_scales_from_fp32(source_fp32)
    rows, columns = source.shape
    source_blocks = source_fp32.reshape(
        rows,
        columns // NVFP4_BLOCK_SIZE,
        NVFP4_BLOCK_SIZE,
    )
    block_amax = torch.amax(torch.abs(source_blocks), dim=-1)
    decoded_block_scales = decode_ue4m3(quantized.block_scales)
    positive_scale = decoded_block_scales > 0.0
    conversion_factors = _conversion_factors(
        global_encode_scale,
        decoded_block_scales,
    )
    pre_e2m1 = torch.zeros_like(source_blocks, dtype=torch.float32)
    if _contains_true(positive_scale):
        pre_e2m1[positive_scale] = (
            source_blocks[positive_scale]
            * conversion_factors[positive_scale].unsqueeze(-1)
        )

    decoded_value_blocks = decoded_values.reshape_as(source_blocks)
    saturation_mask = (
        positive_scale.unsqueeze(-1)
        & (torch.abs(pre_e2m1) > E2M1_MAX)
        & (torch.abs(decoded_value_blocks) == E2M1_MAX)
    )
    saturation_count = torch.count_nonzero(saturation_mask)
    saturation_fraction = float(
        saturation_count.to(torch.float64).item() / element_count
    )

    scale_underflow_count = torch.count_nonzero(
        (block_amax > 0.0) & (decoded_block_scales == 0.0)
    )
    block_count = rows * (columns // NVFP4_BLOCK_SIZE)
    scale_underflow_block_fraction = float(
        scale_underflow_count.to(torch.float64).item() / block_count
    )

    return NVFP4ErrorMetrics(
        maximum_absolute_error=maximum_absolute_error,
        mean_absolute_error=mean_absolute_error,
        rmse=rmse,
        cosine_similarity=cosine_similarity,
        zero_fraction=zero_fraction,
        saturation_fraction=saturation_fraction,
        maximum_code_fraction=maximum_code_fraction,
        scale_underflow_block_fraction=scale_underflow_block_fraction,
    )


__all__ = [
    "E2M1_MAX",
    "NVFP4_BLOCK_SIZE",
    "NVFP4_GLOBAL_RANGE",
    "NVFP4ErrorMetrics",
    "NVFP4Tensor",
    "UE4M3_MAX",
    "analyze_nvfp4_error",
    "compute_nvfp4_global_scales",
    "decode_e2m1",
    "decode_ue4m3",
    "dequantize_nvfp4_reference",
    "encode_e2m1",
    "encode_ue4m3",
    "pack_e2m1_codes",
    "quantize_nvfp4_reference",
    "unpack_e2m1_codes",
]
