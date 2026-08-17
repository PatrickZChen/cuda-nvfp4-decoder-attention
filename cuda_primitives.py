"""Python boundary for the normally built modular CUDA primitives."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from reference.nvfp4 import NVFP4Tensor


_REPOSITORY_ROOT = Path(__file__).resolve().parent
_LIBRARY_PATH = Path(
    os.environ.get(
        "CUDA_PRIMITIVES_LIBRARY",
        _REPOSITORY_ROOT / "build-cuda" / "cuda_primitives.so",
    )
)

if not _LIBRARY_PATH.is_file():
    raise ImportError(
        f"CUDA primitives library not found at {_LIBRARY_PATH}; "
        "run scripts/build_cuda.sh from the repository root"
    )

torch.ops.load_library(str(_LIBRARY_PATH))


def cuda_unpack_e2m1_codes(packed_values: torch.Tensor) -> torch.Tensor:
    """Unpack portable even-low, odd-high E2M1 bytes on the current stream."""

    return torch.ops.cuda_nvfp4_decoder_attention.cuda_unpack_e2m1_codes(
        packed_values
    )


def cuda_dequantize_nvfp4(quantized: NVFP4Tensor) -> torch.Tensor:
    """Software-decode validated portable NVFP4 storage into CUDA FP32.

    ``NVFP4Tensor`` construction owns canonical byte/value validation. This hot
    path deliberately performs only host-visible metadata checks here and
    structural tensor checks in C++; it does not rescan device scale contents.
    Mutating a validated object's numerical storage afterward is unsupported.
    """

    if not isinstance(quantized, NVFP4Tensor):
        raise TypeError("quantized must be an NVFP4Tensor")
    derived_shape = (
        quantized.packed_values.shape[0],
        quantized.packed_values.shape[1] * 2,
    )
    if quantized.logical_shape != derived_shape:
        raise ValueError(
            "quantized.logical_shape must match packed_values-derived shape "
            f"({quantized.logical_shape} != {derived_shape})"
        )
    return torch.ops.cuda_nvfp4_decoder_attention.cuda_dequantize_nvfp4(
        quantized.packed_values,
        quantized.block_scales,
        quantized.global_decode_scale,
    )


def cuda_w4a16_linear(
    x: torch.Tensor,
    weight: NVFP4Tensor,
) -> torch.Tensor:
    """Project BF16 activations with directly consumed portable NVFP4 weights.

    Weight nibbles and row-local block-scale bytes are decoded inside the CUDA
    projection kernel. This wrapper does not materialize an FP32 weight matrix.
    """

    if not isinstance(weight, NVFP4Tensor):
        raise TypeError("weight must be an NVFP4Tensor")
    derived_shape = (
        weight.packed_values.shape[0],
        weight.packed_values.shape[1] * 2,
    )
    if weight.logical_shape != derived_shape:
        raise ValueError(
            "weight.logical_shape must match packed_values-derived shape "
            f"({weight.logical_shape} != {derived_shape})"
        )
    return torch.ops.cuda_nvfp4_decoder_attention.cuda_w4a16_linear(
        x,
        weight.packed_values,
        weight.block_scales,
        weight.global_decode_scale,
    )


def cuda_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize the final dimension with BF16 storage and FP32 arithmetic."""

    return torch.ops.cuda_nvfp4_decoder_attention.cuda_rms_norm(x, weight, eps)


def cuda_apply_rope(
    x: torch.Tensor,
    past_length: int,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Apply adjacent-pair RoPE in FP32 and return BF16 storage."""

    return torch.ops.cuda_nvfp4_decoder_attention.cuda_apply_rope(
        x,
        past_length,
        rope_theta,
    )


__all__ = [
    "cuda_apply_rope",
    "cuda_dequantize_nvfp4",
    "cuda_rms_norm",
    "cuda_unpack_e2m1_codes",
    "cuda_w4a16_linear",
]
