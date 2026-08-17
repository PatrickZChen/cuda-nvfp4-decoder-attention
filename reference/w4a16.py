"""Explicit correctness oracle for direct portable-NVFP4 W4A16 projection."""

from __future__ import annotations

import torch
from torch import Tensor

from .nvfp4 import NVFP4Tensor, dequantize_nvfp4_reference


def w4a16_linear_reference(x: Tensor, quantized: NVFP4Tensor) -> Tensor:
    """Compute ``x @ W_hat.T`` with FP32 products/reduction and BF16 output.

    This oracle intentionally materializes the reconstructed FP32 matrix. The
    direct CUDA operator does not; the materialization is only a transparent
    correctness path.
    """

    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"x must have dtype torch.bfloat16, got {x.dtype}")
    if x.ndim < 1 or x.ndim > 4:
        raise ValueError(f"x must have rank in [1, 4], got rank {x.ndim}")
    if x.numel() == 0:
        raise ValueError("x must be nonempty")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not isinstance(quantized, NVFP4Tensor):
        raise TypeError("quantized must be an NVFP4Tensor")

    rows, reduction_size = quantized.logical_shape
    if x.shape[-1] != reduction_size:
        raise ValueError(
            "x final dimension must match quantized weight K "
            f"({x.shape[-1]} != {reduction_size})"
        )
    if x.device != quantized.packed_values.device:
        raise ValueError(
            "x and quantized storage must be on the same device "
            f"({x.device} != {quantized.packed_values.device})"
        )

    reconstructed_weight = dequantize_nvfp4_reference(quantized)
    x_rows_fp32 = x.to(torch.float32).reshape(-1, reduction_size)

    # One output feature at a time keeps the FP32 multiply and reduction
    # visible and avoids dependence on matmul/TF32 backend modes.
    output_columns = [
        torch.sum(
            x_rows_fp32 * reconstructed_weight[index][None, :],
            dim=-1,
            dtype=torch.float32,
        )
        for index in range(rows)
    ]
    result_shape = (*x.shape[:-1], rows)
    result_fp32 = torch.stack(output_columns, dim=-1).reshape(result_shape)
    return result_fp32.to(torch.bfloat16)


__all__ = ["w4a16_linear_reference"]
