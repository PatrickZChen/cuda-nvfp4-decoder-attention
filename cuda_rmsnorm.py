"""Python boundary for the normally built CUDA RMSNorm operator."""

from __future__ import annotations

import os
from pathlib import Path

import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parent
_LIBRARY_PATH = Path(
    os.environ.get(
        "CUDA_RMSNORM_LIBRARY",
        _REPOSITORY_ROOT / "build-cuda" / "cuda_rmsnorm.so",
    )
)

if not _LIBRARY_PATH.is_file():
    raise ImportError(
        f"CUDA RMSNorm library not found at {_LIBRARY_PATH}; "
        "run scripts/build_cuda.sh from the repository root"
    )

torch.ops.load_library(str(_LIBRARY_PATH))


def cuda_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize the final dimension with BF16 storage and FP32 arithmetic."""

    return torch.ops.cuda_nvfp4_decoder_attention.cuda_rms_norm(x, weight, eps)


__all__ = ["cuda_rms_norm"]
