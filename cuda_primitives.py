"""Python boundary for the normally built modular CUDA primitives."""

from __future__ import annotations

import os
from pathlib import Path

import torch


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


__all__ = ["cuda_apply_rope", "cuda_rms_norm"]
