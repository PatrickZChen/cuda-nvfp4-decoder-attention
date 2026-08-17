"""Run deterministic custom-kernel RMSNorm checks and report actual error."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cuda_rmsnorm  # noqa: E402
from reference import rms_norm_reference  # noqa: E402


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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the CUDA device does not support BF16")

    print(f"device={torch.cuda.get_device_name()}")
    print(f"capability={torch.cuda.get_device_capability()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")

    maximum_observed = 0.0
    maximum_cpu_reference_error = 0.0
    for shape in SHAPES:
        generator = torch.Generator(device="cpu").manual_seed(
            1_003 + math.prod(shape)
        )
        x_cpu = (torch.randn(shape, generator=generator) * 0.75).to(
            torch.bfloat16
        )
        weight_cpu = torch.linspace(
            0.5,
            1.5,
            shape[-1],
            dtype=torch.float32,
        ).to(torch.bfloat16)
        expected_cpu = rms_norm_reference(x_cpu, weight_cpu, 1.0e-6)
        x_cuda = x_cpu.cuda()
        weight_cuda = weight_cpu.cuda()
        expected_cuda = rms_norm_reference(x_cuda, weight_cuda, 1.0e-6)
        actual = cuda_rmsnorm.cuda_rms_norm(
            x_cuda,
            weight_cuda,
            1.0e-6,
        )
        torch.cuda.synchronize()

        actual_cpu = actual.cpu()
        error = (actual_cpu.float() - expected_cuda.cpu().float()).abs()
        cpu_reference_error = (
            actual_cpu.float() - expected_cpu.float()
        ).abs()
        maximum = float(error.max().item())
        mean = float(error.mean().item())
        cpu_reference_maximum = float(cpu_reference_error.max().item())
        maximum_observed = max(maximum_observed, maximum)
        maximum_cpu_reference_error = max(
            maximum_cpu_reference_error,
            cpu_reference_maximum,
        )
        print(
            f"shape={shape} max_abs={maximum:.9g} mean_abs={mean:.9g} "
            f"cpu_reference_max_abs={cpu_reference_maximum:.9g}"
        )

    print(f"maximum_observed_abs_error={maximum_observed:.9g}")
    print(
        "maximum_cpu_reference_abs_error="
        f"{maximum_cpu_reference_error:.9g}"
    )
    if maximum_observed != 0.0:
        raise AssertionError(
            "CUDA RMSNorm diverged from the frozen BF16 reference; "
            f"maximum absolute error was {maximum_observed}"
        )


if __name__ == "__main__":
    main()
