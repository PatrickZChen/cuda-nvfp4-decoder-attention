"""Run deterministic CUDA RoPE and modular Q/K checks with error reports."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cuda_primitives  # noqa: E402
from reference import apply_rope_reference, rms_norm_reference  # noqa: E402


ACCEPTED_MAX_ABSOLUTE_ERROR = 0.0
CASES = (
    ("d2", (1, 1, 3, 2), 1),
    ("multi-token", (2, 3, 4, 8), 17),
    ("d128", (1, 2, 2, 128), 128),
    ("canonical-q", (1, 4, 24, 128), 2_048),
    ("canonical-k", (2, 1, 6, 128), 8_192),
)


def _deterministic_input(
    shape: tuple[int, int, int, int],
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=generator) * 0.75).to(torch.bfloat16)


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = (actual.float() - expected.float()).abs()
    return float(error.max().item()), float(error.mean().item())


def _integration_case(
    label: str,
    shape: tuple[int, int, int, int],
    seed: int,
) -> tuple[float, float]:
    projected = _deterministic_input(shape, seed).cuda()
    weight = torch.linspace(
        0.5,
        1.5,
        shape[-1],
        dtype=torch.float32,
    ).to(torch.bfloat16).cuda()
    expected_normalized = rms_norm_reference(projected, weight, 1.0e-6)
    expected = apply_rope_reference(expected_normalized, 512, 10_000.0)
    actual_normalized = cuda_primitives.cuda_rms_norm(
        projected,
        weight,
        1.0e-6,
    )
    actual = cuda_primitives.cuda_apply_rope(
        actual_normalized,
        512,
        10_000.0,
    )
    torch.cuda.synchronize()

    if not torch.equal(actual_normalized, expected_normalized):
        raise AssertionError(f"{label} CUDA RMSNorm diverged from its reference")
    maximum, mean = _errors(actual, expected)
    print(f"{label}_integration shape={shape} max_abs={maximum:.9g} mean_abs={mean:.9g}")
    return maximum, mean


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the CUDA device does not support BF16")

    print(f"device={torch.cuda.get_device_name()}")
    print(f"capability={torch.cuda.get_device_capability()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")

    identity_cpu = _deterministic_input((2, 1, 5, 8), seed=8_029)
    identity_cpu[0, 0, 0, 0] = 0.0
    identity_cpu[0, 0, 0, 1] = -0.0
    identity = identity_cpu.cuda()
    identity_actual = cuda_primitives.cuda_apply_rope(identity, 0, 10_000.0)
    torch.cuda.synchronize()
    identity_exact = torch.equal(
        identity_actual.cpu().view(torch.int16),
        identity_cpu.view(torch.int16),
    )
    print(f"position_zero_storage_exact={identity_exact}")
    if not identity_exact:
        raise AssertionError("position-zero RoPE was not storage exact")

    maximum_observed = 0.0
    absolute_error_sum = 0.0
    element_count = 0
    for index, (label, shape, past_length) in enumerate(CASES):
        x = _deterministic_input(shape, 9_031 + index * 101).cuda()
        expected = apply_rope_reference(x, past_length, 10_000.0)
        actual = cuda_primitives.cuda_apply_rope(x, past_length, 10_000.0)
        torch.cuda.synchronize()

        error = (actual.float() - expected.float()).abs()
        maximum = float(error.max().item())
        mean = float(error.mean().item())
        maximum_observed = max(maximum_observed, maximum)
        absolute_error_sum += float(error.sum(dtype=torch.float64).item())
        element_count += error.numel()
        print(
            f"case={label} shape={shape} past_length={past_length} "
            f"max_abs={maximum:.9g} mean_abs={mean:.9g}"
        )

    q_maximum, q_mean = _integration_case(
        "q",
        (2, 4, 24, 128),
        seed=10_037,
    )
    k_maximum, k_mean = _integration_case(
        "k",
        (2, 4, 6, 128),
        seed=10_039,
    )
    maximum_observed = max(maximum_observed, q_maximum, k_maximum)

    aggregate_mean = absolute_error_sum / element_count
    print(f"maximum_observed_nonzero_abs_error={maximum_observed:.9g}")
    print(f"aggregate_nonzero_mean_abs_error={aggregate_mean:.9g}")
    print(f"q_integration_mean_abs_error={q_mean:.9g}")
    print(f"k_integration_mean_abs_error={k_mean:.9g}")
    if maximum_observed > ACCEPTED_MAX_ABSOLUTE_ERROR:
        raise AssertionError(
            "CUDA RoPE diverged from the frozen BF16 reference; "
            f"maximum absolute error was {maximum_observed}"
        )


if __name__ == "__main__":
    main()
