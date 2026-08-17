"""Run direct W4A16 CUDA correctness cases under validation tools."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cuda_primitives  # noqa: E402
from reference import (  # noqa: E402
    NVFP4Tensor,
    quantize_nvfp4_reference,
    w4a16_linear_reference,
)


CASES = (
    ("multiple-rows", 3, 5, 32, 20_003),
    ("k128", 2, 8, 128, 20_009),
    ("k3072", 2, 4, 3072, 20_011),
)


def _move_quantized(quantized: NVFP4Tensor) -> NVFP4Tensor:
    return NVFP4Tensor(
        packed_values=quantized.packed_values.cuda(),
        block_scales=quantized.block_scales.cuda(),
        global_decode_scale=quantized.global_decode_scale.cuda(),
        logical_shape=quantized.logical_shape,
    )


def _bf16_ordered_keys(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (bits & 0x8000) != 0
    magnitude = bits & 0x7FFF
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def _check(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float, int, int, int]:
    error = (actual.float() - expected.float()).abs()
    adjacency = (
        _bf16_ordered_keys(actual) - _bf16_ordered_keys(expected)
    ).abs()
    maximum = float(error.max().item())
    mean = float(error.mean().item())
    exact_count = int(torch.count_nonzero(actual == expected).item())
    maximum_distance = int(adjacency.max().item())
    print(
        f"case={label} shape={tuple(actual.shape)} max_abs={maximum:.9g} "
        f"mean_abs={mean:.9g} exact_fraction="
        f"{exact_count / actual.numel():.9g} "
        f"max_bf16_distance={maximum_distance}"
    )
    if maximum_distance > 1:
        raise AssertionError(
            f"{label} differed by more than one adjacent BF16 value: "
            f"maximum distance {maximum_distance}"
        )
    return maximum, mean, exact_count, actual.numel(), maximum_distance


def _hand_case() -> tuple[float, float, int, int, int]:
    weight_cpu = NVFP4Tensor(
        packed_values=torch.tensor(
            [[0x42] * 8, [0xA2] * 8],
            dtype=torch.uint8,
        ),
        block_scales=torch.tensor(
            [[0x38], [0x38]],
            dtype=torch.uint8,
        ),
        global_decode_scale=torch.tensor(1.0, dtype=torch.float32),
        logical_shape=(2, 16),
    )
    x_cpu = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.arange(1, 17, dtype=torch.float32).to(torch.bfloat16),
        )
    )
    expected_cpu = torch.tensor(
        [[24.0, 0.0], [208.0, -8.0]],
        dtype=torch.bfloat16,
    )
    if not torch.equal(w4a16_linear_reference(x_cpu, weight_cpu), expected_cpu):
        raise AssertionError("the independent hand-case oracle is incorrect")

    actual = cuda_primitives.cuda_w4a16_linear(
        x_cpu.cuda(),
        _move_quantized(weight_cpu),
    )
    expected = expected_cpu.cuda()
    result = _check("hand-k16", actual, expected)
    if not torch.equal(actual, expected):
        raise AssertionError("the hand-computable K=16 case was not exact")
    return result


def _generated_case(
    label: str,
    activation_rows: int,
    output_features: int,
    reduction_size: int,
    seed: int,
) -> tuple[float, float, int, int, int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = torch.randn(
        (output_features, reduction_size),
        generator=generator,
    ) * 0.5
    x_cpu = (
        torch.randn(
            (activation_rows, reduction_size),
            generator=generator,
        )
        * 0.75
    ).to(torch.bfloat16)
    weight_cpu = quantize_nvfp4_reference(source)
    expected_cpu = w4a16_linear_reference(x_cpu, weight_cpu)
    actual = cuda_primitives.cuda_w4a16_linear(
        x_cpu.cuda(),
        _move_quantized(weight_cpu),
    )
    return _check(label, actual, expected_cpu.cuda())


def _zero_scale_case() -> tuple[float, float, int, int, int]:
    weight_cpu = NVFP4Tensor(
        packed_values=torch.tensor(
            [[0x42] * 16, [0xA2] * 16],
            dtype=torch.uint8,
        ),
        block_scales=torch.tensor(
            [[0x00, 0x38], [0x38, 0x00]],
            dtype=torch.uint8,
        ),
        global_decode_scale=torch.tensor(1.0, dtype=torch.float32),
        logical_shape=(2, 32),
    )
    x_cpu = torch.arange(1, 65, dtype=torch.float32).reshape(2, 32).to(
        torch.bfloat16
    )
    expected_cpu = w4a16_linear_reference(x_cpu, weight_cpu)
    actual = cuda_primitives.cuda_w4a16_linear(
        x_cpu.cuda(),
        _move_quantized(weight_cpu),
    )
    return _check("zero-scale", actual, expected_cpu.cuda())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the CUDA device does not support BF16")

    print(f"device={torch.cuda.get_device_name()}")
    print(f"capability={torch.cuda.get_device_capability()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")

    results = [_hand_case()]
    results.extend(_generated_case(*case) for case in CASES)
    results.append(_zero_scale_case())
    torch.cuda.synchronize()

    maximum_observed = max(result[0] for result in results)
    total_absolute_error = sum(result[1] * result[3] for result in results)
    exact_count = sum(result[2] for result in results)
    element_count = sum(result[3] for result in results)
    maximum_distance = max(result[4] for result in results)
    print(f"maximum_observed_abs_error={maximum_observed:.9g}")
    print(
        "aggregate_mean_abs_error="
        f"{total_absolute_error / element_count:.9g}"
    )
    print(f"aggregate_exact_bf16_fraction={exact_count / element_count:.9g}")
    print(f"maximum_observed_bf16_distance={maximum_distance}")
    print("direct_w4a16_kernel_executed=true")


if __name__ == "__main__":
    main()
