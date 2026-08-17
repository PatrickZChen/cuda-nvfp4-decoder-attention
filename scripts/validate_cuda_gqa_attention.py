"""Run the three-stage CUDA causal GQA baseline under validation tools."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cuda_primitives  # noqa: E402
from reference import gqa_attention_reference  # noqa: E402


CASES = (
    # label, B, T, Hq, Hkv, D, P, seed
    ("t1-p0-canonical", 1, 1, 24, 6, 128, 0, 60_001),
    ("t1-p128-canonical", 1, 1, 24, 6, 128, 128, 60_007),
    ("t4-p2-causal-ratio4", 1, 4, 8, 2, 32, 2, 60_013),
    ("context301-ratio2", 1, 1, 8, 4, 128, 300, 60_017),
)

ACCEPTED_MAXIMUM_ABSOLUTE_ERROR = 2.0**-7
ACCEPTED_MEAN_ABSOLUTE_ERROR = 2.0**-16
ADJACENCY_ABSOLUTE_FLOOR = 2.0**-20


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
    if maximum > ACCEPTED_MAXIMUM_ABSOLUTE_ERROR:
        raise AssertionError(f"{label} maximum absolute error was {maximum}")
    if mean > ACCEPTED_MEAN_ABSOLUTE_ERROR:
        raise AssertionError(f"{label} mean absolute error was {mean}")
    if not torch.all(
        (adjacency <= 1) | (error <= ADJACENCY_ABSOLUTE_FLOOR)
    ).item():
        raise AssertionError(
            f"{label} had a non-cancellation BF16 adjacency distance above one"
        )
    return maximum, mean, exact_count, actual.numel(), maximum_distance


def _generated_case(
    label: str,
    batch_size: int,
    token_count: int,
    query_head_count: int,
    kv_head_count: int,
    head_dim: int,
    past_length: int,
    seed: int,
) -> tuple[float, float, int, int, int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    context_length = past_length + token_count
    q = (
        torch.randn(
            (batch_size, token_count, query_head_count, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16).cuda()
    present_k = (
        torch.randn(
            (batch_size, kv_head_count, context_length, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16).cuda()
    present_v = (
        torch.randn(
            (batch_size, kv_head_count, context_length, head_dim),
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16).cuda()
    _, _, expected = gqa_attention_reference(
        q,
        present_k,
        present_v,
        past_length,
        return_attention=False,
    )
    actual = cuda_primitives.cuda_gqa_attention(
        q,
        present_k,
        present_v,
        past_length,
    )
    return _check(label, actual, expected)


def _hand_gqa_case() -> tuple[float, float, int, int, int]:
    q = torch.zeros((1, 1, 4, 8), dtype=torch.bfloat16, device="cuda")
    present_k = torch.zeros((1, 2, 1, 8), dtype=torch.bfloat16, device="cuda")
    present_v = torch.empty_like(present_k)
    present_v[0, 0, 0].fill_(3.0)
    present_v[0, 1, 0].fill_(17.0)
    expected = torch.empty_like(q)
    expected[0, 0, 0:2].fill_(3.0)
    expected[0, 0, 2:4].fill_(17.0)
    actual = cuda_primitives.cuda_gqa_attention(q, present_k, present_v, 0)
    result = _check("hand-gqa-map", actual, expected)
    if not torch.equal(actual, expected):
        raise AssertionError("hand-computable GQA mapping was not exact")
    return result


def _causal_case() -> tuple[float, float, int, int, int]:
    past_length = 2
    token_count = 3
    q = torch.zeros((1, token_count, 4, 8), dtype=torch.bfloat16, device="cuda")
    present_k = torch.zeros((1, 2, 5, 8), dtype=torch.bfloat16, device="cuda")
    present_v = torch.empty_like(present_k)
    for kv_head, multiplier in enumerate((1.0, 10.0)):
        for key, value in enumerate((3.0, 6.0, 9.0, 1_200.0, 24_000.0)):
            present_v[0, kv_head, key].fill_(multiplier * value)
    expected = torch.empty_like(q)
    group_size = 2
    for token in range(token_count):
        visible_count = past_length + token + 1
        for query_head in range(4):
            kv_head = query_head // group_size
            total = sum(
                float(present_v[0, kv_head, key, 0].float().cpu())
                for key in range(visible_count)
            )
            expected[0, token, query_head].fill_(total / visible_count)
    actual = cuda_primitives.cuda_gqa_attention(
        q,
        present_k,
        present_v,
        past_length,
    )
    result = _check("hand-t3-p2-causal", actual, expected)
    if not torch.equal(actual, expected):
        raise AssertionError("hand-computable causal visibility was not exact")
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the CUDA device does not support BF16")

    print(f"device={torch.cuda.get_device_name()}")
    print(f"capability={torch.cuda.get_device_capability()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")

    results = [_hand_gqa_case(), _causal_case()]
    results.extend(_generated_case(*case) for case in CASES)
    torch.cuda.synchronize()

    maximum_observed = max(result[0] for result in results)
    total_absolute_error = sum(result[1] * result[3] for result in results)
    exact_count = sum(result[2] for result in results)
    element_count = sum(result[3] for result in results)
    maximum_distance = max(result[4] for result in results)
    print(f"maximum_observed_abs_error={maximum_observed:.9g}")
    print(f"aggregate_mean_abs_error={total_absolute_error / element_count:.9g}")
    print(f"aggregate_exact_bf16_fraction={exact_count / element_count:.9g}")
    print(f"maximum_observed_bf16_distance={maximum_distance}")
    print("gqa_score_softmax_pv_kernels_executed=true")


if __name__ == "__main__":
    main()
