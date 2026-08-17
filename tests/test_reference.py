"""Semantic tests for the Milestone 1 decoder-attention reference."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from itertools import product

import pytest
import torch

from reference import (
    DecoderAttentionConfig,
    apply_rope_reference,
    decoder_attention_reference,
    gqa_attention_reference,
    linear_reference,
    reshape_heads_reference,
    rms_norm_reference,
)


BF16_EPSILON = 2.0**-7


def _small_config(
    *,
    num_query_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 2,
    rms_eps: float = 1.0e-6,
    rope_theta: float = 10_000.0,
) -> DecoderAttentionConfig:
    return DecoderAttentionConfig(
        hidden_size=num_query_heads * head_dim,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rms_eps=rms_eps,
        rope_theta=rope_theta,
    )


def _make_tensors(
    config: DecoderAttentionConfig,
    *,
    batch_size: int = 1,
    tokens: int = 3,
    seed: int = 17,
) -> dict[str, torch.Tensor]:
    """Create deterministic, normal-range BF16 inputs and parameters."""

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def normal(shape: tuple[int, ...], scale: float = 0.3) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    hidden_size = config.hidden_size
    kv_width = config.num_kv_heads * config.head_dim
    return {
        "x": normal((batch_size, tokens, hidden_size), 0.6),
        "input_norm_weight": torch.linspace(
            0.75, 1.25, hidden_size, dtype=torch.float32
        ).to(torch.bfloat16),
        "q_weight": normal((hidden_size, hidden_size)),
        "k_weight": normal((kv_width, hidden_size)),
        "v_weight": normal((kv_width, hidden_size)),
        "q_norm_weight": torch.linspace(
            0.8, 1.2, config.head_dim, dtype=torch.float32
        ).to(torch.bfloat16),
        "k_norm_weight": torch.linspace(
            1.1, 0.7, config.head_dim, dtype=torch.float32
        ).to(torch.bfloat16),
        "out_weight": normal((hidden_size, hidden_size)),
    }


def _run(
    config: DecoderAttentionConfig,
    tensors: dict[str, torch.Tensor],
    *,
    past_k: torch.Tensor | None = None,
    past_v: torch.Tensor | None = None,
    return_debug: bool = False,
):
    return decoder_attention_reference(
        **tensors,
        past_k=past_k,
        past_v=past_v,
        config=config,
        return_debug=return_debug,
    )


def _manual_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Small scalar-loop oracle, independent of the production helper."""

    result = torch.empty_like(x, dtype=torch.bfloat16)
    for prefix in product(*(range(size) for size in x.shape[:-1])):
        values = [float(value) for value in x[prefix].float()]
        mean_square = sum(value * value for value in values) / len(values)
        inverse_rms = 1.0 / math.sqrt(mean_square + eps)
        normalized = [
            value * inverse_rms * float(weight[index].float())
            for index, value in enumerate(values)
        ]
        result[prefix] = torch.tensor(normalized, dtype=torch.bfloat16)
    return result


def _manual_adjacent_rope(
    x: torch.Tensor,
    *,
    position_offset: int,
    rope_theta: float,
    store_bf16: bool = True,
) -> torch.Tensor:
    """Explicit scalar adjacent-pair rotation used only as a test oracle."""

    batch_size, tokens, heads, head_dim = x.shape
    result = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for batch in range(batch_size):
        for token in range(tokens):
            position = position_offset + token
            for head in range(heads):
                for pair in range(head_dim // 2):
                    even_index = 2 * pair
                    odd_index = even_index + 1
                    angle = position / (rope_theta ** (even_index / head_dim))
                    cosine = math.cos(angle)
                    sine = math.sin(angle)
                    even = float(x[batch, token, head, even_index].float())
                    odd = float(x[batch, token, head, odd_index].float())
                    result[batch, token, head, even_index] = (
                        even * cosine - odd * sine
                    )
                    result[batch, token, head, odd_index] = (
                        even * sine + odd * cosine
                    )
    return result.to(torch.bfloat16) if store_bf16 else result


def _zero_tensors(
    config: DecoderAttentionConfig, *, tokens: int
) -> dict[str, torch.Tensor]:
    tensors = _make_tensors(config, tokens=tokens)
    for name in ("x", "q_weight", "k_weight", "v_weight", "out_weight"):
        tensors[name] = torch.zeros_like(tensors[name])
    tensors["input_norm_weight"] = torch.ones_like(tensors["input_norm_weight"])
    tensors["q_norm_weight"] = torch.ones_like(tensors["q_norm_weight"])
    tensors["k_norm_weight"] = torch.ones_like(tensors["k_norm_weight"])
    return tensors


def test_canonical_and_reduced_configurations_are_valid() -> None:
    canonical = DecoderAttentionConfig()
    assert canonical.hidden_size == 3072
    assert canonical.num_query_heads == 24
    assert canonical.num_kv_heads == 6
    assert canonical.head_dim == 128
    assert canonical.rms_eps == 1.0e-6
    assert canonical.rope_theta == 10_000.0

    reduced = _small_config(num_query_heads=4, num_kv_heads=1, head_dim=4)
    assert reduced.hidden_size == 16
    assert reduced.num_query_heads // reduced.num_kv_heads == 4

    with pytest.raises((FrozenInstanceError, AttributeError)):
        reduced.hidden_size = 32  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 7, "num_query_heads": 4, "num_kv_heads": 2, "head_dim": 2},
        {"hidden_size": 6, "num_query_heads": 3, "num_kv_heads": 2, "head_dim": 2},
        {"hidden_size": 9, "num_query_heads": 3, "num_kv_heads": 1, "head_dim": 3},
        {"hidden_size": 0},
        {"num_query_heads": 0},
        {"num_kv_heads": 0},
        {"head_dim": 0},
        {"rms_eps": 0.0},
        {"rms_eps": -1.0e-6},
        {"rms_eps": math.inf},
        {"rms_eps": math.nan},
        {"rope_theta": 0.0},
        {"rope_theta": -1.0},
        {"rope_theta": math.inf},
        {"rope_theta": math.nan},
    ],
    ids=[
        "hidden-size-mismatch",
        "nondivisible-gqa",
        "odd-head-dimension",
        "zero-hidden-size",
        "zero-query-heads",
        "zero-kv-heads",
        "zero-head-dimension",
        "zero-epsilon",
        "negative-epsilon",
        "infinite-epsilon",
        "nan-epsilon",
        "zero-rope-theta",
        "negative-rope-theta",
        "infinite-rope-theta",
        "nan-rope-theta",
    ],
)
def test_invalid_configurations_are_rejected(kwargs: dict[str, object]) -> None:
    defaults: dict[str, object] = {
        "hidden_size": 8,
        "num_query_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 2,
        "rms_eps": 1.0e-6,
        "rope_theta": 10_000.0,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        DecoderAttentionConfig(**defaults)  # type: ignore[arg-type]


def test_rms_norm_matches_hand_calculation() -> None:
    x = torch.tensor([[[3.0, 4.0]]], dtype=torch.bfloat16)
    weight = torch.tensor([2.0, 0.5], dtype=torch.bfloat16)
    eps = 1.0e-6

    inverse_rms = 1.0 / math.sqrt((3.0**2 + 4.0**2) / 2.0 + eps)
    expected = torch.tensor(
        [[[3.0 * inverse_rms * 2.0, 4.0 * inverse_rms * 0.5]]],
        dtype=torch.bfloat16,
    )

    actual = rms_norm_reference(x, weight, eps)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_rms_norm_axes_are_independent_in_the_full_pipeline() -> None:
    config = _small_config(num_query_heads=2, num_kv_heads=1, head_dim=2)
    tensors = _make_tensors(config, tokens=2, seed=29)
    tensors["x"] = torch.tensor(
        [[[1.0, 2.0, 4.0, 8.0], [16.0, 4.0, 2.0, 1.0]]],
        dtype=torch.bfloat16,
    )

    result = _run(config, tensors, return_debug=True)
    assert result.debug is not None
    debug = result.debug

    expected_input = _manual_rms_norm(
        tensors["x"], tensors["input_norm_weight"], config.rms_eps
    )
    assert torch.equal(debug.input_normalized, expected_input)

    q_heads = debug.q_projected.reshape(1, 2, 2, 2)
    k_heads = debug.k_projected.reshape(1, 2, 1, 2)
    expected_q = _manual_rms_norm(q_heads, tensors["q_norm_weight"], config.rms_eps)
    expected_k = _manual_rms_norm(k_heads, tensors["k_norm_weight"], config.rms_eps)
    assert torch.equal(debug.q_normalized, expected_q)
    assert torch.equal(debug.k_normalized, expected_k)


def test_projection_orientation_is_x_times_weight_transpose() -> None:
    x = torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16)
    weight = torch.tensor([[1.0, 3.0], [2.0, 5.0]], dtype=torch.bfloat16)

    actual = linear_reference(x, weight)
    expected = torch.tensor([[[7.0, 12.0]]], dtype=torch.bfloat16)
    wrong_orientation = torch.tensor([[[5.0, 13.0]]], dtype=torch.bfloat16)

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, wrong_orientation)


def test_head_reshape_uses_h_times_d_plus_d_ordering() -> None:
    flat = torch.tensor(
        [[[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]]], dtype=torch.bfloat16
    )
    heads = reshape_heads_reference(flat, num_heads=3, head_dim=2)

    expected = torch.tensor(
        [[[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]]],
        dtype=torch.bfloat16,
    )
    assert heads.shape == (1, 1, 3, 2)
    assert torch.equal(heads, expected)


def test_context_heads_are_flattened_as_inverse_reshape_before_output() -> None:
    config = _small_config(num_query_heads=4, num_kv_heads=2, head_dim=2)
    tensors = _make_tensors(config, tokens=2, seed=37)
    tensors["out_weight"] = torch.eye(
        config.hidden_size, dtype=torch.bfloat16
    )

    result = _run(config, tensors, return_debug=True)
    assert result.debug is not None
    expected = result.debug.context.reshape(1, 2, config.hidden_size)

    assert torch.equal(result.output, expected)


def test_rope_position_zero_is_storage_exact_identity() -> None:
    x = torch.tensor(
        [[[[1.0, -2.0, 3.5, -4.5], [5.0, 6.0, -7.0, 8.0]]]],
        dtype=torch.bfloat16,
    )
    actual = apply_rope_reference(
        x, position_offset=0, rope_theta=10_000.0, store_bf16=True
    )
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, x)


def test_rope_rotates_adjacent_pairs_with_real_valued_frequency() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.bfloat16)
    actual = apply_rope_reference(
        x, position_offset=2, rope_theta=16.0, store_bf16=True
    )

    angle_first = 2.0
    angle_second = 2.0 / math.sqrt(16.0)
    expected = torch.tensor(
        [
            [
                [
                    [
                        math.cos(angle_first) - 2.0 * math.sin(angle_first),
                        math.sin(angle_first) + 2.0 * math.cos(angle_first),
                        3.0 * math.cos(angle_second) - 4.0 * math.sin(angle_second),
                        3.0 * math.sin(angle_second) + 4.0 * math.cos(angle_second),
                    ]
                ]
            ]
        ],
        dtype=torch.bfloat16,
    )
    assert torch.equal(actual, expected)


def test_rope_preserves_each_pair_norm_before_bf16_storage() -> None:
    x = torch.tensor(
        [
            [
                [[1.0, 2.0, 3.0, 4.0], [-2.5, 0.75, 6.0, -1.0]],
                [[0.5, -3.0, 2.0, 7.0], [4.0, 1.0, -5.0, 2.0]],
            ]
        ],
        dtype=torch.bfloat16,
    )
    rotated = apply_rope_reference(
        x, position_offset=7, rope_theta=31.0, store_bf16=False
    )

    original_norm_sq = x.float().reshape(1, 2, 2, 2, 2).square().sum(dim=-1)
    rotated_norm_sq = rotated.reshape(1, 2, 2, 2, 2).square().sum(dim=-1)
    assert rotated.dtype == torch.float32
    torch.testing.assert_close(
        rotated_norm_sq, original_norm_sq, rtol=2.0e-6, atol=2.0e-6
    )


def test_cached_rope_uses_absolute_position_offset() -> None:
    config = _small_config(num_query_heads=2, num_kv_heads=1, head_dim=2)
    tensors = _make_tensors(config, tokens=2, seed=41)
    past_length = 3
    past_k = torch.zeros(
        1, config.num_kv_heads, past_length, config.head_dim, dtype=torch.bfloat16
    )
    past_v = torch.zeros_like(past_k)

    result = _run(
        config, tensors, past_k=past_k, past_v=past_v, return_debug=True
    )
    assert result.debug is not None
    debug = result.debug

    expected_q = _manual_adjacent_rope(
        debug.q_normalized,
        position_offset=past_length,
        rope_theta=config.rope_theta,
    )
    expected_k = _manual_adjacent_rope(
        debug.k_normalized,
        position_offset=past_length,
        rope_theta=config.rope_theta,
    )
    restarted_q = _manual_adjacent_rope(
        debug.q_normalized, position_offset=0, rope_theta=config.rope_theta
    )

    assert torch.equal(debug.q_rope, expected_q)
    assert torch.equal(debug.k_rope, expected_k)
    assert not torch.equal(debug.q_rope, restarted_q)


@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads"),
    [(3, 3), (4, 2), (4, 1)],
    ids=["mha", "two-to-one-gqa", "four-to-one-gqa"],
)
def test_gqa_query_heads_map_to_exact_kv_heads(
    num_query_heads: int, num_kv_heads: int
) -> None:
    head_dim = 2
    q = torch.zeros(1, 1, num_query_heads, head_dim, dtype=torch.bfloat16)
    present_k = torch.zeros(
        1, num_kv_heads, 1, head_dim, dtype=torch.bfloat16
    )
    present_v = torch.empty_like(present_k)
    for kv_head in range(num_kv_heads):
        present_v[0, kv_head, 0] = torch.tensor(
            [10.0 * kv_head + 1.0, 10.0 * kv_head + 2.0],
            dtype=torch.bfloat16,
        )

    scores, probabilities, context = gqa_attention_reference(
        q, present_k, present_v, past_length=0
    )
    group_size = num_query_heads // num_kv_heads
    expected = torch.empty_like(context)
    for query_head in range(num_query_heads):
        kv_head = query_head // group_size
        expected[0, 0, query_head] = present_v[0, kv_head, 0]

    assert scores.dtype == torch.float32
    assert probabilities.dtype == torch.float32
    assert context.dtype == torch.bfloat16
    assert torch.equal(context, expected)
    assert torch.equal(probabilities, torch.ones_like(probabilities))


@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads"),
    [(3, 3), (4, 2), (4, 1)],
    ids=["mha", "two-to-one-gqa", "four-to-one-gqa"],
)
def test_gqa_scores_use_exact_mapped_kv_head(
    num_query_heads: int, num_kv_heads: int
) -> None:
    head_dim = 2
    q = torch.zeros(1, 1, num_query_heads, head_dim, dtype=torch.bfloat16)
    q[..., 0] = 1.0
    present_k = torch.zeros(
        1, num_kv_heads, 2, head_dim, dtype=torch.bfloat16
    )
    present_v = torch.zeros_like(present_k)
    for kv_head in range(num_kv_heads):
        present_k[0, kv_head, 1, 0] = float(kv_head + 1)

    scores, _, _ = gqa_attention_reference(
        q, present_k, present_v, past_length=1
    )
    assert scores is not None
    group_size = num_query_heads // num_kv_heads
    for query_head in range(num_query_heads):
        kv_head = query_head // group_size
        expected = torch.tensor(
            [0.0, (kv_head + 1) / math.sqrt(head_dim)],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            scores[0, query_head, 0], expected, rtol=1.0e-6, atol=1.0e-7
        )


def test_attention_scores_softmax_and_pv_match_scalar_calculation() -> None:
    q = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.bfloat16)
    present_k = torch.tensor(
        [[[[3.0, 4.0], [-1.0, 2.0]]]], dtype=torch.bfloat16
    )
    present_v = torch.tensor(
        [[[[2.0, -1.0], [6.0, 3.0]]]], dtype=torch.bfloat16
    )

    scores, probabilities, context = gqa_attention_reference(
        q, present_k, present_v, past_length=1
    )
    assert scores is not None and probabilities is not None

    first_score = 11.0 / math.sqrt(2.0)
    second_score = 3.0 / math.sqrt(2.0)
    normalizer = math.exp(first_score - first_score) + math.exp(
        second_score - first_score
    )
    first_probability = 1.0 / normalizer
    second_probability = math.exp(second_score - first_score) / normalizer
    expected_scores = torch.tensor(
        [[[[first_score, second_score]]]], dtype=torch.float32
    )
    expected_probabilities = torch.tensor(
        [[[[first_probability, second_probability]]]], dtype=torch.float32
    )
    expected_context = torch.tensor(
        [
            [
                [
                    [
                        2.0 * first_probability + 6.0 * second_probability,
                        -first_probability + 3.0 * second_probability,
                    ]
                ]
            ]
        ],
        dtype=torch.bfloat16,
    )

    torch.testing.assert_close(scores, expected_scores, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(
        probabilities, expected_probabilities, rtol=1.0e-6, atol=1.0e-7
    )
    assert torch.equal(context, expected_context)


def test_causal_visibility_and_past_visibility_are_exact() -> None:
    config = _small_config(num_query_heads=2, num_kv_heads=1, head_dim=2)
    current_tokens = 3
    past_length = 2
    tensors = _zero_tensors(config, tokens=current_tokens)
    past_k = torch.zeros(
        1, config.num_kv_heads, past_length, config.head_dim, dtype=torch.bfloat16
    )
    past_v = torch.zeros_like(past_k)

    result = _run(
        config, tensors, past_k=past_k, past_v=past_v, return_debug=True
    )
    assert result.debug is not None
    scores = result.debug.attention_scores
    probabilities = result.debug.attention_probabilities

    assert scores.shape == (1, config.num_query_heads, current_tokens, 5)
    for token in range(current_tokens):
        visible_count = past_length + token + 1
        visible = probabilities[0, :, token, :visible_count]
        future = probabilities[0, :, token, visible_count:]
        expected_visible = torch.full_like(visible, 1.0 / visible_count)

        torch.testing.assert_close(visible, expected_visible, rtol=1.0e-6, atol=1.0e-7)
        assert torch.equal(future, torch.zeros_like(future))
        assert torch.all(torch.isfinite(scores[0, :, token, :visible_count]))
        assert torch.all(scores[0, :, token, visible_count:] == -torch.inf)
        assert torch.all(probabilities[0, :, token, :past_length] > 0.0)


def test_cache_state_transition_preserves_old_and_places_new_entries() -> None:
    config = _small_config(num_query_heads=2, num_kv_heads=1, head_dim=4)
    batch_size = 1
    tokens = 2
    past_length = 2
    tensors = _make_tensors(config, batch_size=batch_size, tokens=tokens, seed=53)
    past_k = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]],
        dtype=torch.bfloat16,
    )
    past_v = torch.tensor(
        [[[[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
        dtype=torch.bfloat16,
    )
    original_k = past_k.clone()
    original_v = past_v.clone()

    result = _run(
        config, tensors, past_k=past_k, past_v=past_v, return_debug=True
    )
    assert result.debug is not None
    expected_new_k = result.debug.k_rope.permute(0, 2, 1, 3)
    expected_new_v = result.debug.v_projected.reshape(
        batch_size, tokens, config.num_kv_heads, config.head_dim
    ).permute(0, 2, 1, 3)

    assert result.present_k.shape == (1, 1, past_length + tokens, 4)
    assert result.present_v.shape == (1, 1, past_length + tokens, 4)
    assert torch.equal(result.present_k[:, :, :past_length], original_k)
    assert torch.equal(result.present_v[:, :, :past_length], original_v)
    assert torch.equal(result.present_k[:, :, past_length:], expected_new_k)
    assert torch.equal(result.present_v[:, :, past_length:], expected_new_v)
    assert torch.equal(past_k, original_k)
    assert torch.equal(past_v, original_v)


@pytest.mark.parametrize("tokens", [1, 3], ids=["single-token", "multi-token"])
def test_empty_cache_for_single_and_multiple_tokens(tokens: int) -> None:
    config = _small_config()
    tensors = _make_tensors(config, tokens=tokens, seed=61 + tokens)
    result = _run(config, tensors, return_debug=True)

    assert result.output.shape == (1, tokens, config.hidden_size)
    assert result.present_k.shape == (
        1,
        config.num_kv_heads,
        tokens,
        config.head_dim,
    )
    assert result.present_v.shape == result.present_k.shape
    assert result.debug is not None
    for token in range(tokens):
        assert torch.equal(
            result.debug.attention_probabilities[0, :, token, token + 1 :],
            torch.zeros_like(
                result.debug.attention_probabilities[0, :, token, token + 1 :]
            ),
        )


def test_one_token_cached_decode_matches_full_sequence() -> None:
    config = _small_config(num_query_heads=4, num_kv_heads=1, head_dim=2)
    sequence_length = 5
    tensors = _make_tensors(config, tokens=sequence_length, seed=73)
    prefix_length = sequence_length - 1

    full = _run(config, tensors)
    prefix_tensors = dict(tensors)
    prefix_tensors["x"] = tensors["x"][:, :prefix_length]
    prefix = _run(config, prefix_tensors)
    decode_tensors = dict(tensors)
    decode_tensors["x"] = tensors["x"][:, prefix_length:]
    decoded = _run(
        config,
        decode_tensors,
        past_k=prefix.present_k,
        past_v=prefix.present_v,
    )

    assert torch.equal(decoded.output, full.output[:, prefix_length:])
    assert torch.equal(decoded.present_k, full.present_k)
    assert torch.equal(decoded.present_v, full.present_v)


def test_multi_token_chunked_decode_matches_full_sequence() -> None:
    config = _small_config(num_query_heads=4, num_kv_heads=2, head_dim=4)
    sequence_length = 6
    prefix_length = 2
    tensors = _make_tensors(config, tokens=sequence_length, seed=89)

    full = _run(config, tensors)
    prefix_tensors = dict(tensors)
    prefix_tensors["x"] = tensors["x"][:, :prefix_length]
    prefix = _run(config, prefix_tensors)
    chunk_tensors = dict(tensors)
    chunk_tensors["x"] = tensors["x"][:, prefix_length:]
    chunked = _run(
        config,
        chunk_tensors,
        past_k=prefix.present_k,
        past_v=prefix.present_v,
    )

    assert torch.equal(chunked.output, full.output[:, prefix_length:])
    assert torch.equal(chunked.present_k, full.present_k)
    assert torch.equal(chunked.present_v, full.present_v)


def test_dtype_boundaries_and_optional_debug_contract() -> None:
    config = _small_config()
    tensors = _make_tensors(config, tokens=2, seed=97)

    without_debug = _run(config, tensors, return_debug=False)
    assert without_debug.debug is None
    assert without_debug.output.dtype == torch.bfloat16
    assert without_debug.present_k.dtype == torch.bfloat16
    assert without_debug.present_v.dtype == torch.bfloat16

    result = _run(config, tensors, return_debug=True)
    assert result.debug is not None
    for tensor in (
        result.debug.input_normalized,
        result.debug.q_projected,
        result.debug.k_projected,
        result.debug.v_projected,
        result.debug.q_normalized,
        result.debug.k_normalized,
        result.debug.q_rope,
        result.debug.k_rope,
        result.debug.context,
    ):
        assert tensor.dtype == torch.bfloat16
    assert result.debug.attention_scores.dtype == torch.float32
    assert result.debug.attention_probabilities.dtype == torch.float32


@pytest.mark.parametrize(
    ("name", "bad_shape"),
    [
        ("x", (1, 2, 9)),
        ("x", (1, 8)),
        ("input_norm_weight", (9,)),
        ("q_weight", (8, 9)),
        ("k_weight", (5, 8)),
        ("v_weight", (4, 9)),
        ("q_norm_weight", (3,)),
        ("k_norm_weight", (1,)),
        ("out_weight", (9, 8)),
    ],
)
def test_malformed_inputs_and_weights_are_rejected(
    name: str, bad_shape: tuple[int, ...]
) -> None:
    config = _small_config()
    tensors = _make_tensors(config, tokens=2)
    tensors[name] = torch.zeros(bad_shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _run(config, tensors)


@pytest.mark.parametrize("shape", [(0, 1, 8), (1, 0, 8)])
def test_empty_batch_or_token_dimension_is_rejected(shape: tuple[int, ...]) -> None:
    config = _small_config()
    tensors = _make_tensors(config, tokens=1)
    tensors["x"] = torch.zeros(shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _run(config, tensors)


@pytest.mark.parametrize(
    "name",
    [
        "x",
        "input_norm_weight",
        "q_weight",
        "k_weight",
        "v_weight",
        "q_norm_weight",
        "k_norm_weight",
        "out_weight",
    ],
)
def test_non_bf16_semantic_tensors_are_rejected(name: str) -> None:
    config = _small_config()
    tensors = _make_tensors(config, tokens=1)
    tensors[name] = tensors[name].float()
    with pytest.raises((TypeError, ValueError)):
        _run(config, tensors)


@pytest.mark.parametrize(
    ("past_k_shape", "past_v_shape"),
    [
        ((2, 2, 2, 2), (2, 2, 2, 2)),
        ((1, 1, 2, 2), (1, 1, 2, 2)),
        ((1, 2, 2, 3), (1, 2, 2, 3)),
        ((1, 2, 3, 2), (1, 2, 2, 2)),
        ((1, 2, 2), (1, 2, 2)),
    ],
    ids=["batch", "kv-heads", "head-dimension", "cache-length", "rank"],
)
def test_malformed_cache_shapes_are_rejected(
    past_k_shape: tuple[int, ...], past_v_shape: tuple[int, ...]
) -> None:
    config = _small_config()
    tensors = _make_tensors(config, tokens=1)
    past_k = torch.zeros(past_k_shape, dtype=torch.bfloat16)
    past_v = torch.zeros(past_v_shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _run(config, tensors, past_k=past_k, past_v=past_v)


@pytest.mark.parametrize("missing", ["past_k", "past_v"])
def test_cache_tensors_must_be_supplied_as_a_pair(missing: str) -> None:
    config = _small_config()
    tensors = _make_tensors(config, tokens=1)
    cache = torch.zeros(1, 2, 2, 2, dtype=torch.bfloat16)
    past_k = None if missing == "past_k" else cache
    past_v = None if missing == "past_v" else cache
    with pytest.raises(ValueError):
        _run(config, tensors, past_k=past_k, past_v=past_v)


def test_non_bf16_cache_is_rejected() -> None:
    config = _small_config()
    tensors = _make_tensors(config, tokens=1)
    cache = torch.zeros(1, 2, 2, 2, dtype=torch.float32)
    with pytest.raises((TypeError, ValueError)):
        _run(config, tensors, past_k=cache, past_v=cache.clone())


def test_cache_on_incompatible_device_is_rejected() -> None:
    config = _small_config()
    tensors = _make_tensors(config, tokens=1)
    past_k = torch.empty(
        1, 2, 2, 2, dtype=torch.bfloat16, device=torch.device("meta")
    )
    past_v = torch.empty_like(past_k)
    with pytest.raises(ValueError):
        _run(config, tensors, past_k=past_k, past_v=past_v)


def test_finite_normal_range_inputs_produce_only_finite_outputs_and_caches() -> None:
    config = _small_config(num_query_heads=4, num_kv_heads=1, head_dim=4)
    tensors = _make_tensors(config, batch_size=2, tokens=4, seed=101)
    result = _run(config, tensors)

    assert torch.isfinite(result.output).all()
    assert torch.isfinite(result.present_k).all()
    assert torch.isfinite(result.present_v).all()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="requires a CUDA device with PyTorch BF16 support",
)
def test_reference_runs_on_cuda_with_the_same_semantic_boundaries() -> None:
    config = _small_config(num_query_heads=4, num_kv_heads=1, head_dim=4)
    cpu_tensors = _make_tensors(config, batch_size=2, tokens=3, seed=113)
    cpu_result = _run(config, cpu_tensors)

    cuda_tensors = {name: tensor.to("cuda") for name, tensor in cpu_tensors.items()}
    cuda_result = _run(config, cuda_tensors)
    torch.cuda.synchronize()

    for tensor in (
        cuda_result.output,
        cuda_result.present_k,
        cuda_result.present_v,
    ):
        assert tensor.device.type == "cuda"
        assert tensor.dtype == torch.bfloat16

    # FP32 reduction trees and trigonometric implementations may differ by
    # device; one BF16 epsilon is a cross-device storage-level comparison.
    for cuda_tensor, cpu_tensor in (
        (cuda_result.output, cpu_result.output),
        (cuda_result.present_k, cpu_result.present_k),
        (cuda_result.present_v, cpu_result.present_v),
    ):
        torch.testing.assert_close(
            cuda_tensor.cpu(),
            cpu_tensor,
            rtol=BF16_EPSILON,
            atol=BF16_EPSILON,
        )
