# Milestone 5C — Capacity-Aware GQA and Modular Decoder Attention

Milestone 5C is a correctness and integration milestone. It adds a separate
capacity-aware causal GQA primitive and composes the previously validated CUDA
primitives into the first complete modular `nvfp4_w4a16` decoder-attention
forward path. It does not add an attention optimization, fusion, or benchmark.

## Logical context versus physical capacity

The persistent caches have physical layout:

```text
k_cache: BF16 [B,Hkv,C,D]
v_cache: BF16 [B,Hkv,C,D]
```

For query storage `q: BF16 [B,T,Hq,D]` and `past_length=P`, the operator derives
the logical context length:

```text
S = P + T
```

It requires `S <= C`. Only logical positions `0 <= j < S` participate in
attention. Slots `S:C` are unused capacity and are never logically attended.
The distinction is fundamental: `S` limits valid positions and sizes temporary
attention state, while `C` remains the physical cache batch/head stride.

The exact K address is:

```text
((b * Hkv + kv_head) * C + j) * D + d
```

The exact V address is identical in its physical dimensions:

```text
((b * Hkv + kv_head) * C + j) * D + d
```

Neither address substitutes `S` for `C`. Tests use `C >> S`, multiple batches,
and multiple KV heads so that such a substitution reads obvious sentinels or a
different batch/head.

## Two explicit GQA APIs

M5A remains unchanged:

```python
cuda_gqa_attention(q, present_k, present_v, past_length)
```

It consumes compact contiguous logical caches `[B,Hkv,S,D]` and requires
`S=P+T`.

M5C adds:

```python
cuda_gqa_attention_cached(q, k_cache, v_cache, past_length)
```

Its raw operator schema is:

```text
cuda_nvfp4_decoder_attention::cuda_gqa_attention_cached(
    Tensor q,
    Tensor k_cache,
    Tensor v_cache,
    int past_length
) -> Tensor
```

It reads capacity-backed caches `[B,Hkv,C,D]` directly and returns contiguous
BF16 context `[B,T,Hq,D]`. It does not mutate either cache and does not route to
the compact M5A operator.

The implementation retains M5A's transparent three stages:

1. FP32 QK scores with `j <= P+i` causality;
2. FP32 materialized softmax using ordinary `expf`, masked `-inf`, no epsilon,
   and no clamp;
3. FP32 PV accumulation with BF16 context storage.

Scores and probabilities have logical shape `[B,Hq,T,S]`. They are the only
large attention temporaries. No compact K/V prefix, cache clone, concatenation,
or prefix copy exists in the production cached-attention path.

## Complete modular CUDA forward path

The public mutating Python API is:

```python
cuda_decoder_attention_forward_(
    x,
    input_norm_weight,
    q_weight,
    k_weight,
    v_weight,
    q_norm_weight,
    k_norm_weight,
    out_weight,
    k_cache,
    v_cache,
    past_length,
    *,
    rms_eps=1e-6,
    rope_theta=10000.0,
) -> output
```

The trailing underscore denotes in-place cache mutation. The output is BF16
`[B,T,H]`; the supplied cache objects and storage pointers are retained.

Dimensions are inferred rather than hard-coded:

```text
B,T,H = x.shape
Hkv,C,D = k_cache.shape[1:]
Hq = H / D
```

The composition requires `H % D == 0`, `Hq % Hkv == 0`, and weight logical
shapes `Wq=[H,H]`, `Wk=[Hkv*D,H]`, `Wv=[Hkv*D,H]`, and `Wo=[H,H]`.

The exact stage order is:

1. input RMSNorm;
2. baseline W4A16 Q projection;
3. baseline W4A16 K projection;
4. baseline W4A16 V projection;
5. metadata-only Q/K/V head reshapes;
6. per-head Q RMSNorm over `D`;
7. per-head K RMSNorm over `D`;
8. Q RoPE starting at position `P`;
9. K RoPE starting at position `P`;
10. in-place append of finalized K and projected BF16 V;
11. capacity-aware causal GQA directly over the physical caches;
12. metadata-only context flattening with flat index `h*D+d`;
13. baseline W4A16 output projection.

V receives neither per-head RMSNorm nor RoPE. There is no residual addition.
All four projections call `cuda_w4a16_linear`. The retained experimental
`cuda_w4a16_linear_grouped_decode` operator is not selected by this path.

Projection and context outputs are required to be contiguous, so flat/head
transitions use `reshape` only. No layout-fixing copy or head-transpose kernel
is inserted.

## Cache transition

The CUDA path appends:

```text
K: post-K-RMSNorm, post-RoPE BF16
V: projected BF16, without normalization or RoPE
```

Current token `i` is stored at physical slot `P+i`. Attention launches only
after all current-chunk K/V slots are physically available, but its causal
condition remains exactly `j <= P+i`; future current-chunk slots therefore
remain logically invisible. The prefix `0:P` and unused suffix `P+T:C` are
preserved, and cache data pointers do not change.

The independent reference oracle uses compact `torch.cat` semantics, as the
architecture permits for a mathematical reference. The production CUDA module
does not concatenate or compact cache storage.

## Correctness evidence

The capacity-aware targeted suite covers:

- separate hand-computable K- and V-stride oracles with `C=7`, `S=2`, and two
  KV heads;
- `B=2`, `T>1`, distinct batch/head/position values, and `C >> S`;
- `P=0`, `P=3`, `P=128`, and `P=2048`;
- ratios 1:1, 2:1, 4:1, and canonical 24:6;
- `D=8`, `D=32`, and `D=128`;
- `C=8192` with both small `S` and long logical context;
- bit-exact equivalence to the frozen compact M5A CUDA result; and
- a real non-default stream with a downstream context consumer.

The end-to-end suite covers a stored-stage diagnostic case, `B=2` batch
isolation, `T>1` causal hiding of already appended future slots, `P=0`, cache
prefix/reference equality, suffix preservation, pointer identity, four-call
baseline projection proof, and a non-default stream with output and cache
consumers. The canonical `B=1,T=1,H=3072,Hq=24,Hkv=6,D=128,P=128,C=512` run
has final-output metrics:

```text
maximum absolute error:          0
mean absolute error:             0
exact BF16 fraction:             1
maximum BF16 adjacency distance: 0
```

Compute Sanitizer memcheck reports `ERROR SUMMARY: 0 errors` for both the
capacity-aware validator and the full modular-pipeline validator. The latter
executes input RMSNorm, Q/K/V projections, Q/K RMSNorm, Q/K RoPE, cache append,
cached QK/softmax/PV, and the output projection, including a reduced
`B=2,T=3,P=2,C=17` reference-checked case and a canonical-shape `T=1` smoke
case.

Both required regression entry points completed with the same result:

```text
.venv/bin/pytest -q:            461 passed, 14 skipped
.venv/bin/python -m pytest -q:  461 passed, 14 skipped
```

The frozen M5A compact-GQA and M5B cache-append validators were also rerun
under memcheck and each reported `ERROR SUMMARY: 0 errors`.

## Limitations

- This is a modular multi-kernel path, not a fused operator.
- Attention still materializes FP32 scores and probabilities.
- It is not FlashAttention and does not use Tensor Core attention.
- It is not performance-characterized in M5C.
- It uses software-decoded portable NVFP4 weights on Ada, not native FP4
  Tensor Core execution.
- Grouped-decode W4A16 remains a separate experimental primitive and is not
  silently selected.
- KV cache remains BF16, fixed-capacity, unpaged, and without eviction or a
  sliding window.
- There is no residual, FFN, dropout, model loading, or tokenization.
