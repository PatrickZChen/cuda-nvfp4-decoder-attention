# Architecture Specification

Status: Milestone 0 — specification only.

This document fixes the public semantic contract for the decoder-attention block. It deliberately does not choose CUDA kernels, launch geometry, a fused design, or a hardware-specific physical tensor layout.

## 1. Project goal and clean-room boundary

This repository was independently designed and implemented as a clean-room CUDA performance-engineering project. Its goal is to study one transformer decoder-attention block, with particular attention to incremental decoding, low-precision projection weights, memory traffic, intermediate elimination, profiling, and evidence-based kernel fusion.

Technical decisions may be informed by public documentation and public research. Private repositories, prompts, task descriptions, tests, benchmark harnesses, tensor dumps, constants, artifacts, and other non-public implementation sources are outside the design boundary and must not be consulted or incorporated. Future code, test vectors, benchmarks, and measurements must be created independently for this repository.

Milestone 0 contains no implementation and makes no performance claim. The project is a focused performance study, not a transformer framework.

## 2. Hardware and execution scope

The primary development target is:

| Component | Primary target |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU |
| Architecture | Ada, SM89 |
| VRAM | 12 GB |
| Host environment | Windows with WSL2 |
| Distribution | Ubuntu 24.04 |
| CUDA Toolkit | 12.5 |
| Python | 3.12 |
| Framework for the planned reference | PyTorch with CUDA |

Incremental autoregressive decoding with `T = 1` is the primary performance target. Prefill is secondary and is included for correctness, pipeline completeness, comparison, and later characterization.

Ada/SM89 does not provide native NVFP4 Tensor Core matrix execution. The planned Ada low-precision path therefore uses packed NVFP4 projection weights with software unpack/dequantization for W4A16 projection execution. It must not be described as native FP4 Tensor Core GEMM or Blackwell-style native NVFP4 execution. Native Blackwell FP4 work is optional future scope and would be a separate backend with its own layout and validation requirements.

In this repository, an NVFP4 representation and a native FP4 execution mechanism are separate concepts.

## 3. Scope and explicit non-goals

The block begins with BF16 hidden states and ends with a BF16 decoder-attention output. It includes input RMSNorm, Q/K/V projections, per-head Q and K RMSNorm, adjacent-pair RoPE, BF16 KV caching, causal grouped-query attention, and the output projection.

It excludes projection bias, normalization bias, dropout, residual addition, an FFN, training, and a backward pass. It has no arbitrary attention-mask or padding-mask input.

This project is not intended to implement:

- a full transformer or a full LLM;
- model loading or tokenization;
- training or backward propagation;
- distributed execution;
- a generic GEMM library or generic tensor framework;
- a model server or REST API;
- a FlashAttention clone as a project requirement;
- arbitrary attention masking or padding-mask APIs;
- arbitrary RoPE variants;
- a quantized KV cache in the initial scope;
- a cache allocator or model-runtime framework; or
- native Blackwell NVFP4 execution on Ada.

## 4. Notation, invariants, and canonical configuration

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `T` | number of newly processed tokens |
| `P` | past context length |
| `H` | hidden size |
| `Hq` | number of query heads |
| `Hkv` | number of key/value heads |
| `D` | head dimension |
| `N`, `K` | logical output and reduction dimensions of a weight matrix |

The base invariants are `B >= 1`, `T >= 1`, `P >= 0`, `H = Hq * D`, `Hq % Hkv == 0`, and even `D` for adjacent-pair RoPE.

The independently chosen canonical public configuration is:

| Quantity | Value |
|---|---:|
| `H` | 3072 |
| `Hq` | 24 |
| `Hkv` | 6 |
| `D` | 128 |
| `Hq / Hkv` | 4 |
| Q projection width | 3072 |
| K projection width | 768 |
| V projection width | 768 |

Thus, `H = Hq * D = 24 * 128 = 3072`. A later reference may accept smaller shapes satisfying the same invariants. The first optimized CUDA implementation may specialize for `D = 128`; Milestone 0 does not promise arbitrary head dimensions or support for every planned benchmark point.

## 5. End-to-end semantic interface

### 5.1 Inputs and parameters

The current hidden states are:

```text
x: BF16 [B, T, H]
```

The primary workload has `T = 1`. Secondary prefill studies may later use `T = 64`, `T = 256`, and `T = 1024`.

The common parameters are:

| Parameter | Dtype and semantic shape |
|---|---|
| `input_norm_weight` | BF16 `[H]` |
| `q_norm_weight` | BF16 `[D]` |
| `k_norm_weight` | BF16 `[D]` |
| `Wq` | `[H, H]` projection weight |
| `Wk` | `[Hkv * D, H]` projection weight |
| `Wv` | `[Hkv * D, H]` projection weight |
| `Wo` | `[H, H]` projection weight |
| `past_k` | BF16 `[B, Hkv, P, D]` |
| `past_v` | BF16 `[B, Hkv, P, D]` |

Projection-weight storage depends on the execution mode: BF16 tensors in `bf16`, or the packed logical representation defined in Section 8 for `nvfp4_w4a16`. The mathematical projection shapes do not change.

`P` is inferred from the common cache length. Both cache tensors must have the same `B`, `Hkv`, `P`, and `D`; all batch elements share that `P`. An empty past is represented by `P = 0`. Variable-length padded batches are outside the initial semantic API.

There are no externally supplied cosine/sine tensors, position tensors, attention masks, padding masks, or projection biases. Absolute positions follow solely from `P` and the current-token index.

### 5.2 Outputs

The semantic operation produces:

| Output | Dtype and semantic shape |
|---|---|
| decoder-attention output | BF16 `[B, T, H]` |
| `present_k` | BF16 `[B, Hkv, P + T, D]` |
| `present_v` | BF16 `[B, Hkv, P + T, D]` |

The present caches describe a state transition, not a required allocation strategy. A mathematical reference may construct them by concatenation. A future CUDA path should write new entries into a preallocated cache.

### 5.3 Pipeline

```text
BF16 hidden states
        ↓
input RMSNorm
        ↓
Q / K / V projections
        ↓
reshape into heads
        ↓
per-head Q RMSNorm
        ↓
per-head K RMSNorm
        ↓
RoPE on Q and K
        ↓
BF16 KV cache
        ↓
causal grouped-query attention
        ↓
concatenate query heads
        ↓
output projection
        ↓
BF16 decoder-attention output
```

All shapes in this document are semantic indexing contracts. Later CUDA implementations may use different transposed, tiled, vectorized, or fused physical layouts if they preserve these semantics and profiling justifies the change.

## 6. Precision boundaries

“FP32 accumulation” is an arithmetic contract, not a promise of one reduction tree or bitwise identity across implementations. Floating-point outputs are judged by the stage-specific policy in Section 9.

| Stage | Conceptual arithmetic | Stored output |
|---|---|---|
| Input RMSNorm | FP32 square, reduction, reciprocal square root, and scaling | BF16 |
| All projections in `bf16` | BF16 inputs and weights, FP32 accumulation | BF16 |
| All projections in `nvfp4_w4a16` | BF16 inputs, software-reconstructed weights, FP32 accumulation | BF16 |
| Per-head Q/K RMSNorm | FP32 square, reduction, reciprocal square root, and scaling | BF16 |
| RoPE | FP32 angle, trigonometric, and rotation arithmetic | BF16 Q and K |
| KV cache | no additional quantization | BF16 |
| QK scores | FP32 accumulation and scaling | FP32 |
| Causal softmax | FP32 after masking | FP32 probabilities |
| PV context | FP32 accumulation | BF16 |

In `nvfp4_w4a16`, projection inputs remain BF16, packed weights are logically reconstructed by software unpack/dequantization, products accumulate in FP32, and projection outputs are stored in BF16. The contract does not require a fully materialized dequantized-weight intermediate.

## 7. Decoder-attention semantics

### 7.1 Input RMSNorm

Input RMSNorm operates independently on every `[H]` vector. Let `gamma = input_norm_weight` and `eps = 1e-6`. For every batch index `b`, current-token index `t`, and hidden index `i`:

```text
rms_inv[b,t] = rsqrt((1 / H) * sum_j FP32(x[b,t,j])^2 + eps)

y[b,t,i] = BF16(
    FP32(x[b,t,i])
    * rms_inv[b,t]
    * FP32(gamma[i])
)
```

The reduction and normalization arithmetic are conceptually FP32. `y` is stored as BF16, and that BF16 value is the input to all three projections. There is no additive normalization bias.

### 7.2 Projection operation and weights

Every projection uses the logical operation:

```text
Y = X @ W^T
```

Leading batch/token dimensions are conceptually flattened into rows; the final input dimension is the reduction dimension. For the BF16 baseline, row `r` and output element `n` are defined by:

```text
Y[r,n] = BF16(
    sum_k FP32(X[r,k]) * FP32(W[n,k])
)
```

The FP32 reduction order is not frozen. In `nvfp4_w4a16`, the same equation uses the reconstructed `W_hat[n,k]` from Section 8.2 in place of the BF16 `W[n,k]`. There is no projection bias.

| Weight | General shape | Canonical shape |
|---|---|---|
| `Wq` | `[H, H]` | `[3072, 3072]` |
| `Wk` | `[Hkv * D, H]` | `[768, 3072]` |
| `Wv` | `[Hkv * D, H]` | `[768, 3072]` |
| `Wo` | `[H, H]` | `[3072, 3072]` |

For the initial BF16 semantic baseline, inputs and weights are BF16, each dot product accumulates conceptually in FP32, and each projection result is stored in BF16. The resulting flat tensors are:

```text
q_flat: BF16 [B, T, Hq * D]
k_flat: BF16 [B, T, Hkv * D]
v_flat: BF16 [B, T, Hkv * D]
```

### 7.3 Head reshaping

The flat-to-head mapping is explicit:

```text
Q[b,t,h,d] = q_flat[b,t,h * D + d]
K[b,t,h,d] = k_flat[b,t,h * D + d]
V[b,t,h,d] = v_flat[b,t,h * D + d]
```

This gives the semantic layouts:

```text
Q: BF16 [B, T, Hq,  D]
K: BF16 [B, T, Hkv, D]
V: BF16 [B, T, Hkv, D]
```

This ordering is also used when context heads are flattened before the output projection. It does not require a CUDA implementation to materialize these exact views.

### 7.4 Per-head Q and K RMSNorm

Q and K are normalized independently over their final `D` elements for every `(b, t, h)`. Let `eps = 1e-6`. For Q:

```text
q_rms_inv[b,t,h] = rsqrt(
    (1 / D) * sum_j FP32(Q[b,t,h,j])^2 + eps
)

Q_norm[b,t,h,d] = BF16(
    FP32(Q[b,t,h,d])
    * q_rms_inv[b,t,h]
    * FP32(q_norm_weight[d])
)
```

K follows the same equations with `K`, `k_norm_weight`, and `Hkv`. The single BF16 vector `q_norm_weight[D]` is shared across all query heads, and the single BF16 vector `k_norm_weight[D]` is shared across all KV heads. Reductions and scaling are conceptually FP32; normalized results are stored in BF16. V is not RMS-normalized.

### 7.5 Adjacent-pair RoPE and absolute positions

RoPE applies to `Q_norm` and `K_norm`, not to V. `D` must be even. For token `i` in the current chunk:

```text
absolute_position p = P + i
rope_theta = 10000.0
```

For each adjacent pair `(2m, 2m + 1)`, where `0 <= m < D / 2`:

```text
theta_m(p) = p / rope_theta^(2m / D)

out_even = x_even * cos(theta_m(p)) - x_odd * sin(theta_m(p))
out_odd  = x_even * sin(theta_m(p)) + x_odd * cos(theta_m(p))
```

The quotient `2m / D` in the exponent is real-valued, not integer division. The BF16 normalized input is promoted for conceptually FP32 angle, trigonometric, and rotation arithmetic; the rotated outputs, denoted `Q_rope` and `K_rope`, are stored in BF16. At `p = 0`, the required result is exactly the unrotated input: position zero is an identity transform.

Cosine and sine values are derived internally from absolute positions. A later implementation may precompute or cache tables internally, but externally supplied trigonometric tensors are not part of the semantic API.

### 7.6 KV-cache state transition

The past cache layout is:

```text
past_k: BF16 [B, Hkv, P, D]
past_v: BF16 [B, Hkv, P, D]
```

The new entries, after converting from token-major head layouts, are:

```text
new_k[b,h,i,d] = RoPE(K_norm)[b,i,h,d]
new_v[b,h,i,d] = V[b,i,h,d]
```

Therefore cached K is post-K-RMSNorm and post-RoPE. Cached V is the projected BF16 V and has neither per-head RMSNorm nor RoPE. The supplied `past_k` and `past_v` must already satisfy these same respective storage contracts.

The mathematical state transition is concatenation along the context axis:

```text
present_k = concatenate(past_k, new_k, axis=context)
present_v = concatenate(past_v, new_v, axis=context)
```

The results have shape `[B, Hkv, P + T, D]`. Existing slots `0:P` remain unchanged, and current token `i` occupies absolute slot `P + i`. The K and V values visible to attention are exactly the corresponding logical present-cache values.

Concatenation defines reference semantics only. The future CUDA implementation should use preallocated storage and write slots `P:P+T`; it must not repeatedly copy the full past cache during autoregressive decode. Allocation policy and model-runtime ownership remain out of scope.

### 7.7 Grouped-query head mapping

Require:

```text
Hq % Hkv == 0
group_size = Hq / Hkv
kv_head(query_head) = query_head // group_size
```

For the canonical 4:1 configuration:

| Query heads | KV head |
|---|---:|
| `0..3` | 0 |
| `4..7` | 1 |
| `8..11` | 2 |
| `12..15` | 3 |
| `16..19` | 4 |
| `20..23` | 5 |

The canonical algorithm indexes the mapped KV head directly. It is not defined by physically duplicating the full K or V tensor. A future independent secondary oracle may repeat K/V explicitly, but such repetition does not change the canonical algorithm.

### 7.8 Causal attention

For current-token index `i` in `[0, T)`, the query absolute position is `P + i`. Keys have logical absolute positions `j` in `[0, P + T)`. Visibility is exactly:

```text
j <= P + i
```

Thus every past key is visible, the current key is visible, earlier keys within the current chunk are visible, and later keys within the current chunk are masked.

For batch `b`, query head `h`, and mapped KV head `g = h // group_size`, the unmasked score is:

```text
score[b,h,i,j] =
    (sum_d FP32(Q_rope[b,i,h,d]) * FP32(present_k[b,g,j,d]))
    / sqrt(D)
```

The QK dot product and scale are conceptually FP32. Invisible scores are replaced by negative infinity before softmax. Softmax is evaluated in FP32 over the context positions after masking, with masked positions contributing zero probability. A numerically stable evaluation is required; a particular reduction algorithm is not frozen in Milestone 0.

Equivalently, for the visible set `J_i = {j | 0 <= j < P + T and j <= P + i}`:

```text
probability[b,h,i,j] =
    exp(score[b,h,i,j]) / sum_r_in_J_i exp(score[b,h,i,r])  if j in J_i
    0                                                        otherwise
```

This equation defines the mathematical result; the implementation must use a stable FP32 softmax rather than evaluate the exponential ratio naively.

Context is:

```text
context_fp32[b,i,h,d] =
    sum_j probability[b,h,i,j] * FP32(present_v[b,g,j,d])

context[b,i,h,d] = BF16(context_fp32[b,i,h,d])
```

PV accumulation is conceptually FP32. The stored attention context has semantic shape `BF16 [B, T, Hq, D]`.

### 7.9 Context flattening and output projection

Query heads are concatenated using the inverse of the mapping in Section 7.3:

```text
context_flat[b,t,h * D + d] = context[b,t,h,d]
```

This gives `context_flat: BF16 [B, T, H]`. The final projection is:

```text
output = context_flat @ Wo^T
```

The baseline uses BF16 operands, FP32 accumulation, and BF16 output storage. The final result is `BF16 [B, T, H]`. Decoder-attention responsibility ends at this boundary; there is no residual addition.

## 8. NVFP4 representation and planned projection modes

### 8.1 Meaning of NVFP4 in this repository

NVFP4 denotes a planned numerical and packed-weight contract composed of:

- FP4 E2M1 element values;
- microscaling blocks of 16 contiguous elements;
- one positive eight-bit floating block scale per microscaling block, described in public material with E4M3/UE4M3 terminology; and
- one FP32 tensor-level decode scale per weight tensor.

This project intentionally chooses the publicly supported 1D weight-scaling variant: each block scale applies to 16 consecutive weight elements along the logical `K` dimension. NVIDIA Transformer Engine may use 2D `16 x 16` scaling for weights by default; its 1D weight scaling is publicly supported when 2D quantization is disabled. The 2D, training-oriented weight recipe is outside this project's initial scope. This choice does not imply that 1D scaling is the only NVFP4 weight-scaling scheme.

The finite E2M1 magnitude set used by the planned logical reference is:

```text
0, 0.5, 1, 1.5, 2, 3, 4, 6
```

Nonzero magnitudes have corresponding signed values. Round-to-nearest-even is the intended deterministic reference value-selection convention. Exact code points and edge behavior remain subject to the public-source validation items in Section 8.4.

NVFP4 does not imply that all attention computation is FP4. The initial low-precision focus is the four projection weights and their projection execution. RMSNorm, RoPE, the KV cache, attention scores, softmax, and context retain the BF16/FP32 boundaries defined above.

Most importantly, the Ada path uses **software-decoded NVFP4 weights on Ada**. It is not native NVFP4 Tensor Core execution.

### 8.2 Planned mathematical scaling contract

For a logical projection weight matrix:

```text
W: [N, K]
K % 16 == 0
```

Blocks never cross rows. Block `(n, b)` contains exactly:

```text
W[n, 16*b : 16*b + 16]
```

There is one logical E2M1 value for each element, one positive logical block scale for each `(n, b)`, and one FP32 global decode scale for the entire matrix. Reconstruction is:

```text
W_hat[n,k] =
    fp4_value[n,k]
    * block_scale[n, k // 16]
    * global_scale
```

The global scale supplies the tensor-wide decode factor. The block scale supplies local dynamic-range adjustment for one row-local 16-element block. The E2M1 value supplies the signed low-precision element value.

Zero tensors must reconstruct exactly and use a deterministic canonical encoding. Tensor-level zero-amax handling must be consistent with validated public NVIDIA semantics. The exact logical zero-block scale value and its encoded byte are deliberately not frozen in Milestone 0; Milestone 2 public-source validation will determine them. This preserves the requirement not to invent undocumented bit-level behavior.

For nonzero tensors, the scale-selection algorithm and all saturation, underflow, exceptional-value, and tie details must be finalized against authoritative public NVIDIA documentation during Milestone 2. Milestone 0 does not invent those rules.

### 8.3 Planned portable packed logical layout

For `W [N, K]` with `K % 16 == 0`, the repository layout is planned as:

```text
values:       uint8 [N, K/2]
scales:       uint8 [N, K/16]
global_scale: FP32 scalar
```

For byte-column index `r`, where `0 <= r < K / 2`:

```text
values[n, r]:
    low nibble  = code for W[n, 2*r]
    high nibble = code for W[n, 2*r + 1]

scales[n, b]:
    byte for the positive scale of W[n, 16*b : 16*b + 16]
```

Nibble order, row-local block membership, array shapes, and the reconstruction direction are repository contracts. The exact E2M1 code-to-value map and exact scale-byte interpretation are Milestone 2 validation items.

This is a portable logical layout for the Ada software-decoded implementation. It is not claimed to match a cuBLAS native Blackwell FP4 layout, a CUTLASS Blackwell tiled scale layout, or any other hardware-native FP4 layout. An optional future Blackwell backend may require explicit layout conversion.

### 8.4 Public-source validation required in Milestone 2

Before quantization or packing is implemented, Milestone 2 must verify and document from authoritative public NVIDIA sources:

- the exact four-bit E2M1 code-to-value assignments, including zero, signed zero, and any special or reserved encodings;
- whether E4M3 or UE4M3 is the authoritative term and representation for each positive NVFP4 block scale—the terms are not silently treated as interchangeable here;
- the scale byte's representable set, exponent bias, exceptional or reserved encodings, and conversion rules;
- tensor-scale and block-scale selection and rounding;
- tensor-level zero-amax handling, the canonical zero-block scale value, and its encoded byte;
- ties, subnormals, overflow, underflow, saturation, NaN, and infinity handling;
- the interaction between value rounding and scale rounding; and
- any physical-layout requirements for an optional native Blackwell backend.

The repository's nibble order and row-local block grouping remain deliberate portable-layout choices even if a future native backend needs a different arrangement.

### 8.5 Conceptual execution modes

#### `bf16`

```text
BF16 projection weights
BF16 activations
FP32 accumulation
BF16 projection outputs
```

This is the higher-precision semantic baseline.

#### `nvfp4_w4a16`

```text
NVFP4-packed projection weights
BF16 activations
software unpack / dequantization on Ada
FP32 accumulation
BF16 projection outputs
```

This is the primary Ada low-precision performance path. In this named mode, `Wq`, `Wk`, `Wv`, and `Wo` all use the packed representation. Any experimental mode packing only a subset must be separately named and reported. The mode fixes mathematical reconstruction and precision boundaries, but does not require decoded weights to be materialized or prescribe a kernel organization.

#### `nvfp4_w4a4_reference`

This optional future numerical experiment uses logical NVFP4 activations and weights with mathematically explicit dequantized execution. It is not initially an optimized Ada path and does not imply native FP4 execution. Activation block partitioning, scaling, packing, and edge behavior are not frozen in Milestone 0 and must be specified before this optional mode is implemented.

A true native Blackwell FP4 backend is optional future work only. It would be named, laid out, implemented, measured, and validated separately.

## 9. Numerical validation philosophy

Correctness has both exact and numerical components. Structural properties must not be hidden behind floating-point tolerances.

Exact validation is required for:

- tensor shapes and dtypes;
- flat/head index mappings;
- GQA divisibility, group size, and query-to-KV mapping;
- causal visibility for every query/key position;
- absolute positions and KV-cache slot placement;
- unchanged past-cache entries and deterministic cache state transitions;
- packed nibble order and row-local 16-element block membership;
- logical FP4 code selection after Milestone 2 freezes it; and
- deterministic zero-tensor behavior.

Floating-point transformations require stage-specific contracts. Candidate comparison metrics include maximum absolute error, mean absolute error, RMSE, relative error where the reference magnitude makes it meaningful, and cosine similarity. Relevant stages include each RMSNorm, each projection, RoPE, QK scores, softmax, context, cache entries, and final output.

Quantized-path reports must include at least:

```text
mean absolute error
RMSE
maximum absolute error
cosine similarity
zero fraction
saturation fraction
```

Tolerance values are deliberately not guessed in Milestone 0. They must be justified per stage before a CUDA implementation is accepted, using the specified accumulation/storage boundaries and independent reference evidence. There is no single loose project-wide tolerance, and tolerances must not be widened merely until a test passes.

Cross-precision comparisons are numerical tradeoff studies, not exact-equivalence tests.

## 10. Benchmark methodology

Milestone 0 contains no benchmark implementation or result. CUDA events are the primary measurement for GPU kernel or pipeline latency, placed around the region under study with required synchronization outside the measured interval as appropriate. CUDA API and host launch overhead, if analyzed, must be measured and reported separately using host-side repeated timing and/or Nsight Systems; it must not be inferred solely from CUDA-event elapsed time.

Unless a result is explicitly labeled otherwise, timing excludes:

- memory allocation;
- random tensor initialization;
- host/device copies;
- process startup;
- compilation;
- printing; and
- one-time setup.

Each result will use warmup iterations and repeated timed samples and will report median, mean, minimum, and p95. Reports must record the relevant GPU, architecture, VRAM, driver, CUDA Toolkit, framework, host environment, build configuration, execution mode, shapes, cache state, and timing boundaries.

Logical bytes divided by elapsed time must not be labeled “DRAM bandwidth.” That term is reserved for traffic measured with appropriate hardware counters. Derived logical traffic rates may be reported under an accurately qualified name.

Performance claims must compare equivalent work:

- **Same-semantics optimization:** for example, `nvfp4_w4a16` baseline versus `nvfp4_w4a16` optimized. Direct latency or speedup comparisons are valid when inputs, outputs, precision rules, and work are equivalent.
- **Cross-precision tradeoff:** for example, `bf16` versus `nvfp4_w4a16`. Reports must pair latency or throughput with numerical-error metrics and must not imply identical numerical semantics.

### 10.1 Planned workload matrix

The matrix is intentionally compact rather than combinatorial.

Primary decode configurations:

| Parameter | Planned values |
|---|---|
| `B` | `1, 2` |
| `T` | `1` |
| `P` | `0, 128, 512, 2048, 8192` |
| `Hq` | `24` |
| `Hkv` | `24` (MHA), `12` (2:1), `6` (4:1 canonical GQA), `3` (8:1 GQA) |
| `D` | `128` |

Secondary prefill configurations:

| Parameter | Planned values |
|---|---|
| `B` | `1` |
| `T` | `64, 256, 1024` |
| `Hq` | `24` |
| `Hkv` | `6` |
| `D` | `128` |

These are planned benchmark points, not mandatory shape support for the first CUDA kernel.

## 11. Performance-development methodology

Development follows this measured sequence:

```text
correctness
→ baseline
→ real GPU validation
→ benchmark
→ profile
→ identify bottleneck
→ isolated optimization
→ correctness validation
→ A/B measurement
→ accept or reject
→ selective fusion
```

Milestone 0 does not pre-select a final fused design. Possible hypotheses include Q/K RMSNorm plus RoPE, norm plus quantization, projection staging, elimination of temporary tensors, and GQA decode preparation. Profiling must determine whether any hypothesis addresses a measured bottleneck. Each change needs isolated correctness validation and equivalent-work A/B measurement. Rejected optimizations are valid engineering results and should be recorded.

## 12. Milestone roadmap

### Milestone 0 — Specification

Freeze semantics, tensor contracts, NVFP4 terminology, scope, numerical philosophy, and benchmark philosophy.

### Milestone 1 — PyTorch Reference

Implement the high-confidence BF16/FP32 decoder-attention reference without FP4.

### Milestone 2 — NVFP4 Numerical Reference

Validate public NVFP4 encoding details, then implement E2M1 values, block scaling, tensor scaling, packing, quantization/dequantization, and numerical-error analysis.

### Milestone 3 — Modular CUDA Primitives

Implement individually testable CUDA primitives such as RMSNorm, Q/K RMSNorm, RoPE, and low-precision helpers.

### Milestone 4 — Baseline CUDA Pipeline

Build a correct multi-kernel end-to-end pipeline.

### Milestone 5 — RTX 4080 Validation

Run deterministic real-GPU correctness tests, multiple shapes, cached-decode tests, and Compute Sanitizer.

### Milestone 6 — Baseline Benchmarking

Create a reproducible CUDA-event benchmark harness.

### Milestone 7 — Profiling

Identify actual runtime, memory-traffic, launch, occupancy, and unpack/dequantization bottlenecks.

### Milestone 8 — First Isolated Optimization

Choose one optimization based on profiling and perform strict A/B validation.

### Milestone 9 — Selective Fusion

Fuse only stages supported by measured evidence.

### Milestone 10 — Advanced Low-Precision Path

Investigate packed decode, vectorized accesses, architecture-specific implementation, and native Blackwell work only if appropriate.

### Milestone 11 — Final Performance Study

Compare baseline, optimized modular, and fused paths together with numerical tradeoffs.

### Milestone 12 — Portfolio Polish

Produce concise documentation, reproducible artifacts, limitations, and evidence-backed performance claims.

## 13. Known limitations and deliberately deferred topics

- The study is decode-first; prefill is secondary.
- The first optimized CUDA path may support only `D = 128` and a subset of planned shapes.
- Milestone 0 freezes semantic layouts, not CUDA physical layouts.
- No cache allocator, variable-length batch representation, arbitrary mask API, or runtime integration is designed here.
- The initial KV cache is BF16, not quantized.
- Exact NVFP4 codes, scale encoding/terminology, scale selection, saturation, and underflow behavior await public-source validation in Milestone 2.
- No native Blackwell backend or layout conversion is designed in this milestone.
- Numerical tolerance values require stage-specific evidence and are not frozen yet.
- Fusion choices remain hypotheses until profiling and A/B measurements support them.
- No benchmark or performance number exists yet.

Milestone 0 stops at this specification. Milestone 1 implementation is intentionally not part of this repository state.
