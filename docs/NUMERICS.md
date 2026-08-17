# NVFP4 numerical contract

## 1. Scope

This document is the Milestone 2A public-source audit for the repository's
planned NVFP4 weight representation. It fixes the numerical vocabulary and the
portable reference contract needed by Milestone 2B. It does not implement a
quantizer, a CUDA backend, or a native Blackwell layout.

The audit was performed on 2026-08-17. The installed CUDA Toolkit remains 12.5;
newer public documentation was consulted without installing or changing the
toolkit. Milestone 0 attention, cache, masking, RoPE, GQA, and output semantics
remain unchanged.

Claims below use five labels:

- **NVIDIA-documented**: stated by public NVIDIA documentation.
- **NVIDIA implementation evidence**: observed in official public NVIDIA
  source, used where prose documentation is not bit-level complete.
- **Derived**: a mathematical consequence of cited facts, not NVIDIA prose.
- **Repository contract**: a deliberate portable project choice. It is not a
  claim of byte-for-byte native NVIDIA behavior.
- **Deliberately unresolved**: outside the portable contract or not publicly
  specified; it must not be assumed by a future backend.

## 2. Authoritative public sources

The normative source set is:

1. [CUDA Math API: `__nv_fp4_e2m1`][cuda-fp4-type] and [FP4 conversion][cuda-fp4-convert]
   (CUDA 12.9.1 archive).
2. [PTX ISA alternate floating-point formats][ptx-formats], [PTX `cvt`][ptx-cvt],
   [PTX floating-point rounding modifiers][ptx-rounding], and the
   [CUDA Programming Guide rounding-mode definition][cuda-rounding].
3. [cuBLAS narrow-precision usage][cublas-narrow] and [cuBLAS 1D block
   quantization][cublas-quantization].
4. [CUDA Math API: `__nv_fp8_e4m3`][cuda-fp8-type] and [FP8 conversion][cuda-fp8-convert]
   (CUDA 12.5.1 archive).
5. [Transformer Engine 2.15 NVFP4 documentation][te-nvfp4] and the current
   [Transformer Engine recipe C API][te-recipe-api].
6. Official Transformer Engine source pinned at commit
   `172bd93773ad6ee4ba44b460b7f10ef42fc89d57`: [core scaling][te-core],
   [quantization][te-quantize], [dequantization][te-dequantize], and the
   [Python reference][te-reference].
7. Official NVIDIA CUTLASS source pinned at commit
   `564d267e4c992c456d12ad02665f9acedf7708f1`: [`exmy_base.h`][cutlass-exmy],
   [`float_subbyte.h`][cutlass-subbyte], and [`float8.h`][cutlass-float8].

NVIDIA technical blogs were not used as the sole authority for any code point,
rounding rule, or scaling equation.

## 3. E2M1 format

### 3.1 Format facts and exact decode

**NVIDIA-documented.** E2M1 is four bits wide: one sign bit, two exponent
bits, one explicit mantissa bit, and one implicit significand bit for normal
values. It supports neither infinity nor NaN. CUDA scalar constructors from
FP32, BF16, and FP16 use `cudaRoundNearest` with saturate-to-finite behavior.
PTX carries two values as `.e2m1x2` in a `.b8` register
([CUDA FP4 type][cuda-fp4-type], [PTX formats][ptx-formats]). Transformer
Engine documents maximum magnitude 6 ([TE NVFP4][te-nvfp4]).

**NVIDIA implementation evidence and derived rule.** CUTLASS defines E2M1 as
a signed 4-bit `FpBitRepresentation` with two exponent bits, one mantissa bit,
exponent bias 1, denormals, and no exceptional encodings. Transformer Engine's
public reference contains the same 16-entry decode table
([CUTLASS `exmy_base.h`][cutlass-exmy], [TE reference][te-reference]). For a
nibble `c`:

```text
s = (c >> 3) & 1
e = (c >> 1) & 3
m = c & 1

if e == 0:
    value = (-1)^s * (m / 2)
else:
    value = (-1)^s * (1 + m / 2) * 2^(e - 1)
```

The exact code map is therefore:

| Code | Bits `s ee m` | Value | Class |
|---:|:---:|---:|---|
| `0x0` | `0 00 0` | `+0` | zero |
| `0x1` | `0 00 1` | `+0.5` | subnormal |
| `0x2` | `0 01 0` | `+1` | normal |
| `0x3` | `0 01 1` | `+1.5` | normal |
| `0x4` | `0 10 0` | `+2` | normal |
| `0x5` | `0 10 1` | `+3` | normal |
| `0x6` | `0 11 0` | `+4` | normal |
| `0x7` | `0 11 1` | `+6` | normal |
| `0x8` | `1 00 0` | `-0` | zero |
| `0x9` | `1 00 1` | `-0.5` | subnormal |
| `0xA` | `1 01 0` | `-1` | normal |
| `0xB` | `1 01 1` | `-1.5` | normal |
| `0xC` | `1 10 0` | `-2` | normal |
| `0xD` | `1 10 1` | `-3` | normal |
| `0xE` | `1 11 0` | `-4` | normal |
| `0xF` | `1 11 1` | `-6` | normal |

Thus every code is finite. The magnitude set is exactly
`{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. The only nonzero subnormal magnitude is 0.5;
the minimum positive normal is 1; and the maximum magnitude is 6. Codes `0x0`
and `0x8` are distinct signed zeros. CUTLASS preserves the source sign when a
finite conversion produces zero ([CUTLASS `exmy_base.h`][cutlass-exmy]).

**Repository contract.** The standalone portable E2M1 encoder preserves the
sign of an exact zero and of a negative finite value rounded to zero. The
scale-zero block policy in Section 8 deliberately overrides this by emitting
canonical `+0` payloads.

### 3.2 Rounding, saturation, and exceptional sources

**NVIDIA-documented.** PTX `.rn` and CUDA `rn` are round-to-nearest,
ties-to-even, and E2M1 destinations require `.satfinite`. CUDA's FP4 conversion
API says that large out-of-range inputs become same-sign `MAXNORM` and NaN
becomes positive `MAXNORM` ([PTX `cvt`][ptx-cvt], [PTX rounding][ptx-rounding],
[CUDA rounding][cuda-rounding], [CUDA FP4 conversion][cuda-fp4-convert]).

**Derived.** Combining those conversion rules with E2M1 `MAXNORM = 6` means
finite values outside `[-6, 6]` saturate to `-6` or `+6`, and `NaN -> +6`.

**Derived exceptional-value consequence.** Applying the documented same-sign
saturation rule to infinities yields `-Inf -> -6` and `+Inf -> +6`.

Rounding is an operation property, not a stored-format property. Transformer
Engine training may use stochastic rounding ([TE NVFP4][te-nvfp4]).

**Repository contract.** M2B uses deterministic round-to-nearest,
ties-to-even, followed by finite saturation. It accepts only finite source
tensors, so NVIDIA's exceptional-source conversions are documented here but
are not part of the repository API.

**Derived and repository contract.** Combining nearest-even with the exact code
map and the repository's signed-zero policy gives the required midpoint table:

| Adjacent magnitudes | Midpoint | Positive result | Negative result | Tie reason |
|---|---:|---|---|---|
| `0 <-> 0.5` | `0.25` | `+0` / `0x0` | `-0` / `0x8` | zero code has even LSB |
| `0.5 <-> 1` | `0.75` | `+1` / `0x2` | `-1` / `0xA` | `1` code has even LSB |
| `1 <-> 1.5` | `1.25` | `+1` / `0x2` | `-1` / `0xA` | `1` code has even LSB |
| `1.5 <-> 2` | `1.75` | `+2` / `0x4` | `-2` / `0xC` | `2` code has even LSB |
| `2 <-> 3` | `2.5` | `+2` / `0x4` | `-2` / `0xC` | `2` code has even LSB |
| `3 <-> 4` | `3.5` | `+4` / `0x6` | `-4` / `0xE` | `4` code has even LSB |
| `4 <-> 6` | `5.0` | `+4` / `0x6` | `-4` / `0xE` | `4` code has even LSB |

The table is also cross-validated by the explicit interval comparisons in the
official [Transformer Engine Python reference][te-reference].

## 4. UE4M3 scale format

### 4.1 E4M3 is related to, but not identical to, UE4M3

**NVIDIA-documented.** Signed E4M3 is an 8-bit format with one sign bit, four
exponent bits, three explicit mantissa bits, and an implicit bit for normals.
It has no infinity; `0x7f` and `0xff` are NaNs
([CUDA FP8 type][cuda-fp8-type]). PTX UE4M3 is instead a logical 7-bit unsigned
format with four exponent and three mantissa bits, no infinity, and one NaN
code, `0x7f`. It occupies a `.b8` value whose MSB is padding and must be zero
([PTX formats][ptx-formats]). cuBLAS describes `CUDA_R_8F_UE4M3` as unsigned
E4M3 whose sign bit is ignored ([cuBLAS narrow precision][cublas-narrow]).

**Derived.** NVFP4 block-scale candidates are nonnegative. A nonnegative finite
signed E4M3 value has sign bit zero, so its byte in `0x00..0x7e` is numerically
and bitwise the canonical finite UE4M3 value. This is why Transformer Engine
can call the scale "E4M3" while PTX and cuBLAS name native block-scale storage
"UE4M3". The type names must not otherwise be treated as synonyms.

**Repository contract.** A portable scale byte is canonical UE4M3: bit 7 must
be zero and `0x7f` is rejected as NaN. Valid finite stored bytes are exactly
`0x00..0x7e`. Although cuBLAS ignores bit 7, bytes `0x80..0xff` are not accepted
because PTX requires zero padding and their behavior is not portable across
interfaces.

### 4.2 Exact decode and encode

**NVIDIA implementation evidence and derived rule.** CUDA E4M3 and CUTLASS
establish exponent bias 7, subnormal handling, maximum finite code `0x7e`, and
nearest-even finite conversion ([CUDA FP8 type][cuda-fp8-type],
[CUTLASS `float8.h`][cutlass-float8]). For canonical byte `c`:

```text
require 0x00 <= c <= 0x7f
e = (c >> 3) & 0xf
m = c & 0x7

if e == 0 and m == 0:
    value = 0
elif e == 0:
    value = m * 2^-9
elif c == 0x7f:
    value = NaN
else:
    value = (1 + m / 8) * 2^(e - 7)
```

This is an unambiguous mapping for every canonical byte. Its limits are:

| Property | Code | Value |
|---|---:|---:|
| zero | `0x00` | `0` |
| minimum positive subnormal | `0x01` | `2^-9 = 0.001953125` |
| maximum subnormal | `0x07` | `7 * 2^-9 = 0.013671875` |
| minimum positive normal | `0x08` | `2^-6 = 0.015625` |
| maximum finite | `0x7e` | `448` |
| NaN | `0x7f` | NaN |

UE4M3 has no sign, signed zero, or infinity. Normal values use an implicit
leading one; subnormals do not.

**NVIDIA-documented.** cuBLAS assumes RNE for 1D block quantization, casts the
FP32 candidate with `e4m3(...)`, and then uses the decoded stored scale
([cuBLAS 1D quantization][cublas-quantization]). CUDA FP8 conversion specifies
round-to-nearest-even, and `__NV_SATFINITE` clamps finite overflow to same-sign
`MAXNORM` ([CUDA FP8 conversion][cuda-fp8-convert]).

**Repository contract.** A scale encoder accepts a finite nonnegative FP32
candidate, chooses the nearest finite code in `0x00..0x7e`, resolves an exact
tie toward the endpoint with even destination significand LSB, and saturates
finite overflow to `0x7e`. In particular, `2^-10` is exactly halfway between
zero and `2^-9` and rounds to `0x00`. Negative or non-finite scale candidates
are rejected rather than assigning undocumented standalone UE4M3 behavior.

## 5. Intentional 1D NVFP4 variant

**Repository contract.** For `W[N,K]`, require `K % 16 == 0`. Block `(n,b)` is
exactly:

```text
W[n, 16*b : 16*b + 16]
```

There is one scale for those 16 K-contiguous values. Blocks never cross a row.

**NVIDIA-documented.** cuBLAS supports 16-element 1D FP4 scaling with E2M1
values and UE4M3 scale storage, with a scale for each block in the innermost
dimension ([cuBLAS narrow precision][cublas-narrow]). Transformer Engine
documents both 1D blocks of 16 consecutive elements and 2D `16 x 16` blocks.
Its training recipe uses 2D weight scaling by default and 1D scaling for
activations and gradients; `disable_2d_quantization=True` forces 1D weights
([TE NVFP4][te-nvfp4]).

The initial inference-oriented repository reference intentionally implements
only the 1D weight variant. It does not imply that 1D is the only NVFP4 weight
scheme and does not attempt Transformer Engine's 2D training recipe.

## 6. Encode and decode scale terminology

Use the following unambiguous names:

```text
A     = global_amax = max(abs(W))
alpha = global_encode_scale
gamma = global_decode_scale stored by this repository
```

The audited constants are:

```text
FP4_E2M1_MAX = 6
FP8_E4M3_MAX = 448
FP4_E2M1_MAX * FP8_E4M3_MAX = 2688
```

**NVIDIA-documented.** cuBLAS gives the tensor-wide encode factor
`alpha = Amax(E2M1) * Amax(E4M3) / A`, while Transformer Engine gives the
reconstruction factor `s_global = A / (448 * 6)`
([cuBLAS 1D quantization][cublas-quantization], [TE NVFP4][te-nvfp4]). The
Transformer Engine C API and source compute `2688 / A` and return encode scale
1 when `A <= 0` ([TE recipe API][te-recipe-api], [TE recipe source][te-recipe-source]).
For a valid, nonempty, finite input, `A` cannot be negative, so zero is the only
nonpositive case.

**Derived, in real arithmetic for finite `A > 0`.**

```text
alpha = 2688 / A
gamma = 1 / alpha
      = A / 2688
```

The reciprocal identity is mathematical; it is not a promise that two
independently rounded FP32 expressions are bit-identical.

**NVIDIA implementation evidence and derived consequence.** Transformer Engine
caps an overflowing FP32 `alpha` at `FLT_MAX`, but its dequantizer uses
`A * (1 / 2688)` directly ([TE core][te-core],
[TE dequantization][te-dequantize]). It follows that for extremely tiny
positive `A`, the cap can break machine-level reciprocity and the FP32 decode
factor can underflow to zero.

**Repository contract.** M2B will compute both directions explicitly in FP32:

```text
if A == 0:
    alpha = 1
else:
    alpha = min(FP32(2688 / A), FLT_MAX)

gamma = FP32(A * FP32(1 / 2688))
```

Thus an all-zero tensor stores `gamma = 0`; the internal encode fallback
`alpha = 1` is intentionally not its reciprocal. No minimum global scale is
invented. Tests must cover the ordinary reciprocal derivation, the zero
exception, the `FLT_MAX` cap, and FP32 decode-scale underflow separately.

## 7. Standard max-based quantization procedure

For block `(n,b)`, let `B = max(abs(block))`, let `u` be the stored canonical
UE4M3 byte, and let `beta = UE4M3_decode(u)`.

**NVIDIA-documented and derived conceptual equations.** Transformer Engine's
published decode scales give:

```text
raw_block_decode_scale = B / 6
scaled_block_candidate = raw_block_decode_scale * alpha
stored_block_scale     = UE4M3_RNE_satfinite(scaled_block_candidate)
```

Equivalently, before FP32 rounding:

```text
scaled_block_candidate = 448 * B / A       # ordinary finite A > 0
```

cuBLAS computes the block amax, divides it by the destination maximum, converts
the scale to E4M3/UE4M3 under RNE, and uses the reciprocal decoded scale for
narrow conversion ([TE NVFP4][te-nvfp4], [cuBLAS 1D quantization][cublas-quantization]).

**NVIDIA implementation evidence.** The source-faithful FP32 evaluation order
for the block candidate is:

```text
candidate_multiplier  = FP32(alpha * FP32(1 / 6))
scaled_block_candidate = FP32(B * candidate_multiplier)
u                       = UE4M3_RNE_satfinite(scaled_block_candidate)
beta                    = UE4M3_decode(u)
```

Transformer Engine intentionally uses this ordering for exact emulation
([TE core][te-core], [TE scaling reference][te-reference-scale]).

When `beta > 0`, define the conversion factor and values as:

```text
global_encode_inverse = FP32(1 / alpha)
conversion_factor = min(
    FP32(1 / FP32(beta * global_encode_inverse)),
    FLT_MAX
)
q = E2M1_RNE_satfinite(FP32(x * conversion_factor))
```

**Derived.** In real arithmetic this verifies the expected relationship:

```text
conversion_factor = alpha / beta
```

The official Transformer Engine implementation defines and uses the
reciprocal-product form above ([TE core][te-core],
[TE quantization][te-quantize]). Pulling cuBLAS's tensor-wide input scale out
of its scaled block gives the same algebraic result.

Reconstruction is:

```text
x_hat = E2M1_decode(q) * beta * gamma
```

This is NVIDIA's documented hierarchical decode equation with the repository's
directional names ([TE NVFP4][te-nvfp4]). Scale rounding means the block maximum
can still reach finite E2M1 saturation; the stored, rounded `beta`, not the raw
candidate, must be used in both conversion and reconstruction.

**Repository contract.** The portable FP32 reconstruction evaluation order is:

```text
x_hat = FP32(FP32(E2M1_decode(q) * beta) * gamma)
```

This order makes the reference deterministic but does not claim bitwise parity
with native kernels that can reassociate arithmetic or convert to another
output dtype. M2B uses the other stated FP32 orders and deterministic E2M1 RNE.
Transformer Engine's optional stochastic training cast, optional 4-over-6
variant, fast-math paths, and 2D paths are outside this contract.

## 8. Zero, underflow, and exceptional-value policy

**NVIDIA implementation evidence.** Transformer Engine returns `alpha = 1`
for zero amax, produces an E4M3 zero scale when `B = 0`, and caps the reciprocal
conversion factor at `FLT_MAX` rather than applying a minimum scale clamp
([TE core][te-core], [TE quantization][te-quantize]). Its dequantizer multiplies
by `A / 2688`, so `A = 0` reconstructs zero
([TE dequantization][te-dequantize]). No authoritative source inspected applies
a positive minimum clamp to an underflowed UE4M3 block scale.

**Repository contract.** The portable deterministic cases are:

| Case | `alpha` | `gamma` | Scale byte | E2M1 payload | Reconstruction |
|---|---:|---:|---:|---|---|
| entire tensor has `A = 0` | `1` | `0` | every block `0x00` | canonical `0x0` for every element | exact zero |
| zero block inside `A > 0` | standard | standard | that block `0x00` | canonical `0x0` for every element | exact zero |
| nonzero block whose candidate rounds to zero | standard | standard | that block `0x00` | canonical `0x0` for every element | zero; block information is lost |

Whenever `beta == 0`, the portable reference does not divide. It stores
canonical positive-zero payloads and stops processing that block. Disregarding
the sign of zero, this has the same real-valued reconstruction as any native
payload because `q * 0 * gamma = 0`. It is deterministic on CPU and Ada and
avoids reporting meaningless maximum-code payloads as saturation. It is a
repository policy, not a byte-parity claim for native Transformer Engine or
cuBLAS.

**Derived.** In the ideal, uncapped real-arithmetic path:

```text
scaled_block_candidate = 448 * B / A
```

The UE4M3 zero/subnormal midpoint is `2^-10`, and the tie goes to zero.
Therefore a nonzero block rounds to scale zero when:

```text
B / A <= 1 / (448 * 1024)
      = 1 / 458752
```

This threshold is derived, not NVIDIA prose. FP32 expression order and the
global `FLT_MAX` cap can move the effective boundary. There is no minimum
block-scale or minimum global-decode-scale clamp in the repository contract.

**Repository API policy.** Quantization inputs must be finite. NaN and infinity
are rejected before amax reduction. This is stricter than NVIDIA's raw E2M1
exception conversion and avoids assigning behavior to non-finite UE4M3 scale
candidates.

## 9. Portable packing layout

The repository layout remains:

```text
packed_values:       uint8 [N, K/2]
block_scales:        uint8 [N, K/16]
global_decode_scale: FP32 scalar

packed_values[n,r] low nibble  = logical element W[n, 2*r]
packed_values[n,r] high nibble = logical element W[n, 2*r + 1]
```

This is a **repository contract**. It is row-major, simple to inspect, and not
a claim about a native matrix layout.

**NVIDIA-documented.** For scalar FP32 PTX conversion
`cvt.rn.satfinite.e2m1x2.f32 d, a, b`, operand `a` goes to `d[7:4]` and operand
`b` goes to `d[3:0]`. For packed FP16/BF16 input, the upper source lane goes to
the high nibble and the lower source lane to the low nibble
([PTX `cvt`][ptx-cvt]). Thus the repository's even-low convention must not be
described as the native scalar PTX operand order. A backend can reverse operand
presentation or transform nibbles as needed.

cuBLAS's native 16-element block scales use a tiled physical layout rather than
the repository's `[N,K/16]` row-major array
([cuBLAS 1D quantization][cublas-quantization]). A later native backend must
perform an explicit layout conversion for values and scales.

## 10. Planned quantized tensor contract

M2B should introduce one small reference-oriented Python data object with four
conceptual fields; this section designs it but does not implement it.

| Field | Type | Shape / meaning |
|---|---|---|
| `packed_values` | `torch.uint8` tensor | `[N, K/2]`, portable nibble layout |
| `block_scales` | `torch.uint8` tensor | `[N, K/16]`, canonical UE4M3 bytes |
| `global_decode_scale` | `torch.float32` tensor | scalar shape `[]`, multiplicative decode factor |
| `logical_shape` | Python `tuple[int, int]` | exactly `(N, K)` |

Validation invariants:

- `N >= 1`, `K >= 16`, and `K % 16 == 0`.
- Tensor shapes and dtypes exactly match the table; packed tensors are
  contiguous.
- All tensor fields are on the same device. `logical_shape` is host metadata.
- Every scale byte has bit 7 clear and is not `0x7f`; hence it is in
  `0x00..0x7e`. Every value nibble `0x0..0xf` is valid E2M1.
- `global_decode_scale` is finite and nonnegative. Zero is valid for the
  all-zero case and for an extreme FP32 decode-scale underflow.
- Quantization input is a finite, contiguous, rank-2 FP32 or BF16 tensor. FP16
  is outside the initial API. Amax, scale, rounding-decision, and reconstruction
  arithmetic are performed in FP32; BF16 input is promoted to FP32 for those
  operations.
- CPU and CUDA inputs are allowed for the portable reference. Outputs remain on
  the input device, and there are no implicit device transfers.
- Quantization is deterministic and does not expose a generic tensor framework
  or stochastic-training configuration.

## 11. Numerical error metrics

Let `M = N*K`, compare source `W` with reconstruction `W_hat`, and use FP64 for
metric reductions after forming FP32 values. M2B reports at least:

```text
maximum_absolute_error = max_i |W_hat_i - W_i|
mean_absolute_error    = (1/M) * sum_i |W_hat_i - W_i|
RMSE                   = sqrt((1/M) * sum_i (W_hat_i - W_i)^2)
cosine_similarity      = dot(W_hat, W) / (norm(W_hat) * norm(W))
zero_fraction          = count(q_code in {0x0, 0x8}) / M
```

For cosine similarity, define both-zero vectors as 1 and exactly-one-zero
vectors as 0. Relative error is not a primary aggregate because it is unstable
near zero.

Define the pre-E2M1 value `y_i = FP32(W_i * conversion_factor_i)` only for
blocks with `beta > 0`. The primary saturation fraction is the strict clipping
fraction:

```text
saturation_fraction =
    count(beta_i > 0 and |y_i| > 6 and |E2M1_decode(q_i)| == 6) / M
```

The denominator is every logical element. Exact `|y_i| == 6` is not saturation,
and an in-range value rounded to 6 is not clipping. Scale-zero blocks are not in
the numerator. If useful, report the separate descriptive metric
`maximum_code_fraction = count(|E2M1_decode(q_i)| == 6) / M`; it must not be
called saturation. Also report
`scale_underflow_block_fraction = count(B > 0 and beta == 0) / (N*K/16)` in the
tiny-value stress case.

## 12. Planned Milestone 2B validation matrix

No tests are implemented in Milestone 2A. The future matrix explicitly
separates exact representation/contract tests from distribution-level quality
tests.

| ID | Kind | Planned assertion |
|:--:|---|---|
| A | exact representation | Every E2M1 nibble has the exact code-to-value map in Section 3. |
| B | exact representation | Decode all 16 E2M1 codes, including both zero signs. |
| C | exact representation | Raw `+0` and `-0` encode/decode as specified; scale-zero block canonicalization emits `+0`. |
| D | exact representation | All positive and negative midpoint cases use the table in Section 3.2. |
| E | exact representation | Finite values outside `[-6,6]` saturate with the correct sign. |
| F | exact representation | Decode every canonical UE4M3 byte and reject `0x7f` and nonzero MSB bytes. |
| G | exact representation | Every finite UE4M3 code `0x00..0x7e` survives encode/decode round trip exactly. |
| H | exact contract | An all-zero tensor yields `alpha=1`, `global_decode_scale=0`, zero scales, canonical zero payloads, and exact reconstruction. |
| I | exact contract | A zero block inside a nonzero tensor uses scale `0x00`, canonical zero payloads, and no division. |
| J | exact contract | One hand-computable 16-value block matches scale, codes, bytes, and reconstruction. |
| K | exact contract | Multiple blocks in a row use independent row-local scales. |
| L | exact contract | Blocks stop at row boundaries and never cross rows. |
| M | exact representation | Even logical element is the low nibble and odd logical element is the high nibble. |
| N | exact representation | Pack followed by unpack is an exact nibble round trip. |
| O | exact contract | `alpha=2688/A` and `gamma=A/2688` satisfy the real-arithmetic reciprocal derivation for ordinary `A>0`; zero and FP32-cap cases follow their explicit branches. |
| P | exact contract | Reconstruction uses decoded `q`, decoded stored `beta`, and stored `gamma`, never the raw block candidate. |
| Q | statistical quality | Seeded random normal input reports all required metrics. |
| R | statistical quality | Seeded random uniform input reports all required metrics. |
| S | statistical quality | Seeded outlier-heavy input reports all required metrics. |
| T | exact contract / stress | Local UE4M3 scale underflow and global FP32 decode-scale underflow follow Section 8 with no clamp. |
| U | exact contract / stress | Large finite inputs exercise maximum-code and strict saturation counters separately. |
| V | statistical quality | Error-statistic formulas, zero fraction, saturation fraction, and cosine zero conventions are independently checked. |
| W | exact contract | Repeated runs are byte-for-byte deterministic on the same device. |
| X | exact contract | Wrong rank, empty dimensions, `K % 16 != 0`, dtype, shape, device, contiguity, and invalid scale bytes are rejected. |
| Y | exact contract | NaN and infinity source tensors are rejected. |

Statistical tests use fixed seeds and assert finite metrics and documented broad
quality bounds; they must not replace the exact code, byte, scale, or equation
tests.

## 13. Native-layout and toolkit limitations

- CUDA Toolkit 12.5 on the development machine does not provide the newer FP4
  interface used by the audited public documentation. No toolkit, driver, or
  system change is required or permitted for this reference contract.
- Current PTX native E2M1 conversions target Blackwell-family architectures,
  not Ada SM89 ([PTX `cvt`][ptx-cvt]). The future Ada path must implement these
  audited semantics in software.
- Repository value bytes and row-major scale bytes are portable logical data,
  not native cuBLAS/CUTLASS Blackwell physical layouts.
- The reference does not promise parity with Transformer Engine stochastic
  rounding, 2D weight scaling, optional 4-over-6 scaling, swizzles, or training
  transforms.

## 14. Architecture review finding and unresolved native questions

### 14.1 Frozen architecture wording requiring review

No change was made to `docs/ARCHITECTURE.md`.

**Contradiction.** The same positive-versus-zero issue occurs four times:

- Section 8.1: "one positive eight-bit floating block scale per microscaling
  block";
- Section 8.2: "one positive logical block scale for each `(n,b)`";
- Section 8.3: "byte for the positive scale"; and
- Section 8.4: "each positive NVFP4 block scale."

Authoritative NVIDIA formulas and source produce scale code `0x00` for a zero
block and for a candidate that underflows to zero ([TE NVFP4][te-nvfp4],
[TE core][te-core]). The source-backed correction is to use "nonnegative" for
the block scale throughout; where the representation is named, it is a
nonnegative UE4M3-compatible scale. Zero must be permitted. This is a strict
mathematical contradiction, not merely a clarification, and requires review
before editing the frozen architecture file.

**Clarification only.** Sections 8.2 and 8.3 use `global_scale` even though the
text already defines it in the decode direction. Renaming it
`global_decode_scale` would remove encode/decode ambiguity without changing the
architecture equation or semantics.

### 14.2 Deliberately unresolved, non-blocking for the portable reference

- Public cuBLAS documentation does not normatively fix E2M1 payload bytes for a
  nonzero block whose stored scale is zero. The repository's canonical-zero
  policy resolves portable M2B but is not a native byte-parity promise.
- A cuBLAS UE4M3 byte with bit 7 set has an ignored sign bit, whereas PTX
  requires that bit to be zero padding. The repository accepts only canonical
  bytes.
- A standalone negative, NaN, or infinity FP32-to-UE4M3 conversion contract is
  not needed: scale candidates are finite and nonnegative by repository policy.
- Native value/scale swizzles remain backend concerns and are not inferred from
  the portable layout.

The pinned CUTLASS `float_subbyte.h` contains a non-normative human-readable
E2M1 range comment that includes `5`. That source anomaly conflicts with the
file's own four-bit encoding implementation, the CUDA format definition, and
Transformer Engine's explicit 16-code table. It is treated as a comment typo,
not a representable value or a contradiction with the frozen magnitude set.

There is no unresolved E2M1 code-map, UE4M3 finite-encoding, rounding, or
max-based-scaling blocker for a portable M2B. M2B must nevertheless wait for
review of the Section 8 "positive" to "nonnegative" architecture correction.

[cuda-fp4-type]: https://docs.nvidia.com/cuda/archive/12.9.1/cuda-math-api/cuda_math_api/struct____nv__fp4__e2m1.html
[cuda-fp4-convert]: https://docs.nvidia.com/cuda/archive/12.9.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html
[ptx-formats]: https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats
[ptx-cvt]: https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
[ptx-rounding]: https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-rounding-modifiers
[cuda-rounding]: https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html#rounding-modes
[cublas-narrow]: https://docs.nvidia.com/cuda/cublas/#narrow-precision-data-types-usage
[cublas-quantization]: https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization
[cuda-fp8-type]: https://docs.nvidia.com/cuda/archive/12.5.1/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html
[cuda-fp8-convert]: https://docs.nvidia.com/cuda/archive/12.5.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html
[te-nvfp4]: https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html
[te-recipe-api]: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/c/recipe.html
[te-core]: https://github.com/NVIDIA/TransformerEngine/blob/172bd93773ad6ee4ba44b460b7f10ef42fc89d57/transformer_engine/common/cast/nvfp4/core_nvfp4.cuh#L34-L95
[te-quantize]: https://github.com/NVIDIA/TransformerEngine/blob/172bd93773ad6ee4ba44b460b7f10ef42fc89d57/transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L734-L785
[te-dequantize]: https://github.com/NVIDIA/TransformerEngine/blob/172bd93773ad6ee4ba44b460b7f10ef42fc89d57/transformer_engine/common/cast/nvfp4/dequantize_nvfp4.cuh#L34-L80
[te-reference]: https://github.com/NVIDIA/TransformerEngine/blob/172bd93773ad6ee4ba44b460b7f10ef42fc89d57/transformer_engine/pytorch/custom_recipes/reference_nvfp4.py#L49-L141
[te-reference-scale]: https://github.com/NVIDIA/TransformerEngine/blob/172bd93773ad6ee4ba44b460b7f10ef42fc89d57/transformer_engine/pytorch/custom_recipes/reference_nvfp4.py#L651-L777
[te-recipe-source]: https://github.com/NVIDIA/TransformerEngine/blob/172bd93773ad6ee4ba44b460b7f10ef42fc89d57/transformer_engine/common/recipe/nvfp4.cu#L614-L677
[cutlass-exmy]: https://github.com/NVIDIA/cutlass/blob/564d267e4c992c456d12ad02665f9acedf7708f1/include/cutlass/exmy_base.h
[cutlass-subbyte]: https://github.com/NVIDIA/cutlass/blob/564d267e4c992c456d12ad02665f9acedf7708f1/include/cutlass/float_subbyte.h
[cutlass-float8]: https://github.com/NVIDIA/cutlass/blob/564d267e4c992c456d12ad02665f9acedf7708f1/include/cutlass/float8.h
