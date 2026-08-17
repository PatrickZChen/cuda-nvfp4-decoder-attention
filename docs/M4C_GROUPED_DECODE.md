# M4C grouped-decode W4A16 experiment

## Decision

**RETAIN** the grouped-decode candidate as a separate experimental primitive.
Do not promote it over the frozen M4A API yet.

The candidate passed the frozen numerical policy and both required Compute
Sanitizer runs. In five alternating-order A/B rounds it was faster in every
round for both primary decode cases. The median round speedup was `1.283582x`
for canonical Q/O and `1.190476x` for canonical K/V. K=128 was a real
secondary-regime regression, with a median round ratio of `0.866667x`; it is
retained as a limitation rather than hidden or tuned away in this experiment.

The complete raw samples and machine metadata are in
[`m4c_grouped_decode.json`](../benchmarks/results/rtx4080-laptop-sm89/m4c_grouped_decode.json).
M4B remains the frozen unoptimized baseline documented in
[`PERFORMANCE_BASELINE.md`](PERFORMANCE_BASELINE.md).

## Baseline and isolation boundary

The baseline remains `cuda_w4a16_linear`, one 256-thread CUDA block per output
scalar. M4C adds the separate API
`cuda_w4a16_linear_grouped_decode`; the baseline wrapper is not routed to the
candidate.

The frozen artifacts remained byte-identical after implementation and
measurement:

| Artifact | Frozen source | SHA-256 |
|---|---|---|
| `src/w4a16.cu` | `ee6f7ef:src/w4a16.cu` | `b5043098eb2b5245087683131a41e05f6d7cf1aea5de807ca31624331bc37961` |
| `baseline_w4a16.json` | `511b1c5:benchmarks/results/rtx4080-laptop-sm89/baseline_w4a16.json` | `e6b74b562fa669491da29b265bf05233e32c3b4793a280b12fb87b168355fbd9` |

No activation tiling, LUT, predecoded scale table, block-size tuning,
warp-per-output mapping, multiple-output block, vectorized wide-load
experiment, asynchronous copy, Tensor Core path, or fusion was added.

## Candidate mapping

The grid and block mapping are unchanged: output scalar `(m,n)` maps to one
CUDA block, and every block has 256 active threads. The candidate partitions
those threads into 32 contiguous groups of eight:

```text
group_id     = threadIdx.x / 8       # 0..31
lane         = threadIdx.x % 8       # 0..7
group blocks = group_id, group_id + 32, ... < K/16
```

For microscale block `b`, lane `l` loads exactly:

```text
packed = packed_values[n, 8*b + l]
even k = 16*b + 2*l      code = packed & 0x0f
odd  k = 16*b + 2*l + 1  code = packed >> 4
```

Thus one group consumes one complete 16-weight block per loop iteration, and
one packed-byte load supplies two logical weights. For K=3072 there are 192
microscale blocks, so every group receives six. Nothing hard-codes either
value; the loop uses `blocks_per_row = K/16`.

### Scale broadcast safety

Only `lane == 0` loads the row-local scale byte and calls the unchanged UE4M3
decoder. The eight lanes share a uniform loop condition because they have the
same `group_id`. The source forms:

```text
warp_lane       = threadIdx.x % 32
group_base_lane = warp_lane - lane
subgroup_mask   = 0xff << group_base_lane
active_mask     = __activemask() & subgroup_mask
beta            = __shfl_sync(active_mask, beta_from_lane_zero, 0, 8)
```

Eight divides the 32-lane warp, so group bases are `0`, `8`, `16`, or `24` and
no group crosses a warp. `srcLane=0` is relative to each width-eight segment.
Intersecting with `__activemask()` excludes sibling groups that have finished
their loops, while all eight lanes of a participating group are named and
execute the same shuffle. There is no block-wide synchronization in the K
loop.

After loop reconvergence, all 256 threads—including zero-contributing groups—
join the same full-warp and shared-warp-sum reduction as M4A. K=16 therefore
has one data-bearing group and 248 safe zero contributors.

## Numerical operation order

The candidate duplicates the audited E2M1 and UE4M3 software decoders without
changing the baseline source. For both nibbles it explicitly evaluates:

```text
q            = E2M1_decode(code)
beta         = UE4M3_decode(scale_byte)       # once by subgroup lane zero
block_scaled = __fmul_rn(q, beta)
weight       = __fmul_rn(block_scaled, gamma)
activation   = __bfloat162float(x[k])
product      = __fmul_rn(activation, weight)
partial_sum  = __fadd_rn(partial_sum, product)
```

The even product is added before the odd product. The block reduction remains
explicit FP32 addition, and lane zero converts the final result with
`__float2bfloat16_rn`. There is no fast math, native FP4 instruction, BF16
decoded-weight storage, FP32 weight materialization, or FFMA reassociation.

## Derived source-level work

The following is structural source accounting, not measured traffic or a
dynamic instruction count:

| Per output scalar | Frozen baseline | Grouped candidate |
|---|---:|---:|
| Packed-byte loads requested | `K` at logical-weight granularity | `K/2` |
| UE4M3 scale decodes | `K` | `K/16` |
| E2M1 decodes | `K` | `K` |
| BF16 activation values loaded | `K` | `K` |
| Products | `K` | `K` |

## API and structural validation

The Python API is:

```python
cuda_w4a16_linear_grouped_decode(x, weight: NVFP4Tensor) -> torch.Tensor
```

The registered raw operator is:

```text
cuda_w4a16_linear_grouped_decode(
    x,
    packed_values,
    block_scales,
    global_decode_scale,
) -> Tensor
```

It returns contiguous BF16 `[..., N]` storage for `X @ W_hat.T`. The copied
M4A structural checks reject wrong device, dtype, rank, shape, contiguity,
empty storage, cross-device inputs, K below 16, K not divisible by 16, K
mismatch, invalid `[N,K/16]` scale shape, nonscalar FP32 global scale, and an
unsafe `M*N` grid. The wrapper neither copies nor makes inputs contiguous, and
the raw hot path performs no device-content scan or scalar `.item()`.

## Correctness and safety

Candidate-specific pytest covered the exact K=16 hand projection,
`X @ W.T` orientation, independent low/high nibbles, row-local scale indexing,
all 16 E2M1 codes, all 127 finite canonical UE4M3 codes, zero activation,
all-zero weight, scale-zero blocks, signed mixed E2M1 values, K of 16, 32, 128,
512, and 3072, ranks 1 through 4, multiple M/N values, M>1 at K=3072, a real
non-default stream chain, and invalid raw inputs.

| Validation | Result |
|---|---|
| Candidate-targeted pytest | 40 passed, 3 multi-GPU skips |
| Full `.venv/bin/pytest -q` | 319 passed, 9 skipped |
| Full `.venv/bin/python -m pytest -q` | 319 passed, 9 skipped |
| Candidate memcheck | `ERROR SUMMARY: 0 errors` |
| Frozen M4A memcheck | `ERROR SUMMARY: 0 errors` |
| Non-default stream | preparation → candidate → consumer completed correctly |

The benchmark correctness guards used the exact same deterministic inputs as
the timed paths and ran outside timing:

| Case | Comparison | Max abs | Mean abs | Exact BF16 fraction | Max BF16 adjacency |
|---|---|---:|---:|---:|---:|
| Q/O M=1, N=3072, K=3072 | candidate vs reference | 0 | 0 | 1.0 | 0 |
| Q/O M=1, N=3072, K=3072 | candidate vs baseline | 0.00390625 | 1.2715658e-6 | 0.999674499 | 1 |
| K/V M=1, N=768, K=3072 | candidate vs reference | 0 | 0 | 1.0 | 0 |
| K/V M=1, N=768, K=3072 | candidate vs baseline | 0 | 0 | 1.0 | 0 |
| Q/O M=2, N=3072, K=3072 | candidate vs reference | 0 | 0 | 1.0 | 0 |
| Q/O M=2, N=3072, K=3072 | candidate vs baseline | 0.00390625 | 1.2715658e-6 | 0.999674499 | 1 |

These inputs happened to make the candidate bit-exact to the reference. That
observation does not replace the frozen acceptance policy: a future valid
input may differ by one adjacent BF16 value because thread association changed.

## A/B timing method

Every case used one `x`, `packed_values`, `block_scales`, and
`global_decode_scale` object for both paths. Generation, quantization,
correctness, allocation of benchmark inputs, and transfers were outside the
timed regions. CUDA events were recorded on PyTorch's current CUDA stream.

Each of five rounds warmed both paths for 25 launches per path, synchronized,
then recorded 200 samples for each path. Measurement order alternated
`B→C, C→B, B→C, C→B, B→C`. Every round and raw sample is retained. Speedup is
the same-semantics ratio `baseline median / candidate median`.

The baseline numbers below are fresh, paired M4C measurements of the unchanged
M4A kernel. They are not substituted from the historical M4B JSON, because a
ratio across separate uncontrolled runs would not be the intended A/B result.

### Canonical Q/O rounds

All times are microseconds.

| Round | Order | B median | B mean | B min | B p95 | B std | C median | C mean | C min | C p95 | C std | Speedup |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | B→C | 88.064 | 88.310 | 88.064 | 89.088 | 0.590 | 69.632 | 69.268 | 68.608 | 69.632 | 0.701 | 1.264706x |
| 2 | C→B | 88.064 | 88.238 | 88.064 | 89.088 | 0.661 | 69.632 | 69.274 | 68.608 | 69.632 | 0.488 | 1.264706x |
| 3 | B→C | 88.064 | 88.192 | 87.040 | 89.088 | 0.368 | 68.608 | 69.043 | 68.608 | 69.632 | 0.651 | 1.283582x |
| 4 | C→B | 88.064 | 87.731 | 87.040 | 88.064 | 0.510 | 68.608 | 69.018 | 68.608 | 69.632 | 0.502 | 1.283582x |
| 5 | B→C | 88.064 | 87.669 | 87.040 | 88.064 | 0.507 | 61.440 | 62.440 | 60.416 | 68.608 | 2.627 | 1.433333x |

The fifth candidate round shifted to a faster timing level under uncontrolled
clocks. It is retained, not selected as the headline. The median across round
medians is 68.608 microseconds and the median round speedup is 1.283582x.

### Canonical K/V rounds

All times are microseconds.

| Round | Order | B median | B mean | B min | B p95 | B std | C median | C mean | C min | C p95 | C std | Speedup |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | B→C | 25.600 | 25.649 | 24.576 | 26.624 | 0.433 | 21.504 | 22.472 | 20.480 | 23.541 | 6.665 | 1.190476x |
| 2 | C→B | 25.600 | 25.673 | 24.576 | 26.624 | 0.643 | 21.504 | 21.720 | 20.480 | 22.528 | 3.764 | 1.190476x |
| 3 | B→C | 25.600 | 25.776 | 24.576 | 26.624 | 1.540 | 21.504 | 21.173 | 20.480 | 21.504 | 2.265 | 1.190476x |
| 4 | C→B | 25.600 | 25.770 | 24.576 | 26.624 | 1.923 | 21.504 | 21.556 | 20.480 | 22.528 | 2.461 | 1.190476x |
| 5 | B→C | 25.600 | 25.673 | 24.576 | 26.624 | 0.474 | 21.504 | 21.137 | 20.480 | 21.504 | 0.605 | 1.190476x |

### Aggregate and scaling results

The latency columns below are medians of the five per-round medians.

| Case | Baseline us | Candidate us | Median speedup | Min–max round speedup | Faster rounds |
|---|---:|---:|---:|---:|---:|
| Q/O M=1 N=3072 K=3072 | 88.064 | 68.608 | 1.283582x | 1.264706–1.433333x | 5/5 |
| K/V M=1 N=768 K=3072 | 25.600 | 21.504 | 1.190476x | 1.190476–1.190476x | 5/5 |
| Q/O M=2 N=3072 K=3072 | 149.504 | 116.736 | 1.280702x | 1.280702–1.280702x | 5/5 |
| Q/O M=4 N=3072 K=3072 | 291.840 | 227.328 | 1.283784x | 1.283784–1.283784x | 5/5 |
| Q/O M=8 N=3072 K=3072 | 577.536 | 450.560 | 1.281818x | 1.281818–1.283447x | 5/5 |
| M=1 N=3072 K=128 | 13.312 | 15.360 | 0.866667x | 0.789374–1.992708x | 1/5 |
| M=1 N=3072 K=512 | 20.480 | 18.432 | 1.111111x | 1.111111–1.111111x | 5/5 |
| M=1 N=3072 K=1024 | 31.744 | 26.624 | 1.192308x | 1.192308–1.192308x | 5/5 |

At K=3072, scaling M from 1 through 8 preserves a roughly 1.28x Q/O ratio,
so no larger-M pathology appears. The grouped organization loses at K=128,
where only eight of 32 groups have data but all 256 threads still pay the
reduction and subgroup-organization costs. One K=128 baseline round moved to a
30.608-microsecond median, causing the isolated 1.992708x ratio; the median
round result still correctly classifies the candidate as slower. K=512 and
K=1024 improve consistently. Secondary K behavior did not override the
decode-first primary result, and no second optimization was mixed in.

## Static resources and SASS

`cuobjdump --dump-resource-usage` on the normally built Release binary reports:

| Resource | Candidate |
|---|---:|
| Registers/thread | 29 |
| Static shared memory/block | 32 bytes |
| Local memory/thread | 0 bytes |
| Stack/thread | 0 bytes |
| Block size | 256 threads |

The device limit is 1,536 threads and 65,536 registers per SM. Six blocks use
`6 * 256 = 1,536` threads and a raw `6 * 256 * 29 = 44,544` registers. The raw
register limit alone permits eight blocks, while the thread limit permits six;
therefore the source/resource-derived result is six resident blocks, 48 warps,
and 100% **derived theoretical thread occupancy**. Allocation granularity does
not alter the thread-limit conclusion at 29 registers/thread. This is not an
achieved-occupancy claim.

Candidate SASS provides the following structural evidence:

| Stage | Static offsets / observation |
|---|---|
| Gamma | one `LDG.E` at `0x02f0` |
| Scale leader | predicated path to one `LDG.E.U8` at `0x0550` and UE4M3 decode through `0x0880` |
| Eight-lane broadcast | `SHFL.IDX` at `0x0910`; width/mask lowering remains subgroup-scoped |
| Packed payload | one `LDG.E.U8` at `0x0920`, followed by low/high E2M1 decode paths |
| Activations | two `LDG.E.U16` sites at `0x0d90` and `0x0da0` |
| Reconstruction/products | FP32 `FMUL` at `0x0dc0`, `0x0e00`, `0x0e70`, `0x0e80`, `0x0ec0`, and `0x0ee0` |
| Private accumulation | FP32 `FADD` at `0x0ed0` then `0x0ef0` |
| Reduction | shuffle/add stages at `0x0f50–0x1150`, with two block barriers |
| Output | BF16 RNE pack at `0x11e0`, `STG.E.U16` at `0x11f0` |

Across the full static candidate section, including helper and mutually
exclusive paths, there are 5 LDG, 1 STG, 13 SHFL, 14 FMUL, 19 FADD, 0 FFMA,
7 MUFU, 113 IMAD, 30 IADD3, 51 branch, and 2 barrier sites. These are static
opcode occurrences, not executed counts. In particular, the unchanged count
of static MUFU helper sites says nothing about the intended K-to-K/16 dynamic
scale-decode reduction.

## Retention criteria

| Criterion | Evidence | Result |
|---|---|:---:|
| Candidate/reference adjacency <= 1 | maximum observed 0 in M4C tests and benchmark cases | Pass |
| All tests | targeted and both full invocations passed | Pass |
| Candidate memcheck | zero errors | Pass |
| Q/O faster in clear majority | 5/5 rounds | Pass |
| K/V faster in clear majority | 5/5 rounds | Pass |
| At least one canonical case >=5% | both cases exceeded 5% | Pass |
| Other canonical case has no meaningful regression | K/V median ratio 1.190476x | Pass |
| No catastrophic resource loss | thread-limit-derived 100% theoretical occupancy | Pass |
| No K=3072 pathology | minimum measured K=3072 median ratio 1.190476x | Pass |
| One optimization family only | packed-pair consumption plus per-block scale broadcast | Pass |

The K=128 regression is material but limited to the secondary short-K regime;
it is not a catastrophic canonical regression. The candidate therefore meets
the stated M4C policy and is **RETAINED** as an isolated experimental path.

## Self-review checklist

| # | Review question | Answer and evidence |
|---:|---|---|
| 1 | Still computes `X @ W.T`? | Yes; orientation exact test and canonical oracle guards pass. |
| 2 | One output scalar per CUDA block? | Yes; `blockIdx.x` is one flattened `(m,n)` output. |
| 3 | Block size still 256? | Yes; launch constant and resource record are 256. |
| 4 | 32 groups of eight without crossing warps? | Yes; contiguous division/modulo mapping, with four groups per warp. |
| 5 | One 16-weight block per group iteration? | Yes; block `b` maps to bytes `8b..8b+7` and K `16b..16b+15`. |
| 6 | Beta loaded/decoded once per iteration? | Yes; only subgroup lane zero executes the scale load/decode path. |
| 7 | Beta broadcast inside exactly eight lanes? | Yes; active group mask, source lane zero, width eight. |
| 8 | One packed load per two weights? | Yes; each lane performs one U8 load and decodes both nibbles. |
| 9 | Even/odd nibble semantics unchanged? | Yes; low is even and high is odd; isolation tests pass. |
| 10 | E2M1 decoding unchanged? | Yes; audited logic is duplicated and all 16 codes pass. |
| 11 | UE4M3 decoding unchanged? | Yes; audited logic is duplicated and codes `0x00..0x7e` pass. |
| 12 | `(q*beta)*gamma` order unchanged? | Yes; two explicit `__fmul_rn` calls per weight. |
| 13 | Activation products FP32? | Yes; BF16 is promoted before explicit FP32 multiply. |
| 14 | Private accumulation explicit FP32 addition? | Yes; even then odd `__fadd_rn`. |
| 15 | Output BF16 RNE? | Yes; `__float2bfloat16_rn`. |
| 16 | Full reduction safe for K=16? | Yes; exact K=16 test and memcheck pass with 248 zero contributors. |
| 17 | K=3072 correct? | Yes; canonical Q/KV and M>1 guards have max adjacency zero here. |
| 18 | Current-stream execution works? | Yes; non-default prepare → kernel → consumer test is exact. |
| 19 | No FP32 W materialization? | Yes; the candidate consumes packed/scaled storage directly. |
| 20 | Baseline source unchanged? | Yes; SHA-256 matches `ee6f7ef`. |
| 21 | No other optimization family added? | Yes; only grouped scale decode and paired-nibble consumption. |
| 22 | Candidate sanitizer-clean? | Yes; memcheck reports zero errors. |
| 23 | A/B uses identical inputs? | Yes; both callables close over the same four tensor objects. |
| 24 | Alternating order controls order effects? | Yes; five rounds use B→C, C→B, B→C, C→B, B→C. |
| 25 | Decision follows evidence? | Yes; all stated gates pass, while the K=128 loss remains reported. |

## Limitations

- One RTX 4080 Laptop GPU was measured under uncontrolled clocks and power.
- Raw event samples contain timing-level shifts and short-case outliers; no
  best round was selected.
- Nsight performance-counter access remains unavailable. No achieved
  occupancy, SOL, DRAM bandwidth, L2 hit rate, pipeline utilization, or warp
  stall claim is made.
- Source accounting and static SASS sites are not dynamic instruction counts
  or measured memory traffic.
- Deterministic random inputs are not a trained checkpoint.
- Retention does not promote the candidate to `cuda_w4a16_linear`; a later
  milestone can make that separate decision.

## Reproduction

```bash
./scripts/build_cuda.sh
.venv/bin/python -m pytest -q tests/test_cuda_w4a16_grouped.py
compute-sanitizer --tool memcheck --error-exitcode 99 \
    .venv/bin/python scripts/validate_cuda_w4a16_grouped.py
compute-sanitizer --tool memcheck --error-exitcode 99 \
    .venv/bin/python scripts/validate_cuda_w4a16.py
.venv/bin/pytest -q
.venv/bin/python -m pytest -q
.venv/bin/python benchmarks/benchmark_w4a16_grouped_decode.py \
    --warmups 25 \
    --repetitions 200 \
    --external-validation-passed
```
