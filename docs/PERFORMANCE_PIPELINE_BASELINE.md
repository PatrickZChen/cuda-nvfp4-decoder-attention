# M6 modular decoder-pipeline performance baseline

Milestone 6 is a measurement-only baseline for the complete modular `T=1`
decoder-attention path at repository commit
`8aa04d062b6ab59d916619561c12d0e62c645592`. It does not change or optimize a
production CUDA implementation. Complete raw samples, fixture statistics, and
machine-readable accounting are in
[`m6_decoder_pipeline_baseline.json`](../benchmarks/results/rtx4080-laptop-sm89/m6_decoder_pipeline_baseline.json).

The main result is regime-dependent. Projection is the largest isolated
stage-latency category for `P=0,128,512`; cached GQA is largest for
`P=2048,8192`. At `P=8192`, cached GQA accounts for 77.2% of the isolated-stage
sum at `B=1` and 81.8% at `B=2`. This is stage-level latency attribution, not a
hardware-counter bottleneck classification.

## Exact operation and boundary

Every primary end-to-end sample calls exactly one public production operation:

```python
decoder_attention_cuda.cuda_decoder_attention_forward_(...)
```

All four Q/K/V/O projections use the frozen baseline
`cuda_primitives.cuda_w4a16_linear`. The M4C
`cuda_w4a16_linear_grouped_decode` candidate is not selected, substituted, or
used to estimate an end-to-end speedup.

The production sequence remains:

1. input RMSNorm;
2. baseline Q/K/V W4A16 projections;
3. metadata-only head reshapes;
4. Q/K per-head RMSNorm;
5. Q/K RoPE;
6. in-place K/V cache append;
7. capacity-aware cached GQA;
8. metadata-only context flatten;
9. baseline output W4A16 projection.

Weight generation, quantization, fixture creation, transfers, correctness
references, cache establishment, warmup, stage-state preparation, environment
inspection, printing, and JSON serialization are outside timed intervals. The
existing public operations still allocate their normal outputs; M6 does not add
allocation-free variants.

## Capture environment

The capture began at `2026-08-18T00:29:34Z` under uncontrolled consumer-laptop
conditions.

| Item | Captured value |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU |
| Compute capability | 8.9, Ada SM89 |
| SMs | 58 |
| Device memory | 12,878,086,144 bytes |
| Reported L2 capacity | 50,331,648 bytes |
| Driver | 555.97 |
| Python | 3.12.3 |
| PyTorch | 2.6.0+cu124 |
| PyTorch CUDA build | 12.4 (`12040`) |
| CUDA toolkit / nvcc | 12.5 / V12.5.82 |
| Platform | Linux 6.6.87.2 Microsoft WSL2, x86-64 |

The read-only pre-capture `nvidia-smi` snapshot reported 48 C, 2,280 MHz
graphics, 9,100 MHz memory, and 36.11 W draw. Its live power-limit query was
unavailable; default and maximum were 150 W and 175 W. The post-capture snapshot
reported 63 C, 2,520 MHz graphics, 9,100 MHz memory, and 161.82 W. Clocks,
power, driver, registry, Windows, and WSL settings were not changed.

## Controlled representative fixture

Every case uses `H=3072`, `Hq=24`, `Hkv=6`, `D=128`, and `T=1`. The primary
matrix is exactly the Cartesian product of `B={1,2}` and
`P={0,128,512,2048,8192}`. Physical cache capacity is fixed at `C=16384` in all
ten cases, so changing `P` changes logical `S=P+1` without changing the physical
cache stride or cache allocation.

The B=2 master input is deterministic BF16 `Normal(0,1)` with seed 62001; B=1
uses batch 0 of that same tensor. Input, Q-head, and K-head gamma vectors are
BF16 `1 + Normal(0,0.02)` with seeds 62002, 62003, and 62004 and are shared by
all cases.

The deterministic CPU master caches have shape `[2,6,16384,128]`. K is BF16
`Normal(0,0.5)` with seed 62005 and V is BF16 `Normal(0,0.75)` with seed 62006.
B=1 uses master batch 0. The unused suffix remains populated, but cached GQA
logically reads only `S`. Each timing case begins from the corresponding master
slice.

At fixed `P`, every invocation recomputes and overwrites exactly cache slot `P`.
The prefix `0:P` is unchanged. Slot `P` is restored before each benchmark round,
outside timing; no full-cache reset, clone, or copy occurs inside a timed region.
Thus every repeated call sees the same logical prefix and current hidden state.

### Representative NVFP4 weights

The four source weights are independent, dense FP32 normal matrices generated
on CPU with `source_std = 1/sqrt(3072) = 0.0180421959`. They are quantized on CPU
by the frozen `quantize_nvfp4_reference(...)`; only the resulting portable
NVFP4 storage is transferred to CUDA. These are deterministic representative
random weights, not trained-model weights.

| Weight | Shape | Seed | Actual source std | Global decode scale | Zero block scales | Unique scale bytes | E2M1 zero-code fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q | `[3072,3072]` | 61001 | 0.01804179 | 3.627656e-5 | 0 | 26 | 0.067958 |
| K | `[768,3072]` | 61002 | 0.01804142 | 3.621574e-5 | 0 | 25 | 0.067858 |
| V | `[768,3072]` | 61003 | 0.01802574 | 3.392520e-5 | 0 | 26 | 0.067882 |
| O | `[3072,3072]` | 61004 | 0.01804162 | 3.686411e-5 | 0 | 28 | 0.067978 |

All 16 E2M1 nibble codes occur in every weight. Populated UE4M3 exponent bins
are 12–15 with non-uniform counts; for example, Q has counts 42, 58,240,
506,136, and 25,406. The JSON retains the complete E2M1 nibble histogram,
UE4M3 exponent histogram, nonzero scale-byte histogram, source statistics, and
portable-storage SHA-256 for each weight. This rules out the earlier one-hot or
uniform-scale fixtures.

## Correctness before timing

All ten prechecks completed before the first timed sample. For each case they
verified output shape `[B,1,3072]`, BF16 dtype, finite output, unchanged K/V
cache data pointers, bitwise deterministic repeated output, and bitwise
deterministic repeated current K/V slots.

The exact representative B=1, P=128 fixture was also compared untimed with
`decoder_attention_nvfp4_reference(...)` using the same quantized weights and
master cache prefix:

| Metric | Result |
|---|---:|
| Maximum absolute error | 0 |
| Mean absolute error | 0 |
| Exact BF16 fraction | 1.0 |
| Maximum BF16 adjacency distance | 0 |

It passes the frozen elementwise policy of at most one adjacent BF16 value or
absolute error at most `2^-20`.

## Timing methods

### End-to-end stream elapsed time

For each case and each of five rounds, the harness restores slot P outside
timing, executes 25 untimed warmups, synchronizes once, pre-creates 100 CUDA
event pairs, and records start/call/end for 100 individual public pipeline
calls on the PyTorch current stream. It synchronizes only the final queued end
event and extracts all samples afterward.

`end_to_end_stream_elapsed_us` is the CUDA stream interval between those
events. Since Python launches multiple kernels, it can include stream-visible
idle gaps caused by host submission if the GPU catches up. It is not pure GPU
arithmetic time, host-free kernel time, or the sum of kernel durations.

Every round records median, mean, minimum, linearly interpolated p95, population
standard deviation, and all 100 raw samples. Across rounds the reported value
is the median of round medians, accompanied by the min/max round median and
their ratio. No outlier is filtered and no fastest round is selected.

### Synchronized wall time

After 25 warmups, each of 30 wall samples synchronizes, starts
`time.perf_counter_ns()`, invokes one public call, synchronizes, and stops the
timer. `synchronized_wall_us` therefore includes Python orchestration,
validation, allocator behavior, kernel submission, GPU work, and
synchronization. Wall-minus-stream is not called host launch overhead.

### Isolated logical stages

For each B/P case, existing public primitives first create an untimed valid
snapshot of `x_norm`, projected heads, normalized heads, rotated heads, an
already-appended cache, and `context_flat`. Each of the eleven stages is then
timed independently for three rounds of 50 raw CUDA-event samples, with 25
warmups before every round. A stage consumes only its precomputed prerequisite;
prerequisite stages are not inside its interval. The existing stage call may
allocate its normal output.

Head reshapes and context flatten are separately recorded as metadata-only
views with zero source-derived kernel launches. Cached GQA remains one logical
stage even though its implementation launches QK, softmax, and PV kernels.

## Primary latency and stability

Times are microseconds. Stream latency is the median of five round medians.
Spread is `max_round_median / min_round_median - 1`.

| B | P | S | Stream median | Synchronized wall median | Round-median spread |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | 266.240 | 287.164 | 0.3861% |
| 1 | 128 | 129 | 250.880 | 330.430 | 0.4082% |
| 1 | 512 | 513 | 294.912 | 344.385 | 0.3472% |
| 1 | 2048 | 2049 | 489.472 | 557.265 | 0.2092% |
| 1 | 8192 | 8193 | 1215.488 | 1304.650 | 0.0843% |
| 2 | 0 | 1 | 425.984 | 489.323 | 0.0000% |
| 2 | 128 | 129 | 450.560 | 514.277 | 0.2273% |
| 2 | 512 | 513 | 538.624 | 594.784 | 0.0000% |
| 2 | 2048 | 2049 | 938.496 | 1008.348 | 0.1638% |
| 2 | 8192 | 8193 | 2487.808 | 2641.708 | 0.3922% |

All cases receive the descriptive `low_round_median_spread` label. The B=1
P=128 stream median is lower than P=0 despite more attention work. It is
retained as a cross-case timing-level effect; no monotonic fit or corrective
rerun is applied. Wall samples are visibly noisier in several short cases, and
their full distributions remain in the JSON.

## Isolated logical-stage medians

Each value is the median of three per-round medians, in microseconds.

| Case | Input norm | Q proj | K proj | V proj | Q norm | K norm | Q RoPE | K RoPE | Append | Cached GQA | O proj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 P0 | 10.928 | 77.824 | 25.600 | 25.600 | 20.576 | 13.312 | 14.848 | 5.600 | 5.120 | 22.016 | 77.824 |
| B1 P128 | 14.528 | 77.824 | 25.600 | 25.600 | 10.752 | 9.728 | 12.320 | 15.360 | 6.144 | 32.768 | 77.824 |
| B1 P512 | 16.384 | 77.824 | 25.600 | 25.600 | 11.344 | 14.448 | 14.448 | 5.968 | 12.896 | 75.776 | 77.824 |
| B1 P2048 | 20.464 | 77.824 | 25.600 | 25.600 | 7.792 | 7.680 | 16.032 | 22.720 | 6.144 | 269.312 | 77.824 |
| B1 P8192 | 12.288 | 77.824 | 25.600 | 25.600 | 12.720 | 9.200 | 22.160 | 20.816 | 8.688 | 993.792 | 77.824 |
| B2 P0 | 10.848 | 149.504 | 43.008 | 43.008 | 14.336 | 14.336 | 13.824 | 15.072 | 5.152 | 29.696 | 149.504 |
| B2 P128 | 7.168 | 149.504 | 43.008 | 43.008 | 11.968 | 14.848 | 9.072 | 11.920 | 5.120 | 54.272 | 149.504 |
| B2 P512 | 11.264 | 149.504 | 43.008 | 43.008 | 6.480 | 19.456 | 10.752 | 12.384 | 6.144 | 141.312 | 149.504 |
| B2 P2048 | 11.264 | 150.528 | 43.008 | 43.008 | 13.280 | 18.768 | 15.248 | 17.408 | 6.000 | 542.720 | 149.504 |
| B2 P8192 | 9.312 | 149.504 | 43.008 | 43.008 | 13.120 | 13.312 | 14.944 | 12.640 | 5.120 | 2043.904 | 149.504 |

At short context, Q and output projections tie as the largest individual
logical stages except for the 1.024-microsecond Q difference at B2/P2048.
Cached GQA is the largest logical stage at P=2048 and P=8192 for both batch
sizes.

## Category attribution and non-additivity

Parentheses give fraction of the isolated-stage sum.

| Case | Projection | Normalization/RoPE | Cache append | Attention | Isolated sum | Sum / stream E2E |
|---|---:|---:|---:|---:|---:|---:|
| B1 P0 | 206.848 (69.1%) | 65.264 (21.8%) | 5.120 (1.7%) | 22.016 (7.4%) | 299.248 | 1.1240 |
| B1 P128 | 206.848 (67.1%) | 62.688 (20.3%) | 6.144 (2.0%) | 32.768 (10.6%) | 308.448 | 1.2295 |
| B1 P512 | 206.848 (57.8%) | 62.592 (17.5%) | 12.896 (3.6%) | 75.776 (21.2%) | 358.112 | 1.2143 |
| B1 P2048 | 206.848 (37.1%) | 74.688 (13.4%) | 6.144 (1.1%) | 269.312 (48.4%) | 556.992 | 1.1379 |
| B1 P8192 | 206.848 (16.1%) | 77.184 (6.0%) | 8.688 (0.7%) | 993.792 (77.2%) | 1286.512 | 1.0584 |
| B2 P0 | 385.024 (78.9%) | 68.416 (14.0%) | 5.152 (1.1%) | 29.696 (6.1%) | 488.288 | 1.1463 |
| B2 P128 | 385.024 (77.1%) | 54.976 (11.0%) | 5.120 (1.0%) | 54.272 (10.9%) | 499.392 | 1.1084 |
| B2 P512 | 385.024 (64.9%) | 60.336 (10.2%) | 6.144 (1.0%) | 141.312 (23.8%) | 592.816 | 1.1006 |
| B2 P2048 | 386.048 (38.2%) | 75.968 (7.5%) | 6.000 (0.6%) | 542.720 (53.7%) | 1010.736 | 1.0770 |
| B2 P8192 | 385.024 (15.4%) | 63.328 (2.5%) | 5.120 (0.2%) | 2043.904 (81.8%) | 2497.376 | 1.0038 |

The sum is intentionally not forced to equal end-to-end latency. Isolated
calls have different allocation reuse, Python submission timing, event
placement, cache/thermal state, and inter-stage context. No residual stage is
fabricated. Category fractions therefore describe the isolated sum, not a
partition of end-to-end time.

## P and B scaling

For B=1, stream latency relative to P=0 is 0.942x, 1.108x, 1.838x, and 4.565x
at P=128, 512, 2048, and 8192. Cached-GQA isolated latency rises from 22.016 to
993.792 microseconds, while the projection category remains 206.848
microseconds. The attention fraction rises from 7.4% to 77.2%.

For B=2, stream latency relative to P=0 is 1.058x, 1.264x, 2.203x, and 5.840x.
Cached GQA rises from 29.696 to 2043.904 microseconds; projection stays within
385.024–386.048 microseconds. The attention fraction rises from 6.1% to 81.8%.

B=2/B=1 ratios are not assumed to be exactly two:

| P | Stream E2E ratio | Projection ratio | Cached-GQA ratio | Temporary-attention-byte ratio | Persistent-cache-byte ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.6000 | 1.8614 | 1.3488 | 2.0 | 2.0 |
| 128 | 1.7959 | 1.8614 | 1.6563 | 2.0 | 2.0 |
| 512 | 1.8264 | 1.8614 | 1.8649 | 2.0 | 2.0 |
| 2048 | 1.9174 | 1.8663 | 2.0152 | 2.0 | 2.0 |
| 8192 | 2.0468 | 1.8614 | 2.0567 | 2.0 | 2.0 |

The simple descriptive result is enough: fixed-size projections dominate short
contexts, while the present materialized cached-GQA implementation grows with
logical context and dominates the two long-context points. No more elaborate
complexity fit is claimed.

## Launch accounting

Source audit confirms 13 CUDA kernel launches per production call:

| Production portion | Launches |
|---|---:|
| Input RMSNorm | 1 |
| Q/K/V baseline W4A16 | 3 |
| Q/K RMSNorm | 2 |
| Q/K RoPE | 2 |
| KV append | 1 |
| Cached-GQA QK / softmax / PV | 3 |
| Output baseline W4A16 | 1 |
| Total | 13 |

This is explicitly a **source-derived launch count**. A read-only
`torch.profiler` attempt completed but emitted no CUDA activity records in this
WSL environment, so dynamic launch validation is recorded as unavailable. No
profiler timing replaces the event baseline.

## Allocation and memory accounting

Persistent packed Q/K/V/O weight storage is 13,271,056 bytes. Production also
holds x and three norm weights plus two BF16 physical caches. The cache bytes
are separate from temporary attention materialization:

```text
scores_bytes = B * Hq * T * S * sizeof(float)
probabilities_bytes = scores_bytes
kv_cache_bytes = 2 * B * Hkv * C * D * sizeof(BF16)
```

| Case | Scores bytes | Probabilities bytes | Combined attention materialization | Persistent K/V cache bytes | Incremental allocator peak |
|---|---:|---:|---:|---:|---:|
| B1 P0 | 96 | 96 | 192 | 50,331,648 | 43,008 |
| B1 P128 | 12,384 | 12,384 | 24,768 | 50,331,648 | 62,464 |
| B1 P512 | 49,248 | 49,248 | 98,496 | 50,331,648 | 136,192 |
| B1 P2048 | 196,704 | 196,704 | 393,408 | 50,331,648 | 431,104 |
| B1 P8192 | 786,528 | 786,528 | 1,573,056 | 50,331,648 | 1,610,752 |
| B2 P0 | 192 | 192 | 384 | 100,663,296 | 86,016 |
| B2 P128 | 24,768 | 24,768 | 49,536 | 100,663,296 | 123,904 |
| B2 P512 | 98,496 | 98,496 | 196,992 | 100,663,296 | 271,360 |
| B2 P2048 | 393,408 | 393,408 | 786,816 | 100,663,296 | 861,184 |
| B2 P8192 | 1,573,056 | 1,573,056 | 3,146,112 | 100,663,296 | 3,220,480 |

The measured peak is `torch.cuda.max_memory_allocated()` above an already
created, allocator-warmed persistent fixture while retaining the returned
output. It is a PyTorch caching-allocator metric, not exact physical VRAM,
traffic, bandwidth, or a physical `cudaMalloc` count.

At source level, allocated intermediates are the input RMSNorm output, Q/K/V
projection outputs, Q/K norm outputs, Q/K RoPE outputs, FP32 scores, FP32
probabilities, BF16 context, and final BF16 output. Head reshapes and context
flatten are views; KV append mutates persistent storage.

## Post-capture validation and frozen-source proof

Both required regression commands completed successfully after the result was
captured:

```text
.venv/bin/pytest -q:           461 passed, 14 skipped
.venv/bin/python -m pytest -q: 461 passed, 14 skipped
```

An explicit working-tree-versus-HEAD audit reports zero diff for
`decoder_attention_cuda.py`, `reference/decoder_attention_nvfp4.py`, all CUDA
primitive headers and sources, and every existing test. The prior result files
also remain byte-identical to HEAD:

| Frozen result | SHA-256 |
|---|---|
| `baseline_w4a16.json` | `e6b74b562fa669491da29b265bf05233e32c3b4793a280b12fb87b168355fbd9` |
| `m4c_grouped_decode.json` | `fbd22a0c7421466e1fb293497a5cd05ad314f5604a08441d9ed60bd6dc0262be` |

The result JSON also stores working-tree and HEAD hashes for the frozen
production files, all tests, and both prior result artifacts; every comparison
is true.

## Interpretation limits

- One consumer laptop GPU and one capture are represented.
- The path is allocator- and kernel-warmed; this is not cold-start latency.
- All timing levels and outliers remain in the JSON.
- CUDA-event intervals can contain host-submission-induced stream idle gaps.
- Wall-versus-stream deltas are not isolated host overhead.
- Isolated-stage values are non-additive.
- Logical bytes and allocator peaks are not measured traffic or bandwidth.
- No Nsight Compute counters were collected. M6 does not claim DRAM-, L2-,
  compute-, instruction-, SFU-, launch-, or occupancy-bound hardware behavior.
- Representative random weights are not trained data.
- No cross-system comparison is made to native Blackwell FP4, FlashAttention,
  cuBLAS, TensorRT-LLM, or another model/server because semantics are not
  matched.

## M7 recommendation

For M7, investigate one isolated, same-semantics cached-attention
algorithm/materialization optimization, with the first goal of reducing or
eliminating the separate FP32 score and probability materializations. Evidence
is strongest at P=2048 and P=8192, where attention is the largest isolated
category for both batch sizes; at P=8192 it measures 993.792 microseconds for
B=1 and 2043.904 microseconds for B=2.

Short contexts remain projection-led, so any grouped-decode promotion should
be a separate controlled whole-pipeline A/B rather than inferred from M4C. M6
does not implement either recommendation.

## Reproduction

From the repository root with the existing CUDA extension and environment:

```bash
.venv/bin/python benchmarks/benchmark_decoder_pipeline.py
```

The defaults enforce 25 warmups, five 100-sample end-to-end rounds, 30
synchronized-wall samples, and three 50-sample rounds for every isolated
stage. The result writer refuses to overwrite either M4 JSON artifact.
