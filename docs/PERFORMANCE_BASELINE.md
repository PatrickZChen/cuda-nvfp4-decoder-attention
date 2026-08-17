# M4B direct W4A16 performance baseline

This document records the measurement-only Milestone 4B baseline for the
correctness-first direct Ada W4A16 projection introduced at commit
`ee6f7efeae04cd504dd879ac4d147b09e600778a`. No CUDA kernel, launch geometry,
numerical contract, or architecture contract was changed for this work.

The terms used below are deliberate:

- **Measured** means a value came from CUDA events or a tool that executed on
  the target GPU.
- **Derived** means a value was calculated from frozen source, binary
  resources, tensor formats, or device properties.
- **Inferred** means an engineering interpretation of measured and derived
  evidence. It is not presented as a hardware counter.

The complete event samples and environment are in
[`baseline_w4a16.json`](../benchmarks/results/rtx4080-laptop-sm89/baseline_w4a16.json).
The profiler access result, static binary facts, and unavailable metrics are in
[`ncu_profile_status.json`](../benchmarks/results/rtx4080-laptop-sm89/ncu_profile_status.json).

## Environment

The baseline was captured on 2026-08-17 under uncontrolled consumer-laptop
conditions.

| Item | Recorded value |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU |
| Architecture | Ada, SM89, compute capability 8.9 |
| SM count | 58 |
| Device memory | 12,878,086,144 bytes |
| Reported L2 capacity | 50,331,648 bytes (48 MiB) |
| NVIDIA driver | 555.97 |
| Power reporting | WSL live-limit query unavailable; default 150 W, maximum 175 W; a separate `nvidia-smi` snapshot displayed a 175 W cap |
| OS | Linux 6.6.87.2 Microsoft WSL2, x86-64 |
| Python | 3.12.3 |
| PyTorch | 2.6.0+cu124 |
| PyTorch CUDA build | 12.4 |
| CUDA toolkit / nvcc | 12.5 / V12.5.82 |
| Nsight Compute | 2024.2.1.0, build 34372528 |
| Extension build | Release, `-O3`, SM89, no `-lineinfo` |

GPU clocks, the power limit, driver settings, and Windows/WSL settings were not
changed. These results are therefore a reproducible project baseline, not a
datacenter-grade deterministic measurement. No contextual BF16 or FP32
`torch.matmul` was measured: those paths have different precision and storage
semantics and would not be same-semantics competitors.

## Reproduction and timing method

Run the event benchmark from the repository root:

```bash
.venv/bin/python benchmarks/benchmark_w4a16.py \
    --warmups 25 \
    --repetitions 200 \
    --diagnostic-repetitions 50
```

The harness performs the following for every primary direct case:

1. Generate deterministic inputs and quantize the source weight outside the
   timed region.
2. Compare the complete output against `w4a16_linear_reference` outside the
   timed region. Every recorded case passed the frozen M4A tolerance of at most
   one adjacent BF16 value.
3. Execute 25 warmup launches and synchronize once.
4. Pre-create 200 start/end CUDA-event pairs. For each sample, record the start
   event, call the direct operation on the same current stream, and record the
   end event.
5. Synchronize the final end event once, then obtain all elapsed times.

The median, arithmetic mean, minimum, and linearly interpolated p95 are reported.
The JSON retains every raw event sample. Python setup, random generation,
quantization, input allocation, H2D copies, extension loading, compilation, and
the correctness oracle are excluded. The public operation creates its output
tensor on each call, but Python dispatch and host allocator work are outside the
CUDA-event interval; warmup also primes the caching allocator.

For controlled M scaling, the harness repeats one deterministic activation row.
Only the logical row count, output grid, and repeated kernel work change.

## Benchmark matrix

The matrix is intentionally not a Cartesian product:

| Series | M | N | K |
|---|---:|---:|---:|
| Canonical Q/O decode | 1 | 3072 | 3072 |
| Canonical K/V decode | 1 | 768 | 3072 |
| N scaling | 1 | 256, 768, 1536, 3072 | 3072 |
| K scaling | 1 | 3072 | 128, 512, 1024, 3072 |
| Q/O M scaling | 1, 2, 4, 8, 16 | 3072 | 3072 |
| K/V M scaling | 1, 2, 4, 8, 16 | 768 | 3072 |

## Measured direct baseline

Times are microseconds and are rounded to 0.1 microsecond here. The JSON keeps
the event values to three decimal places for machine analysis; the extra digits
should not be interpreted as environmental stability.

| Case | M | N | K | Median | Mean | Min | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Canonical Q/O | 1 | 3072 | 3072 | 86.0 | 85.6 | 85.0 | 86.0 |
| Canonical K/V | 1 | 768 | 3072 | 28.7 | 29.9 | 26.7 | 28.7 |
| N scaling | 1 | 256 | 3072 | 13.3 | 15.1 | 12.3 | 28.7 |
| N scaling | 1 | 1536 | 3072 | 46.1 | 46.5 | 46.1 | 47.1 |
| K scaling | 1 | 3072 | 128 | 14.3 | 15.5 | 13.3 | 19.5 |
| K scaling | 1 | 3072 | 512 | 21.5 | 21.6 | 20.5 | 22.5 |
| K scaling | 1 | 3072 | 1024 | 34.8 | 34.5 | 33.8 | 34.8 |
| Q/O M scaling | 2 | 3072 | 3072 | 163.8 | 163.6 | 162.8 | 163.8 |
| Q/O M scaling | 4 | 3072 | 3072 | 289.8 | 300.4 | 288.8 | 320.5 |
| Q/O M scaling | 8 | 3072 | 3072 | 575.5 | 574.9 | 571.4 | 576.5 |
| Q/O M scaling | 16 | 3072 | 3072 | 1144.8 | 1145.3 | 1143.8 | 1145.9 |
| K/V M scaling | 2 | 768 | 3072 | 43.0 | 42.9 | 42.0 | 44.0 |
| K/V M scaling | 4 | 768 | 3072 | 78.8 | 78.5 | 77.8 | 78.8 |
| K/V M scaling | 8 | 768 | 3072 | 149.5 | 149.9 | 149.5 | 150.5 |
| K/V M scaling | 16 | 768 | 3072 | 292.9 | 292.8 | 291.8 | 292.9 |

### Scaling observations

**Measured:** at fixed M=1 and N=3072, the median rises from 14.3 microseconds
at K=128 to 21.5 at K=512, 34.8 at K=1024, and 86.0 at K=3072. The long-K
region is approximately linear, while K=128 retains a larger fixed launch and
reduction component.

**Measured:** at fixed M=1 and K=3072, the medians are 13.3, 28.7, 46.1, and
86.0 microseconds for N=256, 768, 1536, and 3072. N>=768 trends approximately
with output count. N=256 is a short, variable launch and should not be
characterized from its minimum alone.

M scaling is close to linear once the grid is large enough, but per-row cost
falls as the short-grid effect is amortized:

| M | Q/O median | Q/O median per row | K/V median | K/V median per row |
|---:|---:|---:|---:|---:|
| 1 | 86.0 | 86.0 | 28.7 | 28.7 |
| 2 | 163.8 | 81.9 | 43.0 | 21.5 |
| 4 | 289.8 | 72.4 | 78.8 | 19.7 |
| 8 | 575.5 | 71.9 | 149.5 | 18.7 |
| 16 | 1144.8 | 71.6 | 292.9 | 18.3 |

**Measured:** M=16 reduces median per-row cost by 16.8% for Q/O and 36.2% for
K/V relative to M=1. **Inferred:** M=1 is materially affected by the number of
available blocks/waves, especially for the smaller K/V grid; larger M does not
change the per-output algorithm.

### Variability

The canonical Q/O case is stable in this run: p95-minus-min is 1.0 microsecond,
or about 1.2% of the median. Canonical K/V has a 2.0-microsecond min-to-p95
spread, but three of 200 raw samples exceeded twice its median and the largest
was 178.2 microseconds. Those sparse events raise the mean above p95. They are
retained in the JSON.

The highest sustained variability appears in short cases. N=256 has a
12.3-microsecond minimum and 28.7-microsecond p95 around a 13.3-microsecond
median; K=128 has a 13.3 minimum and 19.5 p95. Q/O M=4 also has a visible
second timing level (288.8 minimum, 320.5 p95). No minimum was substituted for
the median and no system settings were changed to suppress these effects.

## Contextual dequantization result

The standalone CUDA NVFP4 dequantizer was measured separately. It materializes
FP32 `W_hat` and is not a projection implementation or a speedup baseline.

| Case | FP32 output size | Median | Mean | Min | p95 |
|---|---:|---:|---:|---:|---:|
| N=3072, K=3072 | 37,748,736 bytes | 87.0 | 87.5 | 85.0 | 89.1 |
| N=768, K=3072 | 9,437,184 bytes | 20.5 | 22.2 | 19.4 | 37.9 |

The Q/O dequantization median happens to be close to the direct projection
median, but the operations, output storage, and semantics differ. Their ratio
is not a same-semantics speedup.

## Logical storage and controlled diagnostics

For canonical Q/O, the unique logical storage footprint is 5,320,708 bytes:
4,718,592 packed E2M1 payload bytes, 589,824 UE4M3 scale bytes, one four-byte
global scale, 6,144 activation bytes, and 6,144 output bytes. Canonical K/V is
1,334,788 bytes. These are format-derived storage footprints, not measured
traffic.

At source level, each output block independently issues a packed-byte load, a
scale-byte load, and a BF16 activation load for every logical K element. This
derives 37,767,168 scalar load/store bytes for canonical Q/O and 9,441,792 for
canonical K/V before considering transactions or caches. In particular, the
canonical Q/O source requests 18,874,368 activation bytes from a unique 6,144-
byte activation row. These values must not be called DRAM bytes or DRAM
bandwidth.

Two event-timed diagnostics were kept separate from the primary matrix:

| Diagnostic | Median | Min | p95 | Interpretation |
|---|---:|---:|---:|---|
| Uniform normal UE4M3 code `0x70` | 77.8 | 77.8 | 78.8 | Modal normal-exponent class; uses the normal `ldexpf` lowering |
| Uniform exponent-zero UE4M3 code `0x07` | 69.6 | 69.6 | 70.7 | Takes the exponent-zero arithmetic branch |
| Q/O after a same-stream read of 2x reported L2 | 96.3 | 95.2 | 97.3 | Compared with 86.0-microsecond warm primary median |
| K/V after a same-stream read of 2x reported L2 | 33.8 | 32.8 | 34.8 | Compared with 28.7-microsecond warm primary median |

The UE4M3 A/B fixes shape, packed payload, activation, global scale, and kernel;
only the uniform scale code differs. The normal path costs 8.2 microseconds
more, 10.5% of the normal-path latency in this artificial case. The canonical
weight contains 589,824 scale codes: 85.5% use exponent 14, 10.4% exponent 13,
4.1% exponent 15, and 0.005% exponent 12; none use exponent zero. This supports
a meaningful, secondary normal-scale decode cost. It does not isolate all
weight-decode work.

The cache diagnostic reads a 100,663,296-byte uint8 buffer through `torch.sum`
on the same stream before recording each start event. The direct launch alone
is inside the timed interval. It raises Q/O median by 10.2 microseconds (11.9%)
and K/V by 5.1 microseconds (17.9%). This shows cache-state sensitivity, but the
pre-read is not a guarantee of a perfectly cold cache and provides no DRAM
traffic or bandwidth measurement.

## Nsight Compute procedure and access limitation

The local profiler was inspected before collection:

```text
Nsight Compute 2024.2.1.0 (build 34372528)
```

The following locally installed sections were selected rather than `--set
full`:

```text
SpeedOfLight
LaunchStats
Occupancy
MemoryWorkloadAnalysis
ComputeWorkloadAnalysis
InstructionStats
WarpStateStats
SchedulerStats
SourceCounters
```

The helper [`profile_w4a16.sh`](../benchmarks/profile_w4a16.sh) uses the kernel
filter `regex:.*w4a16_linear_kernel.*`, skips three matching warmups, profiles
one matching launch, saves binary reports under ignored `profiling-tmp/`, and
exports details, raw CSV, session, and SASS text. It explicitly passes
`--clock-control none`; Nsight's local default would otherwise lock base clocks.
It also passes `--cache-control none` and reports Nsight's resulting
uncontrolled-cache warning.

The planned profile cases are:

| Case | M | N | K |
|---|---:|---:|---:|
| Canonical Q/O | 1 | 3072 | 3072 |
| Canonical K/V | 1 | 768 | 3072 |
| Larger M | 8 | 3072 | 3072 |

**Measured profiler status:** Nsight connected to the process and matched
`w4a16_linear_kernel`, then the driver rejected performance-counter access with
`ERR_NVGPUCTRPERM` before creating a report. A second attempt requesting only
`LaunchStats` and `Occupancy` failed identically. Passwordless `sudo -n ncu`
was unavailable because sudo requires a password. No driver, Windows, WSL,
clock, or permission setting was changed, as required by M4B. Once the first
case established that counter access itself was globally blocked, the other
two cases were not repeated into the same failure.

Consequently, the following are unavailable rather than zero:

- achieved occupancy and active warps;
- compute and memory SOL throughput;
- measured DRAM throughput or utilization;
- L2 throughput and hit behavior;
- FP32, integer, and MUFU pipeline utilization;
- IPC and issue efficiency;
- dynamic FFMA/FMUL/FADD/integer/MUFU counts;
- warp stall categories and cycles per instruction.

No `.ncu-rep` was produced or added to the repository.

## Derived launch, occupancy, and SASS evidence

`cuobjdump --dump-resource-usage` on the normally built Release extension gives
21 registers per thread, 32 bytes of static shared memory per block, zero local
memory, zero stack, and no dynamic shared memory for the direct W4A16 kernel.

| Profile target | Derived grid | Block | Grid waves at 6 blocks/SM |
|---|---:|---:|---:|
| Canonical Q/O | 3072 blocks | 256 threads | 8.83 |
| Canonical K/V | 768 blocks | 256 threads | 2.21 |
| Q/O M=8 | 24,576 blocks | 256 threads | 70.62 |

The device limit is 1,536 threads/SM, so six 256-thread blocks provide 48 active
warps and 100% **derived theoretical** thread occupancy. Six blocks consume
32,256 raw registers before allocation rounding, well below 65,536 registers,
and only 192 raw shared-memory bytes. Registers and shared memory therefore do
not limit the thread-derived theoretical occupancy. Achieved occupancy remains
unknown.

The Release binary has no line information, and it was not rebuilt with Debug
or altered flags. SASS is still available. The relevant static offsets are:

| Stage | SASS evidence |
|---|---|
| Global gamma load | `0x0280` |
| Packed E2M1 load and decode | load `0x03a0`; branch/select decode `0x03b0-0x0610` |
| UE4M3 scale load and decode | load `0x0690`; decode `0x06d0-0x0a10` |
| `ldexpf`-related sites | `MUFU.EX2` at `0x0850`, `0x0870`, `0x0910`, `0x0930`, `0x0980` on mutually exclusive helper paths |
| BF16 activation load | `0x0aa0` |
| Block scale, global scale, product, private sum | `0x0ac0-0x0b30` |
| Warp/block reductions and barriers | `0x0b60-0x0d90` |
| BF16 output store | `0x0dc0-0x0e00` |

Across the full static kernel code section, including helper code and mutually
exclusive paths, there are 7 MUFU sites (5 `EX2`, 2 `RCP`), 11 `FMUL`, 15
`FADD`, no `FFMA`, 4 `LDG`, 1 `STG`, 11 `SHFL`, 92 `IMAD`, 28 `IADD3`, and 32
branch sites. These are static opcode occurrences, not executed instruction
counts. The lack of FFMA is consistent with the frozen explicit multiply/add
operation order.

## Hypothesis evaluation

| Hypothesis | M4B result |
|---|---|
| H1: activation rows are redundantly reread | **Supported at source/SASS request level.** One block per output scalar independently executes the activation `LDG`; canonical Q/O requests the same 6 KiB row across 3,072 blocks. Actual DRAM rereads are unknown because L2 counters are unavailable. |
| H2: weight decode has substantial instruction/SFU overhead | **Partly supported.** SASS contains branch-heavy E2M1 decode, generic UE4M3 handling, integer address work, and MUFU paths. The total decode share is not isolated. |
| H3: UE4M3 `ldexpf`/MUFU contributes meaningful latency | **Supported, secondary.** Normal versus exponent-zero scale data differs by 8.2 microseconds, or 10.5% of the normal-path diagnostic. Dynamic MUFU utilization is unavailable. |
| H4: the kernel is memory-traffic limited | **Not established.** The cache pre-read penalty is 11.9% for Q/O and 17.9% for K/V, showing memory-hierarchy sensitivity, but no DRAM or SOL counter was collected. |
| H5: instruction latency/dependency chains dominate | **Favored inference, not counter-proven.** Long-K scaling, the explicit serial private `FADD` chain, decode/control SASS, and the bounded cache penalty point this way. Warp-stall and issue counters are unavailable. |
| H6: 256 threads are poorly utilized in some regimes | **Supported for small cases.** K=128 gives data to only half of the threads while all 256 join the reductions. N=256 supplies fewer blocks than the derived 348-block full-residency capacity and shows a large timing tail. |
| H7: occupancy is register- or shared-memory-limited | **Not supported theoretically.** Static resources permit 100% thread-derived occupancy. Achieved occupancy and runtime limiters are unavailable. |
| H8: M=1 differs materially from larger M | **Supported.** Median per-row cost falls 16.8% for Q/O and 36.2% for K/V by M=16, then the larger-M series is close to linear. |

## Bottleneck attribution

The strongest current **inference** for the steady K=3072 path is that
per-element decode, integer/address/control work, and the private accumulation
dependency chain are the primary cost. Evidence is the near-linear K scaling,
the branch/MUFU-heavy static decode path, the explicit non-FFMA multiply/add
sequence, and the measurable but bounded 10.5% normal-scale A/B delta.

The secondary inferred cost is memory-hierarchy work from independently loading
activation, packed payload, and scale data in every output block. The cache
pre-read experiment changes latency by 11.9-17.9%, and the source-level load
duplication is large, but actual DRAM versus L2 behavior is unresolved. For
N=256 and K=128, short-grid occupancy and fixed reduction work are additional
regime-specific bottlenecks.

Because counter access was denied, M4B cannot conclusively classify the kernel
as hardware memory-bound or instruction-latency-bound. The attribution above
is deliberately weaker than an SOL, DRAM, or warp-stall conclusion.

## Ranked M4C hypotheses

These are hypotheses only; none was implemented in M4B. Every candidate must
preserve the frozen numerical operation order and pass M4A correctness.

1. Decode each UE4M3 block scale once per 16 weights and consume both E2M1
   nibbles from each packed byte before reloading. This targets repeated scale
   and packed-byte work plus the observed decode cost.
2. Reuse an activation tile across multiple output features in a block or
   cooperative output group. This targets the confirmed source-level activation
   rereads and the measured cache sensitivity.
3. Replace the generic normal UE4M3 `ldexpf` path with a validated predecoded or
   lookup representation. The artificial A/B bounds an observable opportunity
   at about 8.2 microseconds for canonical Q/O.
4. Revisit the one-block-per-output, 256-thread mapping for short K and small
   grids, including warp-per-output or multiple outputs per block. K=128 wastes
   half the data threads, and N=256 cannot fill the derived resident-block
   capacity.

## Limitations

- One consumer laptop GPU and one captured run are represented.
- Clocks and power were intentionally uncontrolled; sparse latency outliers are
  part of the result.
- Warmups make the primary result a steady-state baseline. The cache diagnostic
  is controlled but does not prove a completely cold cache.
- Deterministic random weights are representative inputs, not a trained model
  checkpoint. M scaling repeats one activation row to isolate grid scaling.
- Logical bytes and source-requested bytes are not measured memory traffic.
- Nsight performance counters, achieved occupancy, SOL, cache rates, dynamic
  instructions, and warp stalls are unavailable due driver permissions.
- Source-line correlation is unavailable in the existing Release binary; no
  compilation-mode change was made.
- No BF16/FP32 GEMM was measured, and the standalone dequantizer is context only.

