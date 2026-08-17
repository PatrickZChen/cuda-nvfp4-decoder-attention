# CUDA NVFP4 Decoder Attention

This repository is independently designed and implemented as a clean-room CUDA performance-engineering study of a transformer decoder-attention block. Its primary development target is the NVIDIA GeForce RTX 4080 Laptop GPU (Ada, SM89). Ada does not provide native NVFP4 Tensor Core matrix execution, so the planned low-precision path uses packed NVFP4 projection weights with software unpack/dequantization, BF16 activations, FP32 accumulation, and BF16 outputs (`nvfp4_w4a16`). Grouped-query attention (GQA), adjacent-pair RoPE, BF16 KV caching, causal attention, numerical analysis, profiler-driven optimization, and selective fusion are in scope. The repository now contains the BF16/FP32 decoder-attention reference, a portable NVFP4 numerical reference, validated modular CUDA primitives for RMSNorm, adjacent-pair RoPE, and portable NVFP4 unpack/dequantization, and a direct software-decoded W4A16 projection baseline. The direct projection consumes portable packed NVFP4 weights without materializing a complete FP32 weight matrix, decodes weights in-kernel, multiplies BF16 activations in FP32, accumulates FP32, and stores BF16 output. These kernels have run on the RTX 4080 target and passed Compute Sanitizer memcheck with zero errors. This is not native FP4 Tensor Core execution or an optimized GEMM; the published measurements are an unoptimized baseline, not a speedup claim.

## Semantic pipeline

```text
BF16 hidden states
        ↓
input RMSNorm
        ↓
Q / K / V projections
        ↓
reshape into heads
        ↓
per-head Q RMSNorm and per-head K RMSNorm
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

The canonical configuration is `H = 3072`, `Hq = 24`, `Hkv = 6`, and `D = 128`, giving a 4:1 GQA ratio. Incremental decoding with `T = 1` is the primary workload; prefill is secondary.

## Planned execution modes

- `bf16`: BF16 weights and activations, FP32 accumulation, and BF16 outputs.
- `nvfp4_w4a16`: software-decoded NVFP4 projection weights on Ada, BF16 activations, FP32 accumulation, and BF16 outputs.
- `nvfp4_w4a4_reference`: an optional future numerical experiment with logical NVFP4 activations and weights; it is not an optimized Ada path.

NVFP4 names a numerical and packed-weight representation in this project; it does not mean that attention or matrix execution is natively FP4 on Ada. A native Blackwell FP4 backend is optional future work and may require a separate physical layout.

Milestone 1 provides the BF16/FP32 PyTorch correctness reference, Milestone 2B provides the portable NVFP4 numerical reference, Milestones 3A–3C add CUDA RMSNorm, adjacent-pair RoPE, and portable E2M1 unpack/NVFP4 software dequantization, and Milestone 4A adds the direct correctness-first W4A16 projection baseline. The modular CUDA primitives compose into the Q/K RMSNorm-then-RoPE path. See [the architecture specification](docs/ARCHITECTURE.md) and [the numerical contract](docs/NUMERICS.md) for the complete semantic and numerical contracts, validation philosophy, limitations, and roadmap.

Milestone 4B adds a reproducible CUDA-event benchmark and a targeted Nsight Compute workflow for that unchanged direct W4A16 baseline. See [the performance baseline](docs/PERFORMANCE_BASELINE.md) for measurements, profiler availability, restrained bottleneck analysis, and M4C hypotheses; no optimization or speedup claim is made.

## CUDA primitives build and validation

The development build assumes Python 3.12 with PyTorch 2.6.0+cu124 and pytest in `.venv`, CMake, CUDA Toolkit 12.5, and an SM89-capable CUDA device. From the repository root:

```bash
./scripts/build_cuda.sh
source .venv/bin/activate
pytest -q
python -m pytest -q
python -m pytest -q tests/test_cuda_rmsnorm.py
python -m pytest -q tests/test_cuda_rope.py
python -m pytest -q tests/test_cuda_nvfp4.py
python -m pytest -q tests/test_cuda_w4a16.py
compute-sanitizer --tool memcheck --error-exitcode 99 \
    .venv/bin/python scripts/validate_cuda_rmsnorm.py
compute-sanitizer --tool memcheck --error-exitcode 99 \
    .venv/bin/python scripts/validate_cuda_rope.py
compute-sanitizer --tool memcheck --error-exitcode 99 \
    .venv/bin/python scripts/validate_cuda_nvfp4.py
compute-sanitizer --tool memcheck --error-exitcode 99 \
    .venv/bin/python scripts/validate_cuda_w4a16.py
```

The script performs a normal out-of-tree `Release` build in `build-cuda` and targets SM89 only. Configuration reports that installed PyTorch was built with CUDA 12.4 while the extension uses Toolkit 12.5; this minor-version difference is retained and validated rather than changing either installation.

The Python loader is `cuda_primitives.py`. It exposes `cuda_rms_norm(x, weight, eps)`, `cuda_apply_rope(x, past_length, rope_theta=10000.0)`, `cuda_unpack_e2m1_codes(packed_values)`, `cuda_dequantize_nvfp4(quantized)`, and `cuda_w4a16_linear(x, weight)` from the same normally built `cuda_primitives.so`. RMSNorm accepts contiguous BF16 CUDA input of rank 1–4 and treats the final dimension as independent rows. RoPE accepts contiguous BF16 CUDA `[B,T,H,D]` input with positive dimensions and even `D >= 2`, applies the frozen adjacent-pair convention at absolute position `past_length + token_index`, and returns new BF16 storage. The NVFP4 operations use the repository's even-low/odd-high portable bytes and row-local 16-element UE4M3 scales. Direct W4A16 accepts contiguous BF16 CUDA activation storage and validated portable `NVFP4Tensor` weight storage and returns BF16 `x @ weight.T` output; it does not call the standalone dequantizer or allocate FP32 `[N,K]` weight storage.

`NVFP4Tensor` construction owns canonical numerical-storage validation, including finite canonical UE4M3 bytes and the global scale value. The low-level CUDA operator checks device, dtype, rank, contiguity, shape, device agreement, and launch bounds without copying tensors or reading device values back to the host. Calling the raw operator therefore requires already validated canonical storage. The normal decode path performs no scale-byte scan or scalar `.item()` synchronization.

RMSNorm, RoPE, portable NVFP4 unpack/dequantization, and direct W4A16 projection have run on the RTX 4080 SM89 target and passed Compute Sanitizer memcheck with zero errors. M4A makes no native FP4, optimized GEMM, attention, fusion, or performance claim.

## License

MIT; see [LICENSE](LICENSE).
