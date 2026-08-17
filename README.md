# CUDA NVFP4 Decoder Attention

This repository is independently designed and implemented as a clean-room CUDA performance-engineering study of a transformer decoder-attention block. Its primary development target is the NVIDIA GeForce RTX 4080 Laptop GPU (Ada, SM89). Ada does not provide native NVFP4 Tensor Core matrix execution, so the planned low-precision path uses packed NVFP4 projection weights with software unpack/dequantization, BF16 activations, FP32 accumulation, and BF16 outputs (`nvfp4_w4a16`). Grouped-query attention (GQA), adjacent-pair RoPE, BF16 KV caching, causal attention, numerical analysis, profiler-driven optimization, and selective fusion are in scope. The repository now contains the BF16/FP32 decoder-attention reference, a portable NVFP4 numerical reference, and its first validated CUDA primitive: RMSNorm. The custom kernel has run on the RTX 4080 target and passed Compute Sanitizer memcheck with zero errors. No performance claims have been published.

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

Milestone 1 provides the BF16/FP32 PyTorch correctness reference, Milestone 2B provides the portable NVFP4 numerical reference, and Milestone 3A adds a modular CMake/PyTorch CUDA boundary plus baseline RMSNorm. See [the architecture specification](docs/ARCHITECTURE.md) and [the numerical contract](docs/NUMERICS.md) for the complete semantic and numerical contracts, validation philosophy, limitations, and roadmap.

## CUDA RMSNorm build and validation

The development build assumes Python 3.12 with PyTorch 2.6.0+cu124 and pytest in `.venv`, CMake, CUDA Toolkit 12.5, and an SM89-capable CUDA device. From the repository root:

```bash
./scripts/build_cuda.sh
source .venv/bin/activate
pytest -q
python -m pytest -q
python -m pytest -q tests/test_cuda_rmsnorm.py
compute-sanitizer --tool memcheck --error-exitcode 99 \
    .venv/bin/python scripts/validate_cuda_rmsnorm.py
```

The script performs a normal out-of-tree `Release` build in `build-cuda` and targets SM89 only. Configuration reports that installed PyTorch was built with CUDA 12.4 while the extension uses Toolkit 12.5; this minor-version difference is retained and validated rather than changing either installation.

The Python API is `cuda_rmsnorm.cuda_rms_norm(x, weight, eps)`. It accepts contiguous BF16 CUDA input of rank 1–4, treats the final dimension as independent RMSNorm rows, and requires a contiguous BF16 one-dimensional weight on the same device.

## License

MIT; see [LICENSE](LICENSE).
