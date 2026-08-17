# CUDA NVFP4 Decoder Attention

This repository is independently designed and implemented as a clean-room CUDA performance-engineering study of a transformer decoder-attention block. Its primary development target is the NVIDIA GeForce RTX 4080 Laptop GPU (Ada, SM89). Ada does not provide native NVFP4 Tensor Core matrix execution, so the planned low-precision path uses packed NVFP4 projection weights with software unpack/dequantization, BF16 activations, FP32 accumulation, and BF16 outputs (`nvfp4_w4a16`). Grouped-query attention (GQA), adjacent-pair RoPE, BF16 KV caching, causal attention, numerical analysis, profiler-driven optimization, and selective fusion are in scope. The project is currently in its reference-implementation phase: Milestone 0 freezes the semantics, and Milestone 1 provides a PyTorch correctness oracle. No performance claims have been published.

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

Milestone 1 adds the PyTorch correctness reference and semantic tests. There is no CUDA, C++, benchmark, build-system, or performance implementation yet. See [the architecture specification](docs/ARCHITECTURE.md) for the complete semantic contract, precision boundaries, validation philosophy, benchmark methodology, limitations, and roadmap.

## License

MIT; see [LICENSE](LICENSE).
