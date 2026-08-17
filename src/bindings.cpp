#include "nvfp4.h"
#include "rmsnorm.h"
#include "rope.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace cuda_nvfp4_decoder_attention {

namespace {

void validate_packed_values(const at::Tensor& packed_values) {
    TORCH_CHECK(
        packed_values.is_cuda(),
        "packed_values must be a CUDA tensor");
    TORCH_CHECK(
        packed_values.scalar_type() == at::kByte,
        "packed_values must have dtype torch.uint8");
    TORCH_CHECK(
        packed_values.is_contiguous(),
        "packed_values must be contiguous");
    TORCH_CHECK(
        packed_values.dim() == 2,
        "packed_values must have rank 2, got rank ",
        packed_values.dim());
    TORCH_CHECK(
        packed_values.size(0) > 0 && packed_values.size(1) > 0,
        "packed_values must have nonempty N and packed-byte dimensions");
    TORCH_CHECK(
        packed_values.size(1) <=
            std::numeric_limits<std::int64_t>::max() / 2,
        "packed_values byte dimension is too large to derive logical K");
    TORCH_CHECK(
        packed_values.numel() <=
            std::numeric_limits<std::int64_t>::max() / 2,
        "packed_values has too many bytes to allocate the unpacked output");

    const std::int64_t grid_blocks =
        (packed_values.numel() - 1) / kNvfp4BlockThreads + 1;
    TORCH_CHECK(
        grid_blocks <= std::numeric_limits<std::int32_t>::max(),
        "packed_values is too large for a one-dimensional CUDA grid");
}

}  // namespace

at::Tensor cuda_unpack_e2m1_codes(const at::Tensor& packed_values) {
    validate_packed_values(packed_values);

    const c10::cuda::CUDAGuard device_guard(packed_values.device());
    at::Tensor output = at::empty(
        {packed_values.size(0), packed_values.size(1) * 2},
        packed_values.options());
    launch_unpack_e2m1_codes_cuda(packed_values, output);
    return output;
}

at::Tensor cuda_dequantize_nvfp4(
    const at::Tensor& packed_values,
    const at::Tensor& block_scales,
    const at::Tensor& global_decode_scale) {
    validate_packed_values(packed_values);
    TORCH_CHECK(
        block_scales.is_cuda(),
        "block_scales must be a CUDA tensor");
    TORCH_CHECK(
        block_scales.scalar_type() == at::kByte,
        "block_scales must have dtype torch.uint8");
    TORCH_CHECK(
        block_scales.is_contiguous(),
        "block_scales must be contiguous");
    TORCH_CHECK(
        block_scales.dim() == 2,
        "block_scales must have rank 2, got rank ",
        block_scales.dim());
    TORCH_CHECK(
        block_scales.device() == packed_values.device(),
        "block_scales must be on the same CUDA device as packed_values");

    TORCH_CHECK(
        global_decode_scale.is_cuda(),
        "global_decode_scale must be a CUDA tensor");
    TORCH_CHECK(
        global_decode_scale.scalar_type() == at::kFloat,
        "global_decode_scale must have dtype torch.float32");
    TORCH_CHECK(
        global_decode_scale.is_contiguous(),
        "global_decode_scale must be contiguous");
    TORCH_CHECK(
        global_decode_scale.dim() == 0,
        "global_decode_scale must be a scalar tensor with shape []");
    TORCH_CHECK(
        global_decode_scale.device() == packed_values.device(),
        "global_decode_scale must be on the same CUDA device as packed_values");

    const std::int64_t rows = packed_values.size(0);
    const std::int64_t columns = packed_values.size(1) * 2;
    TORCH_CHECK(columns >= 16, "logical K must be at least 16, got ", columns);
    TORCH_CHECK(
        columns % 16 == 0,
        "logical K must be divisible by 16, got ",
        columns);
    TORCH_CHECK(
        block_scales.size(0) == rows,
        "block_scales row count must match packed_values (",
        block_scales.size(0),
        " != ",
        rows,
        ")");
    TORCH_CHECK(
        block_scales.size(1) == columns / 16,
        "block_scales must have shape [N, K/16]; expected [",
        rows,
        ", ",
        columns / 16,
        "], got [",
        block_scales.size(0),
        ", ",
        block_scales.size(1),
        "]");

    const c10::cuda::CUDAGuard device_guard(packed_values.device());
    at::Tensor output = at::empty(
        {rows, columns},
        packed_values.options().dtype(at::kFloat));
    launch_dequantize_nvfp4_cuda(
        packed_values,
        block_scales,
        global_decode_scale,
        output);
    return output;
}

at::Tensor cuda_rms_norm(
    const at::Tensor& x,
    const at::Tensor& weight,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16,
        "x must have dtype torch.bfloat16");
    TORCH_CHECK(
        weight.scalar_type() == at::kBFloat16,
        "weight must have dtype torch.bfloat16");
    TORCH_CHECK(
        x.device() == weight.device(),
        "x and weight must be on the same CUDA device");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(
        x.dim() >= 1 && x.dim() <= 4,
        "x must have rank in [1, 4], got rank ",
        x.dim());
    TORCH_CHECK(weight.dim() == 1, "weight must be rank 1");
    TORCH_CHECK(x.numel() > 0, "x must be nonempty");
    TORCH_CHECK(weight.numel() > 0, "weight must be nonempty");
    TORCH_CHECK(
        weight.numel() == x.size(-1),
        "weight length must match x final dimension (",
        weight.numel(),
        " != ",
        x.size(-1),
        ")");
    TORCH_CHECK(
        std::isfinite(eps) && eps > 0.0,
        "eps must be finite and positive");
    const float eps_fp32 = static_cast<float>(eps);
    TORCH_CHECK(
        std::isfinite(eps_fp32) && eps_fp32 > 0.0F,
        "eps must be representable as a finite positive FP32 value");

    const std::int64_t rows = x.numel() / x.size(-1);
    TORCH_CHECK(
        rows <= std::numeric_limits<std::int32_t>::max(),
        "x has too many logical rows for a one-dimensional CUDA grid");

    const c10::cuda::CUDAGuard device_guard(x.device());
    at::Tensor output = at::empty_like(x);
    launch_rms_norm_cuda(x, weight, output, eps_fp32);
    return output;
}

at::Tensor cuda_apply_rope(
    const at::Tensor& x,
    std::int64_t past_length,
    double rope_theta) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16,
        "x must have dtype torch.bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must have rank 4, got rank ", x.dim());
    TORCH_CHECK(
        x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 && x.size(3) > 0,
        "x dimensions must all be nonempty");

    const std::int64_t token_count = x.size(1);
    const std::int64_t head_dim = x.size(3);
    TORCH_CHECK(head_dim >= 2, "x head dimension must be at least 2");
    TORCH_CHECK(
        head_dim % 2 == 0,
        "x head dimension must be even for adjacent-pair RoPE");
    TORCH_CHECK(past_length >= 0, "past_length must be nonnegative");
    TORCH_CHECK(
        past_length <=
            std::numeric_limits<std::int64_t>::max() - (token_count - 1),
        "past_length plus token index must fit in int64");
    TORCH_CHECK(
        std::isfinite(rope_theta) && rope_theta > 0.0,
        "rope_theta must be finite and positive");
    const float rope_theta_fp32 = static_cast<float>(rope_theta);
    TORCH_CHECK(
        std::isfinite(rope_theta_fp32) && rope_theta_fp32 > 0.0F,
        "rope_theta must be representable as a finite positive FP32 value");

    const std::int64_t total_pairs = x.numel() / 2;
    const std::int64_t grid_blocks =
        (total_pairs - 1) / kRopeBlockThreads + 1;
    TORCH_CHECK(
        grid_blocks <= std::numeric_limits<std::int32_t>::max(),
        "x has too many logical adjacent pairs for a one-dimensional CUDA grid");

    const c10::cuda::CUDAGuard device_guard(x.device());
    at::Tensor output = at::empty_like(x);
    launch_rope_cuda(x, output, past_length, rope_theta_fp32);
    return output;
}

}  // namespace cuda_nvfp4_decoder_attention

TORCH_LIBRARY(cuda_nvfp4_decoder_attention, module) {
    module.def(
        "cuda_unpack_e2m1_codes(Tensor packed_values) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_unpack_e2m1_codes));
    module.def(
        "cuda_dequantize_nvfp4(Tensor packed_values, Tensor block_scales, Tensor global_decode_scale) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_dequantize_nvfp4));
    module.def(
        "cuda_rms_norm(Tensor x, Tensor weight, float eps) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_rms_norm));
    module.def(
        "cuda_apply_rope(Tensor x, int past_length, float rope_theta=10000.0) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_apply_rope));
}
