#include "rmsnorm.h"
#include "rope.h"

#include <ATen/ops/empty_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace cuda_nvfp4_decoder_attention {

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
        "cuda_rms_norm(Tensor x, Tensor weight, float eps) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_rms_norm));
    module.def(
        "cuda_apply_rope(Tensor x, int past_length, float rope_theta=10000.0) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_apply_rope));
}
