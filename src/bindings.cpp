#include "rmsnorm.h"

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

}  // namespace cuda_nvfp4_decoder_attention

TORCH_LIBRARY(cuda_nvfp4_decoder_attention, module) {
    module.def(
        "cuda_rms_norm(Tensor x, Tensor weight, float eps) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_rms_norm));
}
