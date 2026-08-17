#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace cuda_nvfp4_decoder_attention {

constexpr int kRopeBlockThreads = 256;

at::Tensor cuda_apply_rope(
    const at::Tensor& x,
    std::int64_t past_length,
    double rope_theta);

void launch_rope_cuda(
    const at::Tensor& x,
    at::Tensor& output,
    std::int64_t past_length,
    float rope_theta);

}  // namespace cuda_nvfp4_decoder_attention
