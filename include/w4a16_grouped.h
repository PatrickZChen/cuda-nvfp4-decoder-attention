#pragma once

#include <ATen/core/Tensor.h>

namespace cuda_nvfp4_decoder_attention {

constexpr int kW4a16GroupedBlockThreads = 256;

at::Tensor cuda_w4a16_linear_grouped_decode(
    const at::Tensor& x,
    const at::Tensor& packed_values,
    const at::Tensor& block_scales,
    const at::Tensor& global_decode_scale);

void launch_w4a16_linear_grouped_decode_cuda(
    const at::Tensor& x,
    const at::Tensor& packed_values,
    const at::Tensor& block_scales,
    const at::Tensor& global_decode_scale,
    at::Tensor& output);

}  // namespace cuda_nvfp4_decoder_attention
