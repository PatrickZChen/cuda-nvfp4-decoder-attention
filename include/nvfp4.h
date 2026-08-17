#pragma once

#include <ATen/core/Tensor.h>

namespace cuda_nvfp4_decoder_attention {

constexpr int kNvfp4BlockThreads = 256;

at::Tensor cuda_unpack_e2m1_codes(const at::Tensor& packed_values);

at::Tensor cuda_dequantize_nvfp4(
    const at::Tensor& packed_values,
    const at::Tensor& block_scales,
    const at::Tensor& global_decode_scale);

void launch_unpack_e2m1_codes_cuda(
    const at::Tensor& packed_values,
    at::Tensor& output);

void launch_dequantize_nvfp4_cuda(
    const at::Tensor& packed_values,
    const at::Tensor& block_scales,
    const at::Tensor& global_decode_scale,
    at::Tensor& output);

}  // namespace cuda_nvfp4_decoder_attention
