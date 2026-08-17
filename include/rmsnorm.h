#pragma once

#include <ATen/core/Tensor.h>

namespace cuda_nvfp4_decoder_attention {

at::Tensor cuda_rms_norm(
    const at::Tensor& x,
    const at::Tensor& weight,
    double eps);

void launch_rms_norm_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    at::Tensor& output,
    float eps);

}  // namespace cuda_nvfp4_decoder_attention
