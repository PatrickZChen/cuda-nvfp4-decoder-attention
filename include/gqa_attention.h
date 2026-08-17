#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace cuda_nvfp4_decoder_attention {

constexpr int kGqaAttentionBlockThreads = 256;

at::Tensor cuda_gqa_attention(
    const at::Tensor& q,
    const at::Tensor& present_k,
    const at::Tensor& present_v,
    std::int64_t past_length);

void launch_gqa_attention_cuda(
    const at::Tensor& q,
    const at::Tensor& present_k,
    const at::Tensor& present_v,
    at::Tensor& scores,
    at::Tensor& probabilities,
    at::Tensor& context,
    std::int64_t past_length,
    float inverse_sqrt_head_dim);

}  // namespace cuda_nvfp4_decoder_attention
