#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace cuda_nvfp4_decoder_attention {

constexpr int kGqaAttentionCachedBlockThreads = 256;

at::Tensor cuda_gqa_attention_cached(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::int64_t past_length);

void launch_gqa_attention_cached_cuda(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    at::Tensor& scores,
    at::Tensor& probabilities,
    at::Tensor& context,
    std::int64_t past_length,
    float inverse_sqrt_head_dim);

}  // namespace cuda_nvfp4_decoder_attention
