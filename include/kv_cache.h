#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace cuda_nvfp4_decoder_attention {

constexpr int kKvCacheAppendBlockThreads = 256;

void cuda_kv_cache_append_(
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    const at::Tensor& new_k,
    const at::Tensor& new_v,
    std::int64_t past_length);

void launch_kv_cache_append_cuda(
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    const at::Tensor& new_k,
    const at::Tensor& new_v,
    std::int64_t past_length,
    std::int64_t element_count,
    std::int64_t grid_blocks);

}  // namespace cuda_nvfp4_decoder_attention
