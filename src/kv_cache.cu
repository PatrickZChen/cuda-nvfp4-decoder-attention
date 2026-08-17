#include "kv_cache.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace cuda_nvfp4_decoder_attention {
namespace {

static_assert(kKvCacheAppendBlockThreads == 256);
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));

__global__ void kv_cache_append_kernel(
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    const __nv_bfloat16* new_k,
    const __nv_bfloat16* new_v,
    std::int64_t element_count,
    std::int64_t token_count,
    std::int64_t kv_head_count,
    std::int64_t head_dim,
    std::int64_t cache_capacity,
    std::int64_t past_length) {
    const std::int64_t x =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= element_count) {
        return;
    }

    // Decode contiguous new K/V [B,T,Hkv,D] in d, h, i, b order.
    std::int64_t remaining = x;
    const std::int64_t d = remaining % head_dim;
    remaining /= head_dim;
    const std::int64_t kv_head = remaining % kv_head_count;
    remaining /= kv_head_count;
    const std::int64_t token = remaining % token_count;
    const std::int64_t batch = remaining / token_count;

    const std::int64_t source_offset =
        ((batch * token_count + token) * kv_head_count + kv_head) *
            head_dim +
        d;
    const std::int64_t destination_offset =
        ((batch * kv_head_count + kv_head) * cache_capacity +
         (past_length + token)) *
            head_dim +
        d;

    // Direct BF16 assignments are storage movement only and preserve all bits.
    k_cache[destination_offset] = new_k[source_offset];
    v_cache[destination_offset] = new_v[source_offset];
}

}  // namespace

void launch_kv_cache_append_cuda(
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    const at::Tensor& new_k,
    const at::Tensor& new_v,
    std::int64_t past_length,
    std::int64_t element_count,
    std::int64_t grid_blocks) {
    TORCH_INTERNAL_ASSERT(element_count > 0);
    TORCH_INTERNAL_ASSERT(grid_blocks > 0);
    TORCH_INTERNAL_ASSERT(
        grid_blocks <= std::numeric_limits<std::int32_t>::max());

    auto* k_cache_data = reinterpret_cast<__nv_bfloat16*>(
        k_cache.data_ptr<at::BFloat16>());
    auto* v_cache_data = reinterpret_cast<__nv_bfloat16*>(
        v_cache.data_ptr<at::BFloat16>());
    const auto* new_k_data = reinterpret_cast<const __nv_bfloat16*>(
        new_k.data_ptr<at::BFloat16>());
    const auto* new_v_data = reinterpret_cast<const __nv_bfloat16*>(
        new_v.data_ptr<at::BFloat16>());
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(k_cache.get_device());

    kv_cache_append_kernel<<<
        static_cast<unsigned int>(grid_blocks),
        kKvCacheAppendBlockThreads,
        0,
        stream>>>(
        k_cache_data,
        v_cache_data,
        new_k_data,
        new_v_data,
        element_count,
        new_k.size(1),
        new_k.size(2),
        new_k.size(3),
        k_cache.size(2),
        past_length);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cuda_nvfp4_decoder_attention
