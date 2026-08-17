#include "gqa_attention.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>

namespace cuda_nvfp4_decoder_attention {
namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kGqaAttentionBlockThreads / kWarpSize;
constexpr unsigned int kFullWarpMask = 0xffffffffU;

static_assert(kGqaAttentionBlockThreads == 256);
static_assert(kGqaAttentionBlockThreads % kWarpSize == 0);
static_assert(kWarpsPerBlock <= kWarpSize);
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = __fadd_rn(
            value,
            __shfl_down_sync(kFullWarpMask, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(
            value,
            __shfl_down_sync(kFullWarpMask, value, offset));
    }
    return value;
}

__device__ __forceinline__ float block_sum(
    float value,
    float* warp_values) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = warp_sum(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        float block_value =
            lane < kWarpsPerBlock ? warp_values[lane] : 0.0F;
        block_value = warp_sum(block_value);
        if (lane == 0) {
            warp_values[0] = block_value;
        }
    }
    __syncthreads();
    return warp_values[0];
}

__device__ __forceinline__ float block_max(
    float value,
    float* warp_values) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = warp_max(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        float block_value = lane < kWarpsPerBlock
            ? warp_values[lane]
            : -CUDART_INF_F;
        block_value = warp_max(block_value);
        if (lane == 0) {
            warp_values[0] = block_value;
        }
    }
    __syncthreads();
    return warp_values[0];
}

__global__ void qk_scores_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* present_k,
    float* scores,
    std::int64_t score_count,
    std::int64_t token_count,
    std::int64_t query_head_count,
    std::int64_t kv_head_count,
    std::int64_t head_dim,
    std::int64_t context_length,
    std::int64_t past_length,
    float inverse_sqrt_head_dim) {
    const std::int64_t score_index = static_cast<std::int64_t>(blockIdx.x);
    if (score_index >= score_count) {
        return;
    }

    std::int64_t remaining = score_index;
    const std::int64_t key_index = remaining % context_length;
    remaining /= context_length;
    const std::int64_t token_index = remaining % token_count;
    remaining /= token_count;
    const std::int64_t query_head = remaining % query_head_count;
    const std::int64_t batch = remaining / query_head_count;

    // This branch is uniform across the block. Masked logits are represented
    // by negative infinity before softmax and perform no unnecessary QK work.
    if (key_index > past_length + token_index) {
        if (threadIdx.x == 0) {
            scores[score_index] = -CUDART_INF_F;
        }
        return;
    }

    const std::int64_t group_size = query_head_count / kv_head_count;
    const std::int64_t kv_head = query_head / group_size;
    const std::int64_t q_row_offset =
        ((batch * token_count + token_index) * query_head_count +
         query_head) * head_dim;
    const std::int64_t k_row_offset =
        ((batch * kv_head_count + kv_head) * context_length + key_index) *
        head_dim;

    float partial_sum = 0.0F;
    for (std::int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float q_value = __bfloat162float(q[q_row_offset + d]);
        const float k_value = __bfloat162float(present_k[k_row_offset + d]);
        const float product = __fmul_rn(q_value, k_value);
        partial_sum = __fadd_rn(partial_sum, product);
    }

    // Every unmasked block has 256 active threads. Threads without a D value
    // contribute zero and remain active through both reduction levels.
    __shared__ float warp_values[kWarpsPerBlock];
    const float dot_product = block_sum(partial_sum, warp_values);
    if (threadIdx.x == 0) {
        scores[score_index] = __fmul_rn(
            dot_product,
            inverse_sqrt_head_dim);
    }
}

__global__ void softmax_kernel(
    const float* scores,
    float* probabilities,
    std::int64_t row_count,
    std::int64_t context_length) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= row_count) {
        return;
    }

    const std::int64_t row_offset = row * context_length;
    __shared__ float warp_values[kWarpsPerBlock];

    float thread_maximum = -CUDART_INF_F;
    for (std::int64_t j = threadIdx.x; j < context_length;
         j += blockDim.x) {
        thread_maximum = fmaxf(thread_maximum, scores[row_offset + j]);
    }
    const float row_maximum = block_max(thread_maximum, warp_values);

    float thread_denominator = 0.0F;
    for (std::int64_t j = threadIdx.x; j < context_length;
         j += blockDim.x) {
        // Ordinary expf is deliberate. A masked -inf score subtracts the
        // finite row maximum to remain -inf, then exponentiates exactly to 0.
        const float exponential = expf(__fsub_rn(
            scores[row_offset + j],
            row_maximum));
        probabilities[row_offset + j] = exponential;
        thread_denominator = __fadd_rn(thread_denominator, exponential);
    }
    const float denominator = block_sum(thread_denominator, warp_values);

    // S=P+T and j=P+i is visible for every valid row, so row_maximum is
    // finite and denominator is positive for finite input tensors. There is
    // intentionally no clamp or epsilon in this frozen softmax equation.
    for (std::int64_t j = threadIdx.x; j < context_length;
         j += blockDim.x) {
        probabilities[row_offset + j] = __fdiv_rn(
            probabilities[row_offset + j],
            denominator);
    }
}

__global__ void pv_context_kernel(
    const float* probabilities,
    const __nv_bfloat16* present_v,
    __nv_bfloat16* context,
    std::int64_t context_count,
    std::int64_t token_count,
    std::int64_t query_head_count,
    std::int64_t kv_head_count,
    std::int64_t head_dim,
    std::int64_t context_length) {
    const std::int64_t context_index =
        static_cast<std::int64_t>(blockIdx.x);
    if (context_index >= context_count) {
        return;
    }

    std::int64_t remaining = context_index;
    const std::int64_t d = remaining % head_dim;
    remaining /= head_dim;
    const std::int64_t query_head = remaining % query_head_count;
    remaining /= query_head_count;
    const std::int64_t token_index = remaining % token_count;
    const std::int64_t batch = remaining / token_count;

    const std::int64_t group_size = query_head_count / kv_head_count;
    const std::int64_t kv_head = query_head / group_size;
    const std::int64_t probability_row_offset =
        ((batch * query_head_count + query_head) * token_count +
         token_index) * context_length;
    const std::int64_t v_head_offset =
        (batch * kv_head_count + kv_head) * context_length * head_dim;

    float partial_sum = 0.0F;
    for (std::int64_t j = threadIdx.x; j < context_length;
         j += blockDim.x) {
        const float probability = probabilities[probability_row_offset + j];
        const float v_value = __bfloat162float(
            present_v[v_head_offset + j * head_dim + d]);
        const float product = __fmul_rn(probability, v_value);
        partial_sum = __fadd_rn(partial_sum, product);
    }

    __shared__ float warp_values[kWarpsPerBlock];
    const float context_value = block_sum(partial_sum, warp_values);
    if (threadIdx.x == 0) {
        context[context_index] = __float2bfloat16_rn(context_value);
    }
}

}  // namespace

void launch_gqa_attention_cuda(
    const at::Tensor& q,
    const at::Tensor& present_k,
    const at::Tensor& present_v,
    at::Tensor& scores,
    at::Tensor& probabilities,
    at::Tensor& context,
    std::int64_t past_length,
    float inverse_sqrt_head_dim) {
    const std::int64_t score_count = scores.numel();
    const std::int64_t row_count =
        q.size(0) * q.size(2) * q.size(1);
    const std::int64_t context_count = context.numel();
    TORCH_INTERNAL_ASSERT(score_count > 0);
    TORCH_INTERNAL_ASSERT(row_count > 0);
    TORCH_INTERNAL_ASSERT(context_count > 0);
    TORCH_INTERNAL_ASSERT(
        score_count <= std::numeric_limits<std::int32_t>::max());
    TORCH_INTERNAL_ASSERT(
        row_count <= std::numeric_limits<std::int32_t>::max());
    TORCH_INTERNAL_ASSERT(
        context_count <= std::numeric_limits<std::int32_t>::max());

    const auto* q_data = reinterpret_cast<const __nv_bfloat16*>(
        q.data_ptr<at::BFloat16>());
    const auto* k_data = reinterpret_cast<const __nv_bfloat16*>(
        present_k.data_ptr<at::BFloat16>());
    const auto* v_data = reinterpret_cast<const __nv_bfloat16*>(
        present_v.data_ptr<at::BFloat16>());
    auto* score_data = scores.data_ptr<float>();
    auto* probability_data = probabilities.data_ptr<float>();
    auto* context_data = reinterpret_cast<__nv_bfloat16*>(
        context.data_ptr<at::BFloat16>());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());

    qk_scores_kernel<<<
        static_cast<unsigned int>(score_count),
        kGqaAttentionBlockThreads,
        0,
        stream>>>(
        q_data,
        k_data,
        score_data,
        score_count,
        q.size(1),
        q.size(2),
        present_k.size(1),
        q.size(3),
        present_k.size(2),
        past_length,
        inverse_sqrt_head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    softmax_kernel<<<
        static_cast<unsigned int>(row_count),
        kGqaAttentionBlockThreads,
        0,
        stream>>>(
        score_data,
        probability_data,
        row_count,
        present_k.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    pv_context_kernel<<<
        static_cast<unsigned int>(context_count),
        kGqaAttentionBlockThreads,
        0,
        stream>>>(
        probability_data,
        v_data,
        context_data,
        context_count,
        q.size(1),
        q.size(2),
        present_k.size(1),
        q.size(3),
        present_k.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cuda_nvfp4_decoder_attention
