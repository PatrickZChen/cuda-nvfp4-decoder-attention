#include "rope.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace cuda_nvfp4_decoder_attention {
namespace {

static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));

__global__ void apply_rope_kernel(
    const __nv_bfloat16* x,
    __nv_bfloat16* output,
    std::int64_t total_pairs,
    std::int64_t token_count,
    std::int64_t head_count,
    std::int64_t head_dim,
    std::int64_t past_length,
    float rope_theta) {
    const std::int64_t pair_index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_index >= total_pairs) {
        return;
    }

    const std::int64_t pairs_per_head = head_dim / 2;
    // Contiguous [B,T,H,D] rows are ordered as ((b*T + t)*H + h).
    // Dividing by D/2 recovers that row; m is then confined to the row, and
    // (row / H) % T recovers t without mixing batch or head coordinates.
    const std::int64_t row = pair_index / pairs_per_head;
    const std::int64_t pair_in_head = pair_index % pairs_per_head;
    const std::int64_t token_index = (row / head_count) % token_count;
    const std::int64_t absolute_position = past_length + token_index;
    const std::int64_t element_offset =
        row * head_dim + 2 * pair_in_head;

    // Position zero is the exact storage identity, including BF16 signed zero.
    if (absolute_position == 0) {
        output[element_offset] = x[element_offset];
        output[element_offset + 1] = x[element_offset + 1];
        return;
    }

    const float exponent = __fdiv_rn(
        static_cast<float>(2 * pair_in_head),
        static_cast<float>(head_dim));
    const float denominator = powf(rope_theta, exponent);
    const float angle = __fdiv_rn(
        static_cast<float>(absolute_position),
        denominator);
    const float cosine = cosf(angle);
    const float sine = sinf(angle);

    const float x_even = __bfloat162float(x[element_offset]);
    const float x_odd = __bfloat162float(x[element_offset + 1]);
    const float y_even = __fsub_rn(
        __fmul_rn(x_even, cosine),
        __fmul_rn(x_odd, sine));
    const float y_odd = __fadd_rn(
        __fmul_rn(x_even, sine),
        __fmul_rn(x_odd, cosine));

    output[element_offset] = __float2bfloat16_rn(y_even);
    output[element_offset + 1] = __float2bfloat16_rn(y_odd);
}

}  // namespace

void launch_rope_cuda(
    const at::Tensor& x,
    at::Tensor& output,
    std::int64_t past_length,
    float rope_theta) {
    const std::int64_t total_pairs = x.numel() / 2;
    const std::int64_t grid_blocks =
        (total_pairs - 1) / kRopeBlockThreads + 1;
    TORCH_INTERNAL_ASSERT(total_pairs > 0);
    TORCH_INTERNAL_ASSERT(
        grid_blocks <= std::numeric_limits<std::int32_t>::max());

    const auto* x_data = reinterpret_cast<const __nv_bfloat16*>(
        x.data_ptr<at::BFloat16>());
    auto* output_data = reinterpret_cast<__nv_bfloat16*>(
        output.data_ptr<at::BFloat16>());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    apply_rope_kernel<<<
        static_cast<unsigned int>(grid_blocks),
        kRopeBlockThreads,
        0,
        stream>>>(
        x_data,
        output_data,
        total_pairs,
        x.size(1),
        x.size(2),
        x.size(3),
        past_length,
        rope_theta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cuda_nvfp4_decoder_attention
