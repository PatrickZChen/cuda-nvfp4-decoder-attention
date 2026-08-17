#include "rmsnorm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace cuda_nvfp4_decoder_attention {
namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockThreads = 256;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
constexpr unsigned int kFullWarpMask = 0xffffffffU;

static_assert(kBlockThreads % kWarpSize == 0);
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

__global__ void rms_norm_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    std::int64_t columns,
    float eps) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    const std::int64_t row_offset = row * columns;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    float sum_of_squares = 0.0F;
    for (std::int64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float value = __bfloat162float(x[row_offset + column]);
        const float square = __fmul_rn(value, value);
        sum_of_squares = __fadd_rn(sum_of_squares, square);
    }

    // The launch always uses a whole number of full warps. Threads without a
    // column contribute zero but still participate, so every shuffle uses the
    // full active mask even for dimensions such as 4 or 8.
    sum_of_squares = warp_sum(sum_of_squares);

    __shared__ float warp_sums[kWarpsPerBlock];
    if (lane == 0) {
        warp_sums[warp] = sum_of_squares;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0F;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            warp_sums[0] = block_sum;
        }
    }
    __syncthreads();

    const float mean_square = __fdiv_rn(
        warp_sums[0],
        static_cast<float>(columns));
    const float inverse_rms = rsqrtf(__fadd_rn(mean_square, eps));

    for (std::int64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float value = __bfloat162float(x[row_offset + column]);
        const float scale = __bfloat162float(weight[column]);
        const float normalized = __fmul_rn(
            __fmul_rn(value, inverse_rms),
            scale);
        output[row_offset + column] = __float2bfloat16_rn(normalized);
    }
}

}  // namespace

void launch_rms_norm_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    at::Tensor& output,
    float eps) {
    const std::int64_t columns = x.size(-1);
    const std::int64_t rows = x.numel() / columns;
    TORCH_INTERNAL_ASSERT(rows > 0);
    TORCH_INTERNAL_ASSERT(
        rows <= std::numeric_limits<std::int32_t>::max());

    const auto* x_data = reinterpret_cast<const __nv_bfloat16*>(
        x.data_ptr<at::BFloat16>());
    const auto* weight_data = reinterpret_cast<const __nv_bfloat16*>(
        weight.data_ptr<at::BFloat16>());
    auto* output_data = reinterpret_cast<__nv_bfloat16*>(
        output.data_ptr<at::BFloat16>());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    rms_norm_kernel<<<
        static_cast<unsigned int>(rows),
        kBlockThreads,
        0,
        stream>>>(x_data, weight_data, output_data, columns, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cuda_nvfp4_decoder_attention
