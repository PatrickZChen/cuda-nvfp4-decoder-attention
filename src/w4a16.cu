#include "w4a16.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace cuda_nvfp4_decoder_attention {
namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kW4a16BlockThreads / kWarpSize;
constexpr unsigned int kFullWarpMask = 0xffffffffU;

static_assert(kW4a16BlockThreads % kWarpSize == 0);
static_assert(kWarpsPerBlock <= kWarpSize);
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));

__device__ __forceinline__ float decode_e2m1(std::uint8_t code) {
    float magnitude;
    switch (code & 0x07U) {
        case 0x0U:
            magnitude = 0.0F;
            break;
        case 0x1U:
            magnitude = 0.5F;
            break;
        case 0x2U:
            magnitude = 1.0F;
            break;
        case 0x3U:
            magnitude = 1.5F;
            break;
        case 0x4U:
            magnitude = 2.0F;
            break;
        case 0x5U:
            magnitude = 3.0F;
            break;
        case 0x6U:
            magnitude = 4.0F;
            break;
        default:
            magnitude = 6.0F;
            break;
    }

    // Apply the sign bit directly so E2M1 code 0x8 remains negative zero.
    std::uint32_t bits = __float_as_uint(magnitude);
    bits |= static_cast<std::uint32_t>(code & 0x08U) << 28;
    return __uint_as_float(bits);
}

__device__ __forceinline__ float decode_ue4m3(std::uint8_t code) {
    const int exponent = (code >> 3) & 0x0F;
    const int mantissa = code & 0x07;
    if (exponent == 0) {
        return static_cast<float>(mantissa) * 0x1.0p-9F;
    }

    // (1 + m/8) * 2^(e-7) == (8 + m) * 2^(e-10).
    return ldexpf(static_cast<float>(8 + mantissa), exponent - 10);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = __fadd_rn(
            value,
            __shfl_down_sync(kFullWarpMask, value, offset));
    }
    return value;
}

__global__ void w4a16_linear_kernel(
    const __nv_bfloat16* x,
    const std::uint8_t* packed_values,
    const std::uint8_t* block_scales,
    const float* global_decode_scale,
    __nv_bfloat16* output,
    std::int64_t output_count,
    std::int64_t output_features,
    std::int64_t reduction_size,
    std::int64_t packed_bytes_per_row,
    std::int64_t blocks_per_row) {
    const std::int64_t output_index =
        static_cast<std::int64_t>(blockIdx.x);
    if (output_index >= output_count) {
        return;
    }

    const std::int64_t activation_row = output_index / output_features;
    const std::int64_t output_feature =
        output_index - activation_row * output_features;
    const std::int64_t activation_row_offset =
        activation_row * reduction_size;
    const std::int64_t packed_row_offset =
        output_feature * packed_bytes_per_row;
    const std::int64_t scale_row_offset =
        output_feature * blocks_per_row;
    const float gamma = global_decode_scale[0];

    float partial_sum = 0.0F;
    for (std::int64_t k = threadIdx.x; k < reduction_size;
         k += blockDim.x) {
        const std::uint8_t packed =
            packed_values[packed_row_offset + k / 2];
        const std::uint8_t code =
            (k & 1) == 0 ? packed & 0x0FU : packed >> 4;
        const float q = decode_e2m1(code);
        const float beta = decode_ue4m3(
            block_scales[scale_row_offset + k / 16]);

        // Preserve the frozen reconstruction and projection operation order.
        const float block_scaled = __fmul_rn(q, beta);
        const float weight = __fmul_rn(block_scaled, gamma);
        const float activation = __bfloat162float(
            x[activation_row_offset + k]);
        const float product = __fmul_rn(activation, weight);
        partial_sum = __fadd_rn(partial_sum, product);
    }

    // Every launch has 256 active threads (eight complete warps). Threads
    // without a K element retain a zero partial but still join every shuffle.
    partial_sum = warp_sum(partial_sum);

    __shared__ float warp_sums[kWarpsPerBlock];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        warp_sums[warp] = partial_sum;
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

    if (threadIdx.x == 0) {
        output[output_index] = __float2bfloat16_rn(warp_sums[0]);
    }
}

}  // namespace

void launch_w4a16_linear_cuda(
    const at::Tensor& x,
    const at::Tensor& packed_values,
    const at::Tensor& block_scales,
    const at::Tensor& global_decode_scale,
    at::Tensor& output) {
    const std::int64_t output_count = output.numel();
    TORCH_INTERNAL_ASSERT(output_count > 0);
    TORCH_INTERNAL_ASSERT(
        output_count <= std::numeric_limits<std::int32_t>::max());

    const auto* x_data = reinterpret_cast<const __nv_bfloat16*>(
        x.data_ptr<at::BFloat16>());
    const auto* packed_data = packed_values.data_ptr<std::uint8_t>();
    const auto* scale_data = block_scales.data_ptr<std::uint8_t>();
    const auto* global_scale_data = global_decode_scale.data_ptr<float>();
    auto* output_data = reinterpret_cast<__nv_bfloat16*>(
        output.data_ptr<at::BFloat16>());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    w4a16_linear_kernel<<<
        static_cast<unsigned int>(output_count),
        kW4a16BlockThreads,
        0,
        stream>>>(
        x_data,
        packed_data,
        scale_data,
        global_scale_data,
        output_data,
        output_count,
        packed_values.size(0),
        x.size(-1),
        packed_values.size(1),
        block_scales.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cuda_nvfp4_decoder_attention
