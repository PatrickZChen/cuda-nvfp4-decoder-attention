#include "nvfp4.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace cuda_nvfp4_decoder_attention {
namespace {

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

    // Applying the sign bit at the bit level preserves the distinct -0 code.
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

__global__ void unpack_e2m1_codes_kernel(
    const std::uint8_t* packed_values,
    std::uint8_t* output,
    std::int64_t packed_byte_count) {
    const std::int64_t packed_index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (packed_index >= packed_byte_count) {
        return;
    }

    const std::uint8_t packed = packed_values[packed_index];
    const std::int64_t logical_offset = 2 * packed_index;
    output[logical_offset] = packed & 0x0FU;
    output[logical_offset + 1] = packed >> 4;
}

__global__ void dequantize_nvfp4_kernel(
    const std::uint8_t* packed_values,
    const std::uint8_t* block_scales,
    const float* global_decode_scale,
    float* output,
    std::int64_t packed_byte_count,
    std::int64_t packed_bytes_per_row,
    std::int64_t blocks_per_row) {
    const std::int64_t packed_index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (packed_index >= packed_byte_count) {
        return;
    }

    // A contiguous packed index decomposes into one row and one byte column.
    // Byte r owns logical columns 2*r and 2*r+1. Eight packed bytes are the
    // 16 logical elements in one row-local microscaling block, so r/8 is the
    // block column and cannot cross a row.
    const std::int64_t row = packed_index / packed_bytes_per_row;
    const std::int64_t packed_byte_column =
        packed_index - row * packed_bytes_per_row;
    const std::int64_t block_column = packed_byte_column / 8;
    const std::int64_t scale_index = row * blocks_per_row + block_column;

    const std::uint8_t packed = packed_values[packed_index];
    const float low = decode_e2m1(packed & 0x0FU);
    const float high = decode_e2m1(packed >> 4);
    const float beta = decode_ue4m3(block_scales[scale_index]);
    const float gamma = global_decode_scale[0];

    // Keep the repository's frozen FP32 evaluation order explicit.
    const float low_block_scaled = __fmul_rn(low, beta);
    const float high_block_scaled = __fmul_rn(high, beta);
    const std::int64_t logical_offset = 2 * packed_index;
    output[logical_offset] = __fmul_rn(low_block_scaled, gamma);
    output[logical_offset + 1] = __fmul_rn(high_block_scaled, gamma);
}

}  // namespace

void launch_unpack_e2m1_codes_cuda(
    const at::Tensor& packed_values,
    at::Tensor& output) {
    const std::int64_t packed_byte_count = packed_values.numel();
    const std::int64_t grid_blocks =
        (packed_byte_count - 1) / kNvfp4BlockThreads + 1;
    TORCH_INTERNAL_ASSERT(packed_byte_count > 0);
    TORCH_INTERNAL_ASSERT(
        grid_blocks <= std::numeric_limits<std::int32_t>::max());

    const auto* packed_data = packed_values.data_ptr<std::uint8_t>();
    auto* output_data = output.data_ptr<std::uint8_t>();
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(packed_values.get_device());

    unpack_e2m1_codes_kernel<<<
        static_cast<unsigned int>(grid_blocks),
        kNvfp4BlockThreads,
        0,
        stream>>>(packed_data, output_data, packed_byte_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_dequantize_nvfp4_cuda(
    const at::Tensor& packed_values,
    const at::Tensor& block_scales,
    const at::Tensor& global_decode_scale,
    at::Tensor& output) {
    const std::int64_t packed_byte_count = packed_values.numel();
    const std::int64_t grid_blocks =
        (packed_byte_count - 1) / kNvfp4BlockThreads + 1;
    TORCH_INTERNAL_ASSERT(packed_byte_count > 0);
    TORCH_INTERNAL_ASSERT(
        grid_blocks <= std::numeric_limits<std::int32_t>::max());

    const auto* packed_data = packed_values.data_ptr<std::uint8_t>();
    const auto* scale_data = block_scales.data_ptr<std::uint8_t>();
    const auto* global_scale_data = global_decode_scale.data_ptr<float>();
    auto* output_data = output.data_ptr<float>();
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(packed_values.get_device());

    dequantize_nvfp4_kernel<<<
        static_cast<unsigned int>(grid_blocks),
        kNvfp4BlockThreads,
        0,
        stream>>>(
        packed_data,
        scale_data,
        global_scale_data,
        output_data,
        packed_byte_count,
        packed_values.size(1),
        block_scales.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cuda_nvfp4_decoder_attention
