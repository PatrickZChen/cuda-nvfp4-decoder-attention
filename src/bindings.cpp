#include "gqa_attention.h"
#include "nvfp4.h"
#include "rmsnorm.h"
#include "rope.h"
#include "w4a16.h"
#include "w4a16_grouped.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

namespace cuda_nvfp4_decoder_attention {

namespace {

void validate_packed_values(const at::Tensor& packed_values) {
    TORCH_CHECK(
        packed_values.is_cuda(),
        "packed_values must be a CUDA tensor");
    TORCH_CHECK(
        packed_values.scalar_type() == at::kByte,
        "packed_values must have dtype torch.uint8");
    TORCH_CHECK(
        packed_values.is_contiguous(),
        "packed_values must be contiguous");
    TORCH_CHECK(
        packed_values.dim() == 2,
        "packed_values must have rank 2, got rank ",
        packed_values.dim());
    TORCH_CHECK(
        packed_values.size(0) > 0 && packed_values.size(1) > 0,
        "packed_values must have nonempty N and packed-byte dimensions");
    TORCH_CHECK(
        packed_values.size(1) <=
            std::numeric_limits<std::int64_t>::max() / 2,
        "packed_values byte dimension is too large to derive logical K");
    TORCH_CHECK(
        packed_values.numel() <=
            std::numeric_limits<std::int64_t>::max() / 2,
        "packed_values has too many bytes to allocate the unpacked output");

    const std::int64_t grid_blocks =
        (packed_values.numel() - 1) / kNvfp4BlockThreads + 1;
    TORCH_CHECK(
        grid_blocks <= std::numeric_limits<std::int32_t>::max(),
        "packed_values is too large for a one-dimensional CUDA grid");
}

std::int64_t checked_positive_product(
    const char* name,
    std::initializer_list<std::int64_t> factors) {
    std::int64_t product = 1;
    for (const std::int64_t factor : factors) {
        TORCH_INTERNAL_ASSERT(factor > 0);
        TORCH_CHECK(
            product <= std::numeric_limits<std::int64_t>::max() / factor,
            name,
            " overflows int64");
        product *= factor;
    }
    return product;
}

void validate_gqa_tensor(const char* name, const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.scalar_type() == at::kBFloat16,
        name,
        " must have dtype torch.bfloat16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.dim() == 4,
        name,
        " must have rank 4, got rank ",
        tensor.dim());
    TORCH_CHECK(
        tensor.size(0) > 0 && tensor.size(1) > 0 &&
            tensor.size(2) > 0 && tensor.size(3) > 0,
        name,
        " dimensions must all be positive");
}

}  // namespace

at::Tensor cuda_unpack_e2m1_codes(const at::Tensor& packed_values) {
    validate_packed_values(packed_values);

    const c10::cuda::CUDAGuard device_guard(packed_values.device());
    at::Tensor output = at::empty(
        {packed_values.size(0), packed_values.size(1) * 2},
        packed_values.options());
    launch_unpack_e2m1_codes_cuda(packed_values, output);
    return output;
}

at::Tensor cuda_dequantize_nvfp4(
    const at::Tensor& packed_values,
    const at::Tensor& block_scales,
    const at::Tensor& global_decode_scale) {
    validate_packed_values(packed_values);
    TORCH_CHECK(
        block_scales.is_cuda(),
        "block_scales must be a CUDA tensor");
    TORCH_CHECK(
        block_scales.scalar_type() == at::kByte,
        "block_scales must have dtype torch.uint8");
    TORCH_CHECK(
        block_scales.is_contiguous(),
        "block_scales must be contiguous");
    TORCH_CHECK(
        block_scales.dim() == 2,
        "block_scales must have rank 2, got rank ",
        block_scales.dim());
    TORCH_CHECK(
        block_scales.device() == packed_values.device(),
        "block_scales must be on the same CUDA device as packed_values");

    TORCH_CHECK(
        global_decode_scale.is_cuda(),
        "global_decode_scale must be a CUDA tensor");
    TORCH_CHECK(
        global_decode_scale.scalar_type() == at::kFloat,
        "global_decode_scale must have dtype torch.float32");
    TORCH_CHECK(
        global_decode_scale.is_contiguous(),
        "global_decode_scale must be contiguous");
    TORCH_CHECK(
        global_decode_scale.dim() == 0,
        "global_decode_scale must be a scalar tensor with shape []");
    TORCH_CHECK(
        global_decode_scale.device() == packed_values.device(),
        "global_decode_scale must be on the same CUDA device as packed_values");

    const std::int64_t rows = packed_values.size(0);
    const std::int64_t columns = packed_values.size(1) * 2;
    TORCH_CHECK(columns >= 16, "logical K must be at least 16, got ", columns);
    TORCH_CHECK(
        columns % 16 == 0,
        "logical K must be divisible by 16, got ",
        columns);
    TORCH_CHECK(
        block_scales.size(0) == rows,
        "block_scales row count must match packed_values (",
        block_scales.size(0),
        " != ",
        rows,
        ")");
    TORCH_CHECK(
        block_scales.size(1) == columns / 16,
        "block_scales must have shape [N, K/16]; expected [",
        rows,
        ", ",
        columns / 16,
        "], got [",
        block_scales.size(0),
        ", ",
        block_scales.size(1),
        "]");

    const c10::cuda::CUDAGuard device_guard(packed_values.device());
    at::Tensor output = at::empty(
        {rows, columns},
        packed_values.options().dtype(at::kFloat));
    launch_dequantize_nvfp4_cuda(
        packed_values,
        block_scales,
        global_decode_scale,
        output);
    return output;
}

at::Tensor cuda_w4a16_linear(
    const at::Tensor& x,
    const at::Tensor& packed_values,
    const at::Tensor& block_scales,
    const at::Tensor& global_decode_scale) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16,
        "x must have dtype torch.bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(
        x.dim() >= 1 && x.dim() <= 4,
        "x must have rank in [1, 4], got rank ",
        x.dim());
    TORCH_CHECK(x.numel() > 0, "x must be nonempty");

    validate_packed_values(packed_values);
    TORCH_CHECK(
        block_scales.is_cuda(),
        "block_scales must be a CUDA tensor");
    TORCH_CHECK(
        block_scales.scalar_type() == at::kByte,
        "block_scales must have dtype torch.uint8");
    TORCH_CHECK(
        block_scales.is_contiguous(),
        "block_scales must be contiguous");
    TORCH_CHECK(
        block_scales.dim() == 2,
        "block_scales must have rank 2, got rank ",
        block_scales.dim());

    TORCH_CHECK(
        global_decode_scale.is_cuda(),
        "global_decode_scale must be a CUDA tensor");
    TORCH_CHECK(
        global_decode_scale.scalar_type() == at::kFloat,
        "global_decode_scale must have dtype torch.float32");
    TORCH_CHECK(
        global_decode_scale.is_contiguous(),
        "global_decode_scale must be contiguous");
    TORCH_CHECK(
        global_decode_scale.dim() == 0,
        "global_decode_scale must be a scalar tensor with shape []");

    TORCH_CHECK(
        packed_values.device() == x.device(),
        "packed_values must be on the same CUDA device as x");
    TORCH_CHECK(
        block_scales.device() == x.device(),
        "block_scales must be on the same CUDA device as x");
    TORCH_CHECK(
        global_decode_scale.device() == x.device(),
        "global_decode_scale must be on the same CUDA device as x");

    const std::int64_t output_features = packed_values.size(0);
    const std::int64_t reduction_size = packed_values.size(1) * 2;
    TORCH_CHECK(
        reduction_size >= 16,
        "logical K must be at least 16, got ",
        reduction_size);
    TORCH_CHECK(
        reduction_size % 16 == 0,
        "logical K must be divisible by 16, got ",
        reduction_size);
    TORCH_CHECK(
        x.size(-1) == reduction_size,
        "x final dimension must match logical weight K (",
        x.size(-1),
        " != ",
        reduction_size,
        ")");
    TORCH_CHECK(
        block_scales.size(0) == output_features,
        "block_scales row count must match packed_values (",
        block_scales.size(0),
        " != ",
        output_features,
        ")");
    TORCH_CHECK(
        block_scales.size(1) == reduction_size / 16,
        "block_scales must have shape [N, K/16]; expected [",
        output_features,
        ", ",
        reduction_size / 16,
        "], got [",
        block_scales.size(0),
        ", ",
        block_scales.size(1),
        "]");

    TORCH_CHECK(
        x.numel() % reduction_size == 0,
        "x element count must be divisible by logical weight K");
    const std::int64_t activation_rows = x.numel() / reduction_size;
    TORCH_CHECK(
        activation_rows <=
            std::numeric_limits<std::int64_t>::max() / output_features,
        "M*N overflows int64");
    const std::int64_t output_count = activation_rows * output_features;
    TORCH_CHECK(
        output_count <= std::numeric_limits<std::int32_t>::max(),
        "M*N is too large for a one-dimensional CUDA grid");

    std::vector<std::int64_t> output_sizes(
        x.sizes().begin(),
        x.sizes().end());
    output_sizes.back() = output_features;

    const c10::cuda::CUDAGuard device_guard(x.device());
    at::Tensor output = at::empty(output_sizes, x.options());
    launch_w4a16_linear_cuda(
        x,
        packed_values,
        block_scales,
        global_decode_scale,
        output);
    return output;
}

at::Tensor cuda_w4a16_linear_grouped_decode(
    const at::Tensor& x,
    const at::Tensor& packed_values,
    const at::Tensor& block_scales,
    const at::Tensor& global_decode_scale) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16,
        "x must have dtype torch.bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(
        x.dim() >= 1 && x.dim() <= 4,
        "x must have rank in [1, 4], got rank ",
        x.dim());
    TORCH_CHECK(x.numel() > 0, "x must be nonempty");

    validate_packed_values(packed_values);
    TORCH_CHECK(
        block_scales.is_cuda(),
        "block_scales must be a CUDA tensor");
    TORCH_CHECK(
        block_scales.scalar_type() == at::kByte,
        "block_scales must have dtype torch.uint8");
    TORCH_CHECK(
        block_scales.is_contiguous(),
        "block_scales must be contiguous");
    TORCH_CHECK(
        block_scales.dim() == 2,
        "block_scales must have rank 2, got rank ",
        block_scales.dim());

    TORCH_CHECK(
        global_decode_scale.is_cuda(),
        "global_decode_scale must be a CUDA tensor");
    TORCH_CHECK(
        global_decode_scale.scalar_type() == at::kFloat,
        "global_decode_scale must have dtype torch.float32");
    TORCH_CHECK(
        global_decode_scale.is_contiguous(),
        "global_decode_scale must be contiguous");
    TORCH_CHECK(
        global_decode_scale.dim() == 0,
        "global_decode_scale must be a scalar tensor with shape []");

    TORCH_CHECK(
        packed_values.device() == x.device(),
        "packed_values must be on the same CUDA device as x");
    TORCH_CHECK(
        block_scales.device() == x.device(),
        "block_scales must be on the same CUDA device as x");
    TORCH_CHECK(
        global_decode_scale.device() == x.device(),
        "global_decode_scale must be on the same CUDA device as x");

    const std::int64_t output_features = packed_values.size(0);
    const std::int64_t reduction_size = packed_values.size(1) * 2;
    TORCH_CHECK(
        reduction_size >= 16,
        "logical K must be at least 16, got ",
        reduction_size);
    TORCH_CHECK(
        reduction_size % 16 == 0,
        "logical K must be divisible by 16, got ",
        reduction_size);
    TORCH_CHECK(
        x.size(-1) == reduction_size,
        "x final dimension must match logical weight K (",
        x.size(-1),
        " != ",
        reduction_size,
        ")");
    TORCH_CHECK(
        block_scales.size(0) == output_features,
        "block_scales row count must match packed_values (",
        block_scales.size(0),
        " != ",
        output_features,
        ")");
    TORCH_CHECK(
        block_scales.size(1) == reduction_size / 16,
        "block_scales must have shape [N, K/16]; expected [",
        output_features,
        ", ",
        reduction_size / 16,
        "], got [",
        block_scales.size(0),
        ", ",
        block_scales.size(1),
        "]");

    TORCH_CHECK(
        x.numel() % reduction_size == 0,
        "x element count must be divisible by logical weight K");
    const std::int64_t activation_rows = x.numel() / reduction_size;
    TORCH_CHECK(
        activation_rows <=
            std::numeric_limits<std::int64_t>::max() / output_features,
        "M*N overflows int64");
    const std::int64_t output_count = activation_rows * output_features;
    TORCH_CHECK(
        output_count <= std::numeric_limits<std::int32_t>::max(),
        "M*N is too large for a one-dimensional CUDA grid");

    std::vector<std::int64_t> output_sizes(
        x.sizes().begin(),
        x.sizes().end());
    output_sizes.back() = output_features;

    const c10::cuda::CUDAGuard device_guard(x.device());
    at::Tensor output = at::empty(output_sizes, x.options());
    launch_w4a16_linear_grouped_decode_cuda(
        x,
        packed_values,
        block_scales,
        global_decode_scale,
        output);
    return output;
}

at::Tensor cuda_rms_norm(
    const at::Tensor& x,
    const at::Tensor& weight,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16,
        "x must have dtype torch.bfloat16");
    TORCH_CHECK(
        weight.scalar_type() == at::kBFloat16,
        "weight must have dtype torch.bfloat16");
    TORCH_CHECK(
        x.device() == weight.device(),
        "x and weight must be on the same CUDA device");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(
        x.dim() >= 1 && x.dim() <= 4,
        "x must have rank in [1, 4], got rank ",
        x.dim());
    TORCH_CHECK(weight.dim() == 1, "weight must be rank 1");
    TORCH_CHECK(x.numel() > 0, "x must be nonempty");
    TORCH_CHECK(weight.numel() > 0, "weight must be nonempty");
    TORCH_CHECK(
        weight.numel() == x.size(-1),
        "weight length must match x final dimension (",
        weight.numel(),
        " != ",
        x.size(-1),
        ")");
    TORCH_CHECK(
        std::isfinite(eps) && eps > 0.0,
        "eps must be finite and positive");
    const float eps_fp32 = static_cast<float>(eps);
    TORCH_CHECK(
        std::isfinite(eps_fp32) && eps_fp32 > 0.0F,
        "eps must be representable as a finite positive FP32 value");

    const std::int64_t rows = x.numel() / x.size(-1);
    TORCH_CHECK(
        rows <= std::numeric_limits<std::int32_t>::max(),
        "x has too many logical rows for a one-dimensional CUDA grid");

    const c10::cuda::CUDAGuard device_guard(x.device());
    at::Tensor output = at::empty_like(x);
    launch_rms_norm_cuda(x, weight, output, eps_fp32);
    return output;
}

at::Tensor cuda_apply_rope(
    const at::Tensor& x,
    std::int64_t past_length,
    double rope_theta) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16,
        "x must have dtype torch.bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must have rank 4, got rank ", x.dim());
    TORCH_CHECK(
        x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 && x.size(3) > 0,
        "x dimensions must all be nonempty");

    const std::int64_t token_count = x.size(1);
    const std::int64_t head_dim = x.size(3);
    TORCH_CHECK(head_dim >= 2, "x head dimension must be at least 2");
    TORCH_CHECK(
        head_dim % 2 == 0,
        "x head dimension must be even for adjacent-pair RoPE");
    TORCH_CHECK(past_length >= 0, "past_length must be nonnegative");
    TORCH_CHECK(
        past_length <=
            std::numeric_limits<std::int64_t>::max() - (token_count - 1),
        "past_length plus token index must fit in int64");
    TORCH_CHECK(
        std::isfinite(rope_theta) && rope_theta > 0.0,
        "rope_theta must be finite and positive");
    const float rope_theta_fp32 = static_cast<float>(rope_theta);
    TORCH_CHECK(
        std::isfinite(rope_theta_fp32) && rope_theta_fp32 > 0.0F,
        "rope_theta must be representable as a finite positive FP32 value");

    const std::int64_t total_pairs = x.numel() / 2;
    const std::int64_t grid_blocks =
        (total_pairs - 1) / kRopeBlockThreads + 1;
    TORCH_CHECK(
        grid_blocks <= std::numeric_limits<std::int32_t>::max(),
        "x has too many logical adjacent pairs for a one-dimensional CUDA grid");

    const c10::cuda::CUDAGuard device_guard(x.device());
    at::Tensor output = at::empty_like(x);
    launch_rope_cuda(x, output, past_length, rope_theta_fp32);
    return output;
}

at::Tensor cuda_gqa_attention(
    const at::Tensor& q,
    const at::Tensor& present_k,
    const at::Tensor& present_v,
    std::int64_t past_length) {
    validate_gqa_tensor("q", q);
    validate_gqa_tensor("present_k", present_k);
    validate_gqa_tensor("present_v", present_v);

    TORCH_CHECK(
        present_v.sizes() == present_k.sizes(),
        "present_v must have the same shape as present_k; got ",
        present_v.sizes(),
        " and ",
        present_k.sizes());
    TORCH_CHECK(
        present_k.device() == q.device(),
        "present_k must be on the same CUDA device as q");
    TORCH_CHECK(
        present_v.device() == q.device(),
        "present_v must be on the same CUDA device as q");

    const std::int64_t batch_size = q.size(0);
    const std::int64_t token_count = q.size(1);
    const std::int64_t query_head_count = q.size(2);
    const std::int64_t head_dim = q.size(3);
    const std::int64_t cache_batch_size = present_k.size(0);
    const std::int64_t kv_head_count = present_k.size(1);
    const std::int64_t context_length = present_k.size(2);
    const std::int64_t cache_head_dim = present_k.size(3);

    TORCH_CHECK(
        batch_size == cache_batch_size,
        "q and cache batch sizes must match (",
        batch_size,
        " != ",
        cache_batch_size,
        ")");
    TORCH_CHECK(
        head_dim == cache_head_dim,
        "q and cache head dimensions must match (",
        head_dim,
        " != ",
        cache_head_dim,
        ")");
    TORCH_CHECK(
        query_head_count % kv_head_count == 0,
        "number of query heads must be divisible by number of KV heads (",
        query_head_count,
        " % ",
        kv_head_count,
        " != 0)");
    TORCH_CHECK(past_length >= 0, "past_length must be nonnegative");
    TORCH_CHECK(
        past_length <=
            std::numeric_limits<std::int64_t>::max() - token_count,
        "past_length + current token count overflows int64");
    TORCH_CHECK(
        context_length == past_length + token_count,
        "cache context length must equal past_length + current token count (",
        context_length,
        " != ",
        past_length,
        " + ",
        token_count,
        ")");

    const std::int64_t softmax_rows = checked_positive_product(
        "B*Hq*T",
        {batch_size, query_head_count, token_count});
    const std::int64_t score_count = checked_positive_product(
        "B*Hq*T*S",
        {batch_size, query_head_count, token_count, context_length});
    const std::int64_t context_count = checked_positive_product(
        "B*T*Hq*D",
        {batch_size, token_count, query_head_count, head_dim});
    constexpr std::int64_t kMaximumOneDimensionalGrid =
        std::numeric_limits<std::int32_t>::max();
    TORCH_CHECK(
        score_count <= kMaximumOneDimensionalGrid,
        "B*Hq*T*S is too large for a one-dimensional CUDA grid");
    TORCH_CHECK(
        softmax_rows <= kMaximumOneDimensionalGrid,
        "B*Hq*T is too large for a one-dimensional CUDA grid");
    TORCH_CHECK(
        context_count <= kMaximumOneDimensionalGrid,
        "B*T*Hq*D is too large for a one-dimensional CUDA grid");

    // Match the reference's double-precision metadata calculation followed by
    // one explicit FP32 storage boundary before the QK kernel launch.
    const float inverse_sqrt_head_dim = static_cast<float>(
        1.0 / std::sqrt(static_cast<double>(head_dim)));
    TORCH_CHECK(
        std::isfinite(inverse_sqrt_head_dim) &&
            inverse_sqrt_head_dim > 0.0F,
        "1/sqrt(D) must be representable as a finite positive FP32 value");

    const c10::cuda::CUDAGuard device_guard(q.device());
    at::Tensor scores = at::empty(
        {batch_size, query_head_count, token_count, context_length},
        q.options().dtype(at::kFloat));
    at::Tensor probabilities = at::empty(
        {batch_size, query_head_count, token_count, context_length},
        q.options().dtype(at::kFloat));
    at::Tensor context = at::empty(
        {batch_size, token_count, query_head_count, head_dim},
        q.options());
    launch_gqa_attention_cuda(
        q,
        present_k,
        present_v,
        scores,
        probabilities,
        context,
        past_length,
        inverse_sqrt_head_dim);
    return context;
}

}  // namespace cuda_nvfp4_decoder_attention

TORCH_LIBRARY(cuda_nvfp4_decoder_attention, module) {
    module.def(
        "cuda_unpack_e2m1_codes(Tensor packed_values) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_unpack_e2m1_codes));
    module.def(
        "cuda_dequantize_nvfp4(Tensor packed_values, Tensor block_scales, Tensor global_decode_scale) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_dequantize_nvfp4));
    module.def(
        "cuda_rms_norm(Tensor x, Tensor weight, float eps) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_rms_norm));
    module.def(
        "cuda_apply_rope(Tensor x, int past_length, float rope_theta=10000.0) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_apply_rope));
    module.def(
        "cuda_w4a16_linear(Tensor x, Tensor packed_values, Tensor block_scales, Tensor global_decode_scale) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_w4a16_linear));
    module.def(
        "cuda_w4a16_linear_grouped_decode(Tensor x, Tensor packed_values, Tensor block_scales, Tensor global_decode_scale) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_w4a16_linear_grouped_decode));
    module.def(
        "cuda_gqa_attention(Tensor q, Tensor present_k, Tensor present_v, int past_length) -> Tensor",
        TORCH_FN(cuda_nvfp4_decoder_attention::cuda_gqa_attention));
}
