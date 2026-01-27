#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {
template <typename T>
void rms_norm_lastdim_(T *out,
                       const T *in,
                       const T *weight,
                       size_t outer_size,
                       size_t last_dim,
                       float eps) {
    for (size_t outer = 0; outer < outer_size; ++outer) {
        const size_t base = outer * last_dim;

        float sum_sq = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            const float v = llaisys::utils::cast<float>(in[base + i]);
            sum_sq += v * v;
        }

        const float mean_sq = sum_sq / static_cast<float>(last_dim);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        for (size_t i = 0; i < last_dim; ++i) {
            const float v = llaisys::utils::cast<float>(in[base + i]);
            const float w = llaisys::utils::cast<float>(weight[i]);
            out[base + i] = llaisys::utils::cast<T>(v * inv_rms * w);
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              llaisysDataType_t type,
              size_t outer_size,
              size_t last_dim,
              float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_lastdim_(reinterpret_cast<float *>(out),
                                 reinterpret_cast<const float *>(in),
                                 reinterpret_cast<const float *>(weight),
                                 outer_size,
                                 last_dim,
                                 eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_lastdim_(reinterpret_cast<llaisys::bf16_t *>(out),
                                 reinterpret_cast<const llaisys::bf16_t *>(in),
                                 reinterpret_cast<const llaisys::bf16_t *>(weight),
                                 outer_size,
                                 last_dim,
                                 eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_lastdim_(reinterpret_cast<llaisys::fp16_t *>(out),
                                 reinterpret_cast<const llaisys::fp16_t *>(in),
                                 reinterpret_cast<const llaisys::fp16_t *>(weight),
                                 outer_size,
                                 last_dim,
                                 eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

