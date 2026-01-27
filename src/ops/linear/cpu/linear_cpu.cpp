#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace {
template <typename T>
void linear_last2d_(T *out,
                    const T *in,
                    const T *weight,
                    const T *bias,
                    size_t outer_size,
                    size_t m,
                    size_t k,
                    size_t n,
                    bool has_bias) {
    const size_t in_batch_stride = m * k;
    const size_t out_batch_stride = m * n;

    for (size_t outer = 0; outer < outer_size; ++outer) {
        const T *in_batch = in + outer * in_batch_stride;
        T *out_batch = out + outer * out_batch_stride;

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float acc = has_bias ? llaisys::utils::cast<float>(bias[j]) : 0.0f;
                for (size_t kk = 0; kk < k; ++kk) {
                    const float x = llaisys::utils::cast<float>(in_batch[i * k + kk]);
                    const float w = llaisys::utils::cast<float>(weight[j * k + kk]);
                    acc += x * w;
                }
                out_batch[i * n + j] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t type,
            size_t outer_size,
            size_t m,
            size_t k,
            size_t n,
            bool has_bias) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_last2d_(reinterpret_cast<float *>(out),
                              reinterpret_cast<const float *>(in),
                              reinterpret_cast<const float *>(weight),
                              reinterpret_cast<const float *>(bias),
                              outer_size,
                              m,
                              k,
                              n,
                              has_bias);
    case LLAISYS_DTYPE_BF16:
        return linear_last2d_(reinterpret_cast<llaisys::bf16_t *>(out),
                              reinterpret_cast<const llaisys::bf16_t *>(in),
                              reinterpret_cast<const llaisys::bf16_t *>(weight),
                              reinterpret_cast<const llaisys::bf16_t *>(bias),
                              outer_size,
                              m,
                              k,
                              n,
                              has_bias);
    case LLAISYS_DTYPE_F16:
        return linear_last2d_(reinterpret_cast<llaisys::fp16_t *>(out),
                              reinterpret_cast<const llaisys::fp16_t *>(in),
                              reinterpret_cast<const llaisys::fp16_t *>(weight),
                              reinterpret_cast<const llaisys::fp16_t *>(bias),
                              outer_size,
                              m,
                              k,
                              n,
                              has_bias);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
