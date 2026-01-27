#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void argmax_lastdim_(int64_t *max_idx, T *max_val, const T *vals, size_t outer_size, size_t last_dim) {
    for (size_t outer = 0; outer < outer_size; ++outer) {
        const size_t base = outer * last_dim;
        size_t best = 0;
        float best_val = llaisys::utils::cast<float>(vals[base]);
        for (size_t i = 1; i < last_dim; ++i) {
            float v = llaisys::utils::cast<float>(vals[base + i]);
            if (v > best_val) {
                best_val = v;
                best = i;
            }
        }
        max_idx[outer] = static_cast<int64_t>(best);
        max_val[outer] = llaisys::utils::cast<T>(best_val);
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t outer_size,
            size_t last_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_lastdim_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
                               reinterpret_cast<const float *>(vals), outer_size, last_dim);
    case LLAISYS_DTYPE_BF16:
        return argmax_lastdim_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                               reinterpret_cast<const llaisys::bf16_t *>(vals), outer_size, last_dim);
    case LLAISYS_DTYPE_F16:
        return argmax_lastdim_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                               reinterpret_cast<const llaisys::fp16_t *>(vals), outer_size, last_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
