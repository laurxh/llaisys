#include "add_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <type_traits>

template <typename T>
void add_lastdim_(T *c, const T *a, const T *b, size_t outer_size, size_t last_dim) {
    const size_t total = outer_size * last_dim;
    for (size_t idx = 0; idx < total; ++idx) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            c[idx] =
                llaisys::utils::cast<T>(llaisys::utils::cast<float>(a[idx]) + llaisys::utils::cast<float>(b[idx]));
        } else {
            c[idx] = a[idx] + b[idx];
        }
    }
}

namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t outer_size,
         size_t last_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_lastdim_(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
                            reinterpret_cast<const float *>(b), outer_size, last_dim);
    case LLAISYS_DTYPE_BF16:
        return add_lastdim_(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
                            reinterpret_cast<const llaisys::bf16_t *>(b), outer_size, last_dim);
    case LLAISYS_DTYPE_F16:
        return add_lastdim_(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
                            reinterpret_cast<const llaisys::fp16_t *>(b), outer_size, last_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
