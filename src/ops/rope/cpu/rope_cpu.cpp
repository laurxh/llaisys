#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {
template <typename T>
void rope_impl_(T *out,
                const T *in,
                const int64_t *pos_ids,
                size_t outer_pos,
                size_t nhead,
                size_t head_dim,
                float theta) {
    const size_t half_dim = head_dim / 2;
    const size_t pos_stride = nhead * head_dim;
    const double d = static_cast<double>(head_dim);

    for (size_t pos = 0; pos < outer_pos; ++pos) {
        const double p = static_cast<double>(pos_ids[pos]);
        const size_t pos_base = pos * pos_stride;

        for (size_t h = 0; h < nhead; ++h) {
            const size_t head_base = pos_base + h * head_dim;

            for (size_t j = 0; j < half_dim; ++j) {
                const double exponent = (2.0 * static_cast<double>(j)) / d;
                const double inv_freq = std::pow(static_cast<double>(theta), -exponent);
                const double phi = p * inv_freq;
                const float s = static_cast<float>(std::sin(phi));
                const float c = static_cast<float>(std::cos(phi));

                const double a = llaisys::utils::cast<double>(in[head_base + j]);
                const double b = llaisys::utils::cast<double>(in[head_base + half_dim + j]);

                out[head_base + j] = llaisys::utils::cast<T>(a * c - b * s);
                out[head_base + half_dim + j] = llaisys::utils::cast<T>(b * c + a * s);
            }
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos_ids,
          llaisysDataType_t type,
          size_t outer_pos,
          size_t nhead,
          size_t head_dim,
          float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_impl_(reinterpret_cast<float *>(out),
                          reinterpret_cast<const float *>(in),
                          pos_ids,
                          outer_pos,
                          nhead,
                          head_dim,
                          theta);
    case LLAISYS_DTYPE_BF16:
        return rope_impl_(reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const llaisys::bf16_t *>(in),
                          pos_ids,
                          outer_pos,
                          nhead,
                          head_dim,
                          theta);
    case LLAISYS_DTYPE_F16:
        return rope_impl_(reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const llaisys::fp16_t *>(in),
                          pos_ids,
                          outer_pos,
                          nhead,
                          head_dim,
                          theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu