#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
template <typename T>
void self_attention_impl_(T *attn_val,
                          const T *q,
                          const T *k,
                          const T *v,
                          size_t outer_size,
                          size_t qlen,
                          size_t kvlen,
                          size_t nhead,
                          size_t nkvhead,
                          size_t head_dim,
                          size_t value_dim,
                          float scale) {
    const size_t q_batch_stride = qlen * nhead * head_dim;
    const size_t k_batch_stride = kvlen * nkvhead * head_dim;
    const size_t v_batch_stride = kvlen * nkvhead * value_dim;
    const size_t out_batch_stride = qlen * nhead * value_dim;

    const size_t head_repeat = nhead / nkvhead;
    const int64_t causal_offset = static_cast<int64_t>(kvlen) - static_cast<int64_t>(qlen);

    std::vector<float> scores(kvlen);
    std::vector<float> probs(kvlen);

    for (size_t outer = 0; outer < outer_size; ++outer) {
        const T *q_batch = q + outer * q_batch_stride;
        const T *k_batch = k + outer * k_batch_stride;
        const T *v_batch = v + outer * v_batch_stride;
        T *out_batch = attn_val + outer * out_batch_stride;

        for (size_t h = 0; h < nhead; ++h) {
            const size_t kv_head = h / head_repeat;

            for (size_t qi = 0; qi < qlen; ++qi) {
                const int64_t max_k = std::min<int64_t>(static_cast<int64_t>(kvlen) - 1,
                                                        static_cast<int64_t>(qi) + causal_offset);
                if (max_k < 0) {
                    for (size_t vd = 0; vd < value_dim; ++vd) {
                        out_batch[(qi * nhead + h) * value_dim + vd] = llaisys::utils::cast<T>(0.0f);
                    }
                    continue;
                }

                const size_t q_base = (qi * nhead + h) * head_dim;

                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t ki = 0; ki < kvlen; ++ki) {
                    float score = -std::numeric_limits<float>::infinity();
                    if (static_cast<int64_t>(ki) <= max_k) {
                        const size_t k_base = (ki * nkvhead + kv_head) * head_dim;
                        float dot = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d) {
                            const float qv = llaisys::utils::cast<float>(q_batch[q_base + d]);
                            const float kv = llaisys::utils::cast<float>(k_batch[k_base + d]);
                            dot += qv * kv;
                        }
                        score = dot * scale;
                    }
                    scores[ki] = score;
                    if (score > max_score) {
                        max_score = score;
                    }
                }

                float denom = 0.0f;
                for (size_t ki = 0; ki < kvlen; ++ki) {
                    const float p = std::exp(scores[ki] - max_score);
                    probs[ki] = p;
                    denom += p;
                }
                const float inv_denom = 1.0f / denom;

                const size_t out_base = (qi * nhead + h) * value_dim;
                for (size_t vd = 0; vd < value_dim; ++vd) {
                    float acc = 0.0f;
                    for (size_t ki = 0; ki < kvlen; ++ki) {
                        const size_t v_base = (ki * nkvhead + kv_head) * value_dim;
                        const float vv = llaisys::utils::cast<float>(v_batch[v_base + vd]);
                        acc += (probs[ki] * inv_denom) * vv;
                    }
                    out_batch[out_base + vd] = llaisys::utils::cast<T>(acc);
                }
            }
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t type,
                    size_t outer_size,
                    size_t qlen,
                    size_t kvlen,
                    size_t nhead,
                    size_t nkvhead,
                    size_t head_dim,
                    size_t value_dim,
                    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_impl_(reinterpret_cast<float *>(attn_val),
                                    reinterpret_cast<const float *>(q),
                                    reinterpret_cast<const float *>(k),
                                    reinterpret_cast<const float *>(v),
                                    outer_size,
                                    qlen,
                                    kvlen,
                                    nhead,
                                    nkvhead,
                                    head_dim,
                                    value_dim,
                                    scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_impl_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                                    reinterpret_cast<const llaisys::bf16_t *>(q),
                                    reinterpret_cast<const llaisys::bf16_t *>(k),
                                    reinterpret_cast<const llaisys::bf16_t *>(v),
                                    outer_size,
                                    qlen,
                                    kvlen,
                                    nhead,
                                    nkvhead,
                                    head_dim,
                                    value_dim,
                                    scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_impl_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                                    reinterpret_cast<const llaisys::fp16_t *>(q),
                                    reinterpret_cast<const llaisys::fp16_t *>(k),
                                    reinterpret_cast<const llaisys::fp16_t *>(v),
                                    outer_size,
                                    qlen,
                                    kvlen,
                                    nhead,
                                    nkvhead,
                                    head_dim,
                                    value_dim,
                                    scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

