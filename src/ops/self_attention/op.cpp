#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

#include <vector>

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    CHECK_ARGUMENT(q->ndim() >= 3, "SelfAttention: q must have at least 3 dimensions");
    CHECK_ARGUMENT(k->ndim() == q->ndim(), "SelfAttention: k must have the same ndim as q");
    CHECK_ARGUMENT(v->ndim() == q->ndim(), "SelfAttention: v must have the same ndim as q");

    const size_t qlen = q->shape()[q->ndim() - 3];
    const size_t nhead = q->shape()[q->ndim() - 2];
    const size_t head_dim = q->shape().back();
    CHECK_ARGUMENT(qlen > 0 && nhead > 0 && head_dim > 0, "SelfAttention: q dimensions must be non-zero");

    const size_t kvlen = k->shape()[k->ndim() - 3];
    const size_t nkvhead = k->shape()[k->ndim() - 2];
    const size_t k_head_dim = k->shape().back();
    CHECK_ARGUMENT(kvlen > 0 && nkvhead > 0 && k_head_dim > 0, "SelfAttention: k dimensions must be non-zero");
    CHECK_ARGUMENT(head_dim == k_head_dim, "SelfAttention: q and k head_dim must match");

    const size_t v_kvlen = v->shape()[v->ndim() - 3];
    const size_t v_nkvhead = v->shape()[v->ndim() - 2];
    const size_t value_dim = v->shape().back();
    CHECK_ARGUMENT(v_kvlen == kvlen && v_nkvhead == nkvhead,
                   "SelfAttention: v must match k on kv length and kv heads");
    CHECK_ARGUMENT(value_dim > 0, "SelfAttention: value_dim must be non-zero");

    CHECK_ARGUMENT(nhead % nkvhead == 0, "SelfAttention: nhead must be a multiple of nkvhead");

    std::vector<size_t> prefix_q(q->shape().begin(), q->shape().end() - 3);
    std::vector<size_t> prefix_k(k->shape().begin(), k->shape().end() - 3);
    std::vector<size_t> prefix_v(v->shape().begin(), v->shape().end() - 3);
    CHECK_SAME_SHAPE(prefix_q, prefix_k);
    CHECK_SAME_SHAPE(prefix_q, prefix_v);

    const size_t outer_size = q->numel() / (qlen * nhead * head_dim);

    std::vector<size_t> expected_out_shape = prefix_q;
    expected_out_shape.push_back(qlen);
    expected_out_shape.push_back(nhead);
    expected_out_shape.push_back(value_dim);
    CHECK_SAME_SHAPE(attn_val->shape(), expected_out_shape);

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(),
                                   q->data(),
                                   k->data(),
                                   v->data(),
                                   attn_val->dtype(),
                                   outer_size,
                                   qlen,
                                   kvlen,
                                   nhead,
                                   nkvhead,
                                   head_dim,
                                   value_dim,
                                   scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
