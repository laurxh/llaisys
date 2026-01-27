#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

#include <vector>

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    CHECK_ARGUMENT(in->ndim() >= 2, "RoPE: input must have at least 2 dimensions");
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    const size_t nhead = in->shape()[in->ndim() - 2];
    const size_t head_dim = in->shape().back();
    CHECK_ARGUMENT(nhead > 0, "RoPE: number of heads must be non-zero");
    CHECK_ARGUMENT(head_dim > 0 && head_dim % 2 == 0, "RoPE: head_dim must be positive and even");

    const size_t outer_pos = in->numel() / (nhead * head_dim);
    CHECK_ARGUMENT(pos_ids->numel() == outer_pos, "RoPE: pos_ids must match all position dimensions");

    if (pos_ids->ndim() == in->ndim() - 2) {
        std::vector<size_t> expected_pos_shape(in->shape().begin(), in->shape().end() - 2);
        CHECK_SAME_SHAPE(pos_ids->shape(), expected_pos_shape);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(),
                         in->data(),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         out->dtype(),
                         outer_pos,
                         nhead,
                         head_dim,
                         theta);
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
