#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/add_cpu.hpp"

namespace llaisys::ops {
void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(), "Add: all tensors must be contiguous.");

    CHECK_ARGUMENT(c->ndim() >= 1 && a->ndim() >= 1 && b->ndim() >= 1, "Add: tensors must have at least 1 dimension");
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());

    const size_t last_dim = c->shape().back();
    CHECK_ARGUMENT(last_dim > 0, "Add: last dimension must be non-zero");
    const size_t outer_size = c->numel() / last_dim;

    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());

    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), outer_size, last_dim);
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
