#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RmsNorm: all tensors must be contiguous.");

    CHECK_ARGUMENT(in->ndim() >= 1, "RmsNorm: input must have at least 1 dimension");
    CHECK_ARGUMENT(weight->ndim() == 1, "RmsNorm: weight must be 1D");
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    const size_t last_dim = in->shape().back();
    CHECK_ARGUMENT(last_dim > 0, "RmsNorm: last dimension must be non-zero");
    CHECK_ARGUMENT(weight->shape()[0] == last_dim, "RmsNorm: weight size must match input last dimension");

    const size_t outer_size = in->numel() / last_dim;

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), outer_size, last_dim, eps);
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
