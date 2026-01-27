#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

#include <vector>

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() >= 1, "Argmax: vals must have at least 1 dimension");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be int64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(vals->isContiguous(), "Argmax: tensor must be contiguous.");

    const size_t last_dim = vals->shape().back();
    CHECK_ARGUMENT(last_dim > 0, "Argmax: last dimension must be non-zero");
    const size_t outer_size = vals->numel() / last_dim;

    CHECK_ARGUMENT(max_idx->numel() == outer_size && max_val->numel() == outer_size,
                   "Argmax: outputs must match vals shape without the last dimension");

    if (vals->ndim() > 1) {
        std::vector<size_t> expected_shape = vals->shape();
        expected_shape.pop_back();
        CHECK_SAME_SHAPE(max_idx->shape(), expected_shape);
        CHECK_SAME_SHAPE(max_val->shape(), expected_shape);
    }
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), outer_size, last_dim);
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
