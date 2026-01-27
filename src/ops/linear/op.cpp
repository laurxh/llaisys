#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

#include <vector>

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    const bool has_bias = static_cast<bool>(bias);

    if (has_bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(),
               "Linear: all tensors must be contiguous.");
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
               "Linear: out/in/weight must be contiguous when bias is not provided.");
    }

    CHECK_ARGUMENT(in->ndim() >= 2, "Linear: input must have at least 2 dimensions");
    CHECK_ARGUMENT(weight->ndim() == 2, "Linear: weight must be 2D [out_features, in_features]");

    const size_t k = in->shape().back();
    const size_t m = in->shape()[in->ndim() - 2];
    const size_t n = weight->shape()[0];
    const size_t weight_k = weight->shape()[1];

    CHECK_ARGUMENT(k == weight_k, "Linear: input last dim must match weight in_features");
    if (has_bias) {
        CHECK_ARGUMENT(bias->ndim() == 1, "Linear: bias must be 1D [out_features]");
        CHECK_ARGUMENT(bias->shape()[0] == n, "Linear: bias size must match weight out_features");
    }

    const size_t outer_size = in->numel() / (m * k);
    CHECK_ARGUMENT(outer_size > 0, "Linear: outer size must be non-zero");

    std::vector<size_t> expected_out_shape = in->shape();
    expected_out_shape.back() = n;
    CHECK_SAME_SHAPE(out->shape(), expected_out_shape);
    CHECK_ARGUMENT(out->numel() == outer_size * m * n,
                   "Linear: output numel must equal outer_size * m * out_features");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(),
                           in->data(),
                           weight->data(),
                           has_bias ? bias->data() : nullptr,
                           out->dtype(),
                           outer_size,
                           m,
                           k,
                           n,
                           has_bias);
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
