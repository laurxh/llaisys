#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <numeric>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out,
               const std::byte *in,
               llaisysDataType_t type,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides) {
    const size_t ndim = shape.size();
    const size_t bytes = utils::dsize(type);
    const size_t numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());

    if (ndim == 0 || numel == 0) {
        return;
    }

    std::vector<size_t> idx(ndim, 0);
    for (size_t linear = 0; linear < numel; ++linear) {
        ptrdiff_t out_off = 0;
        ptrdiff_t in_off = 0;
        for (size_t d = 0; d < ndim; ++d) {
            out_off += static_cast<ptrdiff_t>(idx[d]) * out_strides[d];
            in_off += static_cast<ptrdiff_t>(idx[d]) * in_strides[d];
        }

        std::memcpy(out + static_cast<size_t>(out_off) * bytes,
                    in + static_cast<size_t>(in_off) * bytes,
                    bytes);

        for (size_t d = ndim; d-- > 0;) {
            idx[d] += 1;
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
}
} // namespace llaisys::ops::cpu

