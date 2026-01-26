#include "embedding_cpu.hpp"

#include "../../../utils.hpp"
#include <cstring>
namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t n, size_t m) {
    const int64_t *row_index = reinterpret_cast<const int64_t *>(index);
    size_t bytes = utils::dsize(type);
    for (size_t i = 0; i < n; i++) {
        memcpy(out + i * m * bytes, weight + row_index[i] * m * bytes, m * bytes);
    }
}
} // namespace llaisys::ops::cpu
