#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t type,
            size_t outer_size,
            size_t m,
            size_t k,
            size_t n,
            bool has_bias);
}

