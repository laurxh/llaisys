#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void swiglu(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            llaisysDataType_t type,
            size_t numel);
}

