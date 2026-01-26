#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t n, size_t m);
}
