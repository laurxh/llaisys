#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              llaisysDataType_t type,
              size_t outer_size,
              size_t last_dim,
              float eps);
}

