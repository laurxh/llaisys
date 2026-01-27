#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos_ids,
          llaisysDataType_t type,
          size_t outer_pos,
          size_t nhead,
          size_t head_dim,
          float theta);
}

