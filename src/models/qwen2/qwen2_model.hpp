#pragma once

#include "llaisys/models/qwen2.h"

#include "../../llaisys/llaisys_tensor.hpp"
#include "../../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace llaisys::models {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model();

    LlaisysQwen2Weights *weights();
    int64_t infer(const int64_t *token_ids, size_t ntoken);

private:
    struct LayerCache {
        tensor_t k;
        tensor_t v;
    };

    LlaisysQwen2Meta meta_{};
    llaisysDeviceType_t device_ = LLAISYS_DEVICE_CPU;
    int device_id_ = 0;

    LlaisysQwen2Weights weights_{};
    std::vector<LlaisysTensor *> owned_tensors_;

    std::unique_ptr<llaisysTensor_t[]> attn_norm_w_;
    std::unique_ptr<llaisysTensor_t[]> attn_q_w_;
    std::unique_ptr<llaisysTensor_t[]> attn_q_b_;
    std::unique_ptr<llaisysTensor_t[]> attn_k_w_;
    std::unique_ptr<llaisysTensor_t[]> attn_k_b_;
    std::unique_ptr<llaisysTensor_t[]> attn_v_w_;
    std::unique_ptr<llaisysTensor_t[]> attn_v_b_;
    std::unique_ptr<llaisysTensor_t[]> attn_o_w_;
    std::unique_ptr<llaisysTensor_t[]> mlp_norm_w_;
    std::unique_ptr<llaisysTensor_t[]> mlp_gate_w_;
    std::unique_ptr<llaisysTensor_t[]> mlp_up_w_;
    std::unique_ptr<llaisysTensor_t[]> mlp_down_w_;

    std::vector<LayerCache> cache_;
    size_t cur_len_ = 0;

    llaisysTensor_t create_weight(const std::vector<size_t> &shape);
    void zero_tensor(llaisysTensor_t tensor);
    void write_cache(const tensor_t &cache, const tensor_t &src, size_t pos);
};

} // namespace llaisys::models
