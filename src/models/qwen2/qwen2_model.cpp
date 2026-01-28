#include "qwen2_model.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::models {
namespace {
void check_meta(const LlaisysQwen2Meta &meta) {
    CHECK_ARGUMENT(meta.nlayer > 0, "Qwen2: nlayer must be > 0");
    CHECK_ARGUMENT(meta.hs > 0, "Qwen2: hidden size must be > 0");
    CHECK_ARGUMENT(meta.nh > 0, "Qwen2: num heads must be > 0");
    CHECK_ARGUMENT(meta.dh > 0, "Qwen2: head dim must be > 0");
    CHECK_ARGUMENT(meta.nkvh > 0, "Qwen2: num kv heads must be > 0");
    CHECK_ARGUMENT(meta.di > 0, "Qwen2: intermediate size must be > 0");
    CHECK_ARGUMENT(meta.maxseq > 0, "Qwen2: maxseq must be > 0");
    CHECK_ARGUMENT(meta.voc > 0, "Qwen2: vocab size must be > 0");
    CHECK_ARGUMENT(meta.hs == meta.nh * meta.dh, "Qwen2: hs must equal nh * dh");
}
} // namespace

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id)
    : meta_(meta), device_(device), device_id_(device_id) {
    check_meta(meta_);
    CHECK_ARGUMENT(device_ == LLAISYS_DEVICE_CPU, "Qwen2: only CPU is supported for now");

    attn_norm_w_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);
    attn_q_w_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);
    attn_q_b_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);
    attn_k_w_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);
    attn_k_b_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);
    attn_v_w_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);
    attn_v_b_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);
    attn_o_w_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);
    mlp_norm_w_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);
    mlp_gate_w_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);
    mlp_up_w_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);
    mlp_down_w_ = std::make_unique<llaisysTensor_t[]>(meta_.nlayer);

    weights_.in_embed = create_weight({meta_.voc, meta_.hs});
    weights_.out_embed = create_weight({meta_.voc, meta_.hs});
    weights_.out_norm_w = create_weight({meta_.hs});

    weights_.attn_norm_w = attn_norm_w_.get();
    weights_.attn_q_w = attn_q_w_.get();
    weights_.attn_q_b = attn_q_b_.get();
    weights_.attn_k_w = attn_k_w_.get();
    weights_.attn_k_b = attn_k_b_.get();
    weights_.attn_v_w = attn_v_w_.get();
    weights_.attn_v_b = attn_v_b_.get();
    weights_.attn_o_w = attn_o_w_.get();
    weights_.mlp_norm_w = mlp_norm_w_.get();
    weights_.mlp_gate_w = mlp_gate_w_.get();
    weights_.mlp_up_w = mlp_up_w_.get();
    weights_.mlp_down_w = mlp_down_w_.get();

    for (size_t i = 0; i < meta_.nlayer; ++i) {
        weights_.attn_norm_w[i] = create_weight({meta_.hs});
        weights_.attn_q_w[i] = create_weight({meta_.nh * meta_.dh, meta_.hs});
        weights_.attn_q_b[i] = create_weight({meta_.nh * meta_.dh});
        weights_.attn_k_w[i] = create_weight({meta_.nkvh * meta_.dh, meta_.hs});
        weights_.attn_k_b[i] = create_weight({meta_.nkvh * meta_.dh});
        weights_.attn_v_w[i] = create_weight({meta_.nkvh * meta_.dh, meta_.hs});
        weights_.attn_v_b[i] = create_weight({meta_.nkvh * meta_.dh});
        weights_.attn_o_w[i] = create_weight({meta_.hs, meta_.nh * meta_.dh});
        weights_.mlp_norm_w[i] = create_weight({meta_.hs});
        weights_.mlp_gate_w[i] = create_weight({meta_.di, meta_.hs});
        weights_.mlp_up_w[i] = create_weight({meta_.di, meta_.hs});
        weights_.mlp_down_w[i] = create_weight({meta_.hs, meta_.di});

        zero_tensor(weights_.attn_q_b[i]);
        zero_tensor(weights_.attn_k_b[i]);
        zero_tensor(weights_.attn_v_b[i]);
    }

    cache_.resize(meta_.nlayer);
    for (size_t i = 0; i < meta_.nlayer; ++i) {
        cache_[i].k = Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, meta_.dtype, device_, device_id_);
        cache_[i].v = Tensor::create({meta_.maxseq, meta_.nkvh, meta_.dh}, meta_.dtype, device_, device_id_);
    }
}

Qwen2Model::~Qwen2Model() {
    for (auto *tensor : owned_tensors_) {
        delete tensor;
    }
}

LlaisysQwen2Weights *Qwen2Model::weights() { return &weights_; }

llaisysTensor_t Qwen2Model::create_weight(const std::vector<size_t> &shape) {
    auto *tensor = new LlaisysTensor{Tensor::create(shape, meta_.dtype, device_, device_id_)};
    owned_tensors_.push_back(tensor);
    return tensor;
}

void Qwen2Model::zero_tensor(llaisysTensor_t tensor) {
    if (!tensor) {
        return;
    }
    auto size = tensor->tensor->numel() * tensor->tensor->elementSize();
    std::memset(tensor->tensor->data(), 0, size);
}

void Qwen2Model::write_cache(const tensor_t &cache, const tensor_t &src, size_t pos) {
    CHECK_ARGUMENT(cache->deviceType() == LLAISYS_DEVICE_CPU, "Qwen2: cache only supports CPU");
    CHECK_ARGUMENT(src->deviceType() == LLAISYS_DEVICE_CPU, "Qwen2: cache write only supports CPU");
    CHECK_ARGUMENT(src->isContiguous(), "Qwen2: cache src must be contiguous");

    const size_t per_token = cache->shape()[1] * cache->shape()[2];
    CHECK_ARGUMENT(src->numel() == per_token, "Qwen2: cache src shape mismatch");

    const size_t elem_size = src->elementSize();
    const size_t copy_bytes = src->numel() * elem_size;
    std::byte *dst = cache->data() + pos * per_token * elem_size;
    std::memcpy(dst, src->data(), copy_bytes);
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen2: token_ids is null");
    CHECK_ARGUMENT(ntoken > 0, "Qwen2: ntoken must be > 0");

    const float scale = 1.0f / std::sqrt(static_cast<float>(meta_.dh));
    int64_t next_token = -1;

    for (size_t t = 0; t < ntoken; ++t) {
        CHECK_ARGUMENT(cur_len_ < meta_.maxseq, "Qwen2: sequence length exceeds maxseq");

        const int64_t token = token_ids[t];
        const int64_t pos = static_cast<int64_t>(cur_len_);

        auto idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_, device_id_);
        idx->load(&token);

        auto pos_ids = Tensor::create({1}, LLAISYS_DTYPE_I64, device_, device_id_);
        pos_ids->load(&pos);

        auto x = Tensor::create({1, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::embedding(x, idx, weights_.in_embed->tensor);

        for (size_t layer = 0; layer < meta_.nlayer; ++layer) {
            auto x_norm = Tensor::create({1, meta_.hs}, meta_.dtype, device_, device_id_);
            llaisys::ops::rms_norm(x_norm, x, weights_.attn_norm_w[layer]->tensor, meta_.epsilon);

            auto q_lin = Tensor::create({1, meta_.nh * meta_.dh}, meta_.dtype, device_, device_id_);
            auto k_lin = Tensor::create({1, meta_.nkvh * meta_.dh}, meta_.dtype, device_, device_id_);
            auto v_lin = Tensor::create({1, meta_.nkvh * meta_.dh}, meta_.dtype, device_, device_id_);

            llaisys::ops::linear(q_lin, x_norm, weights_.attn_q_w[layer]->tensor, weights_.attn_q_b[layer]->tensor);
            llaisys::ops::linear(k_lin, x_norm, weights_.attn_k_w[layer]->tensor, weights_.attn_k_b[layer]->tensor);
            llaisys::ops::linear(v_lin, x_norm, weights_.attn_v_w[layer]->tensor, weights_.attn_v_b[layer]->tensor);

            auto q = q_lin->view({1, meta_.nh, meta_.dh});
            auto k = k_lin->view({1, meta_.nkvh, meta_.dh});
            auto v = v_lin->view({1, meta_.nkvh, meta_.dh});

            auto q_rope = Tensor::create({1, meta_.nh, meta_.dh}, meta_.dtype, device_, device_id_);
            auto k_rope = Tensor::create({1, meta_.nkvh, meta_.dh}, meta_.dtype, device_, device_id_);

            llaisys::ops::rope(q_rope, q, pos_ids, meta_.theta);
            llaisys::ops::rope(k_rope, k, pos_ids, meta_.theta);

            write_cache(cache_[layer].k, k_rope, cur_len_);
            write_cache(cache_[layer].v, v, cur_len_);

            auto k_cache_view = cache_[layer].k->slice(0, 0, cur_len_ + 1);
            auto v_cache_view = cache_[layer].v->slice(0, 0, cur_len_ + 1);
            auto k_cache = k_cache_view->contiguous();
            auto v_cache = v_cache_view->contiguous();

            auto attn = Tensor::create({1, meta_.nh, meta_.dh}, meta_.dtype, device_, device_id_);
            llaisys::ops::self_attention(attn, q_rope, k_cache, v_cache, scale);

            auto attn_2d = attn->view({1, meta_.hs});
            auto attn_proj = Tensor::create({1, meta_.hs}, meta_.dtype, device_, device_id_);
            llaisys::ops::linear(attn_proj, attn_2d, weights_.attn_o_w[layer]->tensor, nullptr);

            auto x_attn = Tensor::create({1, meta_.hs}, meta_.dtype, device_, device_id_);
            llaisys::ops::add(x_attn, attn_proj, x);

            auto mlp_norm = Tensor::create({1, meta_.hs}, meta_.dtype, device_, device_id_);
            llaisys::ops::rms_norm(mlp_norm, x_attn, weights_.mlp_norm_w[layer]->tensor, meta_.epsilon);

            auto gate = Tensor::create({1, meta_.di}, meta_.dtype, device_, device_id_);
            auto up = Tensor::create({1, meta_.di}, meta_.dtype, device_, device_id_);
            auto swiglu_out = Tensor::create({1, meta_.di}, meta_.dtype, device_, device_id_);
            auto down = Tensor::create({1, meta_.hs}, meta_.dtype, device_, device_id_);

            llaisys::ops::linear(gate, mlp_norm, weights_.mlp_gate_w[layer]->tensor, nullptr);
            llaisys::ops::linear(up, mlp_norm, weights_.mlp_up_w[layer]->tensor, nullptr);
            llaisys::ops::swiglu(swiglu_out, gate, up);
            llaisys::ops::linear(down, swiglu_out, weights_.mlp_down_w[layer]->tensor, nullptr);

            auto x_next = Tensor::create({1, meta_.hs}, meta_.dtype, device_, device_id_);
            llaisys::ops::add(x_next, down, x_attn);
            x = x_next;
        }

        auto final_norm = Tensor::create({1, meta_.hs}, meta_.dtype, device_, device_id_);
        llaisys::ops::rms_norm(final_norm, x, weights_.out_norm_w->tensor, meta_.epsilon);

        auto logits = Tensor::create({1, meta_.voc}, meta_.dtype, device_, device_id_);
        llaisys::ops::linear(logits, final_norm, weights_.out_embed->tensor, nullptr);

        auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_, device_id_);
        auto max_val = Tensor::create({1}, meta_.dtype, device_, device_id_);
        llaisys::ops::argmax(max_idx, max_val, logits);

        next_token = *reinterpret_cast<int64_t *>(max_idx->data());
        cur_len_ += 1;
    }

    return next_token;
}

} // namespace llaisys::models
