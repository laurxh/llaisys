#include "llaisys/models/qwen2.h"

#include "../models/qwen2/qwen2_model.hpp"

__C {
struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model *impl;
};

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                  llaisysDeviceType_t device,
                                                  int *device_ids,
                                                  int ndevice) {
    if (!meta) {
        return nullptr;
    }
    int device_id = 0;
    if (device_ids && ndevice > 0) {
        device_id = device_ids[0];
    }
    auto *model = new LlaisysQwen2Model{new llaisys::models::Qwen2Model(*meta, device, device_id)};
    return model;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    delete model->impl;
    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model) {
        return nullptr;
    }
    return model->impl->weights();
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (!model) {
        return -1;
    }
    return model->impl->infer(token_ids, ntoken);
}
}
