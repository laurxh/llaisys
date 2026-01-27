#include "tensor.hpp"

#include "../utils.hpp"
#include "../ops/rearrange/op.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    ptrdiff_t stride = 1;
    size_t ndim_ = _meta.shape.size();
    for (size_t i = 1; i <= ndim_; ++i) {
        if (stride != _meta.strides[ndim_ - i]) {
            return false;
        }
        stride *= static_cast<ptrdiff_t>(_meta.shape[ndim_ - i]);
    }

    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    TensorMeta new_meta = _meta;
    size_t ndim = order.size();
    for (size_t i = 1; i <= ndim; i++) {
        new_meta.shape[i - 1] = _meta.shape[order[i - 1]];
        new_meta.strides[i - 1] = _meta.strides[order[i - 1]];
    }
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    CHECK_ARGUMENT(isContiguous(), "view on non-contiguous tensor is not supported");
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape = shape;
    size_t ndim = shape.size();
    new_meta.strides = std::vector<ptrdiff_t>(ndim);
    ptrdiff_t stride = 1;
    for (size_t i = 1; i <= ndim; i++) {
        new_meta.strides[ndim - i] = stride;
        stride *= static_cast<ptrdiff_t>(shape[ndim - i]);
    }
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    TensorMeta new_meta = _meta;
    size_t _offset = static_cast<size_t>(_meta.strides[dim]) * start * elementSize();
    new_meta.shape[dim] = end - start;
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());
    auto bytes = this->numel() * this->elementSize();
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            bytes,
            LLAISYS_MEMCPY_H2H);
    } else {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            bytes,
            LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        // Create a non-owning shared_ptr view to return "self" without copying.
        return tensor_t(const_cast<Tensor *>(this), [](Tensor *) {});
    }

    auto out = Tensor::create(this->shape(), this->dtype(), this->deviceType(), this->deviceId());

    // Use the rearrange op to materialize a contiguous copy.
    auto self = tensor_t(const_cast<Tensor *>(this), [](Tensor *) {});
    llaisys::ops::rearrange(out, self);
    return out;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    const size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    CHECK_ARGUMENT(new_numel == this->numel(), "reshape: total number of elements must match");

    // Fast path: contiguous tensors can always be viewed as the new shape.
    if (this->isContiguous()) {
        return this->view(shape);
    }

    // Try to reshape without copying by collapsing contiguous stride groups.
    const auto &old_shape = this->shape();
    const auto &old_strides = this->strides();
    const size_t old_ndim = old_shape.size();

    std::vector<size_t> group_numel_rev;
    std::vector<ptrdiff_t> group_base_stride_rev;
    group_numel_rev.reserve(old_ndim);
    group_base_stride_rev.reserve(old_ndim);

    size_t current_group_numel = old_shape.back();
    ptrdiff_t current_group_base_stride = old_strides.back();
    for (size_t i = old_ndim - 1; i > 0; --i) {
        const size_t prev = i - 1;
        const bool same_group = old_strides[prev] == old_strides[i] * static_cast<ptrdiff_t>(old_shape[i]);
        if (same_group) {
            current_group_numel *= old_shape[prev];
            current_group_base_stride = old_strides[i];
        } else {
            group_numel_rev.push_back(current_group_numel);
            group_base_stride_rev.push_back(current_group_base_stride);
            current_group_numel = old_shape[prev];
            current_group_base_stride = old_strides[prev];
        }
    }
    group_numel_rev.push_back(current_group_numel);
    group_base_stride_rev.push_back(current_group_base_stride);

    std::vector<size_t> group_numel(group_numel_rev.rbegin(), group_numel_rev.rend());
    std::vector<ptrdiff_t> group_base_stride(group_base_stride_rev.rbegin(), group_base_stride_rev.rend());

    std::vector<ptrdiff_t> new_strides(shape.size(), 0);
    size_t group_idx = 0;
    size_t group_remaining = group_numel.empty() ? 1 : group_numel[0];
    ptrdiff_t base_stride = group_base_stride.empty() ? 1 : group_base_stride[0];

    bool viewable = !shape.empty();
    for (size_t i = 0; i < shape.size() && viewable; ++i) {
        const size_t dim = shape[i];
        if (dim == 0 || group_idx >= group_numel.size() || group_remaining % dim != 0) {
            viewable = false;
            break;
        }
        new_strides[i] = base_stride * static_cast<ptrdiff_t>(group_remaining / dim);
        group_remaining /= dim;
        if (group_remaining == 1) {
            group_idx += 1;
            if (group_idx < group_numel.size()) {
                group_remaining = group_numel[group_idx];
                base_stride = group_base_stride[group_idx];
            }
        }
    }
    viewable = viewable && group_idx == group_numel.size() && group_remaining == 1;

    if (viewable) {
        TensorMeta new_meta{this->dtype(), shape, new_strides};
        return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
    }

    // Fallback: materialize a contiguous copy, then view.
    auto base = this->contiguous();
    return base->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
