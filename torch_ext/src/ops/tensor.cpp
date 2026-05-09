#include "minigpu_torch.hpp"

#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/DimVector.h>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace minigpu::torch_backend {

namespace {

constexpr auto kMiniGpuDeviceType = c10::DeviceType::PrivateUse1;

/* Device allocation record stored in each PrivateUse1 DataPtr context. */
struct MiniGpuStorage {
    minigpu::DeviceAddress addr = 0;
    std::size_t size = 0;
};

/* Opaque pointer value used by PyTorch to represent a Mini-GPU address. */
class MiniGpuDataPtr {
public:
    explicit MiniGpuDataPtr(minigpu::DeviceAddress addr) : addr_(addr) {}

    void *opaque() const {
        return reinterpret_cast<void *>(
            static_cast<std::uintptr_t>(addr_) + static_cast<std::uintptr_t>(1));
    }

    static minigpu::DeviceAddress decode(const void *ptr) {
        const auto encoded = reinterpret_cast<std::uintptr_t>(ptr);
        if (encoded == 0) {
            throw std::runtime_error("Mini-GPU tensor has a null data pointer");
        }
        return static_cast<minigpu::DeviceAddress>(encoded - 1u);
    }

private:
    minigpu::DeviceAddress addr_ = 0;
};

/* Free the runtime allocation owned by a Mini-GPU tensor. */
void delete_minigpu_allocation(void *ctx) {
    auto *storage = static_cast<MiniGpuStorage *>(ctx);
    if (!storage) {
        return;
    }

    try {
        runtime_context().device_free(storage->addr);
    } catch (...) {
        /* PyTorch deleters cannot report failures safely during unwinding. */
    }
    delete storage;
}

/* Minimal allocator backing PrivateUse1 tensor storage with Mini-GPU memory. */
class MiniGpuAllocator final : public c10::Allocator {
public:
    c10::DataPtr allocate(std::size_t n) override {
        if (n == 0) {
            return c10::DataPtr(nullptr, c10::Device(kMiniGpuDeviceType, 0));
        }

        auto buffer = runtime_context().device_malloc(n);
        auto *storage = new MiniGpuStorage{buffer.addr(), n};
        MiniGpuDataPtr ptr(storage->addr);
        buffer.release();

        return c10::DataPtr(
            ptr.opaque(),
            storage,
            &delete_minigpu_allocation,
            c10::Device(kMiniGpuDeviceType, 0));
    }

    void copy_data(void *dest, const void *src, std::size_t count) const override {
        if (count == 0) {
            return;
        }
        runtime_context().copy_from_device(dest, MiniGpuDataPtr::decode(src), count);
    }
};

MiniGpuAllocator g_minigpu_allocator;
c10::AllocatorRegisterer<c10::DeviceType::PrivateUse1>
    g_minigpu_allocator_registerer(&g_minigpu_allocator);
C10_REGISTER_GUARD_IMPL(
    PrivateUse1,
    c10::impl::NoOpDeviceGuardImpl<c10::DeviceType::PrivateUse1>)

/* Return true when a tensor lives on the Mini-GPU PrivateUse1 device. */
bool is_minigpu_tensor(const at::Tensor &tensor) {
    return tensor.device().type() == kMiniGpuDeviceType;
}

/* Return the total byte count for a tensor. */
std::size_t tensor_nbytes(const at::Tensor &tensor) {
    return static_cast<std::size_t>(tensor.numel()) *
           static_cast<std::size_t>(tensor.element_size());
}

/* Ensure the first copy path only handles simple dense tensors. */
void require_contiguous(const at::Tensor &tensor, const char *name) {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error(std::string(name) +
                                 " must be contiguous for Mini-GPU copy_");
    }
}

/* Convert a symbolic shape to concrete integers for current Mini-GPU tensors. */
std::vector<std::int64_t> concrete_sizes(c10::SymIntArrayRef size) {
    std::vector<std::int64_t> out;
    out.reserve(size.size());
    for (const auto &dim : size) {
        out.push_back(dim.expect_int());
    }
    return out;
}

/* Compute contiguous strides for a concrete shape. */
c10::SymDimVector contiguous_strides(c10::SymIntArrayRef size) {
    c10::SymDimVector strides(size.size());
    c10::SymInt next_stride = 1;
    for (std::int64_t dim = static_cast<std::int64_t>(size.size()) - 1; dim >= 0;
         --dim) {
        strides[static_cast<std::size_t>(dim)] = next_stride;
        next_stride = next_stride * size[static_cast<std::size_t>(dim)];
    }
    return strides;
}

/* Infer the single -1 dimension accepted by Tensor.view. */
c10::SymDimVector infer_view_size(const at::Tensor &self, c10::SymIntArrayRef size) {
    c10::SymDimVector inferred;
    inferred.reserve(size.size());

    const auto self_numel = self.numel();
    std::int64_t known_product = 1;
    int inferred_dim = -1;

    for (std::size_t i = 0; i < size.size(); ++i) {
        const auto dim = size[i].expect_int();
        if (dim == -1) {
            if (inferred_dim >= 0) {
                throw std::runtime_error("Mini-GPU view only allows one inferred dim");
            }
            inferred_dim = static_cast<int>(i);
            inferred.emplace_back(1);
        } else {
            if (dim < 0) {
                throw std::runtime_error("Mini-GPU view received a negative dim");
            }
            known_product *= dim;
            inferred.emplace_back(dim);
        }
    }

    if (inferred_dim >= 0) {
        if (known_product == 0 || self_numel % known_product != 0) {
            throw std::runtime_error("Mini-GPU view cannot infer requested shape");
        }
        inferred[static_cast<std::size_t>(inferred_dim)] = self_numel / known_product;
    } else if (known_product != self_numel) {
        throw std::runtime_error("Mini-GPU view changes tensor element count");
    }

    return inferred;
}

/* Build a tensor alias that shares storage with another Mini-GPU tensor. */
at::Tensor make_alias(
    const at::Tensor &self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::SymInt storage_offset) {
    auto storage = self.storage();
    auto key_set = self.key_set().remove(c10::DispatchKey::Conjugate);
    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        c10::TensorImpl::VIEW,
        std::move(storage),
        key_set,
        c10::scalarTypeToTypeMeta(self.scalar_type()));

    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride, storage_offset);
    return tensor;
}

} // namespace

at::Tensor empty(
    at::IntArrayRef size,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
    (void)device;

    if (layout.has_value() && *layout != c10::Layout::Strided) {
        throw std::runtime_error("Mini-GPU only supports strided tensors");
    }
    if (pin_memory.value_or(false)) {
        throw std::runtime_error("Mini-GPU does not support pinned memory");
    }

    const auto scalar_type = dtype.value_or(at::kFloat);
    auto tensor_base = at::detail::empty_generic(
        size,
        &g_minigpu_allocator,
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        scalar_type,
        memory_format);

    return at::Tensor(std::move(tensor_base));
}

at::Tensor empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory) {
    (void)device;

    if (layout.has_value() && *layout != c10::Layout::Strided) {
        throw std::runtime_error("Mini-GPU only supports strided tensors");
    }
    if (pin_memory.value_or(false)) {
        throw std::runtime_error("Mini-GPU does not support pinned memory");
    }

    const auto scalar_type = dtype.value_or(at::kFloat);
    auto tensor_base = at::detail::empty_strided_generic(
        size,
        stride,
        &g_minigpu_allocator,
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        scalar_type);

    return at::Tensor(std::move(tensor_base));
}

at::Tensor &copy_(at::Tensor &self, const at::Tensor &src, bool non_blocking) {
    (void)non_blocking;

    const bool dst_is_minigpu = is_minigpu_tensor(self);
    const bool src_is_minigpu = is_minigpu_tensor(src);
    if (!dst_is_minigpu && !src_is_minigpu) {
        throw std::runtime_error("Mini-GPU copy_ called without a Mini-GPU tensor");
    }
    if (self.sizes() != src.sizes()) {
        throw std::runtime_error("Mini-GPU copy_ requires matching tensor shapes");
    }
    if (self.scalar_type() != src.scalar_type()) {
        throw std::runtime_error("Mini-GPU copy_ does not support dtype conversion yet");
    }

    const auto nbytes = tensor_nbytes(self);
    if (nbytes == 0) {
        return self;
    }

    if (dst_is_minigpu && !src_is_minigpu) {
        require_contiguous(self, "destination");
        auto cpu_src = src.contiguous();
        runtime_context().copy_to_device(
            MiniGpuDataPtr::decode(self.data_ptr()), cpu_src.data_ptr(), nbytes);
        return self;
    }

    if (!dst_is_minigpu && src_is_minigpu) {
        require_contiguous(src, "source");
        if (self.is_contiguous()) {
            runtime_context().copy_from_device(
                self.data_ptr(), MiniGpuDataPtr::decode(src.data_ptr()), nbytes);
            return self;
        }

        auto tmp = at::empty_like(self, self.options().device(c10::DeviceType::CPU));
        runtime_context().copy_from_device(
            tmp.data_ptr(), MiniGpuDataPtr::decode(src.data_ptr()), nbytes);
        self.copy_(tmp);
        return self;
    }

    require_contiguous(self, "destination");
    require_contiguous(src, "source");
    std::vector<std::uint8_t> tmp(nbytes);
    runtime_context().copy_from_device(
        tmp.data(), MiniGpuDataPtr::decode(src.data_ptr()), nbytes);
    runtime_context().copy_to_device(
        MiniGpuDataPtr::decode(self.data_ptr()), tmp.data(), tmp.size());
    return self;
}

at::Tensor view(const at::Tensor &self, c10::SymIntArrayRef size) {
    if (!self.is_contiguous()) {
        throw std::runtime_error("Mini-GPU view currently requires contiguous input");
    }

    auto inferred_size = infer_view_size(self, size);
    auto stride = contiguous_strides(inferred_size);
    return make_alias(self, inferred_size, stride, self.sym_storage_offset());
}

at::Tensor as_strided(
    const at::Tensor &self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    ::std::optional<c10::SymInt> storage_offset) {
    return make_alias(
        self,
        size,
        stride,
        storage_offset.value_or(self.sym_storage_offset()));
}

at::Tensor reshape_alias(
    const at::Tensor &self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
    return make_alias(self, size, stride, self.sym_storage_offset());
}

} // namespace minigpu::torch_backend
