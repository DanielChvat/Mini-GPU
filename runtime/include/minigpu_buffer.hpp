#ifndef MINIGPU_BUFFER_HPP
#define MINIGPU_BUFFER_HPP

#include "minigpu_runtime.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>

namespace minigpu {

template <typename T>
class Tensor {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Tensor requires trivially copyable element types");

public:
    class ElementProxy {
    public:
        ElementProxy(Tensor *owner, std::size_t index)
            : owner_(owner), index_(index) {}

        ElementProxy &operator=(const T &value) {
            owner_->write(index_, value);
            return *this;
        }

        operator T() const {
            return owner_->read(index_);
        }

    private:
        Tensor *owner_ = nullptr;
        std::size_t index_ = 0;
    };

    Tensor(Context &context, std::size_t count)
        : context_(&context),
          buffer_(context.device_malloc(count * sizeof(T))),
          count_(count) {}

    Tensor(Context &context, const std::vector<T> &values)
        : Tensor(context, values.size()) {
        write_all(values);
    }

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    DeviceAddress addr() const noexcept {
        return buffer_.addr();
    }

    std::size_t size() const noexcept {
        return count_;
    }

    std::size_t bytes() const noexcept {
        return count_ * sizeof(T);
    }

    T read(std::size_t index) const {
        check_index(index);
        T value{};
        context_->copy_from_device(&value, addr() + index * sizeof(T), sizeof(T));
        return value;
    }

    void write(std::size_t index, const T &value) {
        check_index(index);
        context_->copy_to_device(addr() + index * sizeof(T), &value, sizeof(T));
    }

    std::vector<T> read_all() const {
        std::vector<T> values(count_);
        if (!values.empty()) {
            context_->copy_from_device(values.data(), addr(), bytes());
        }
        return values;
    }

    void write_all(const std::vector<T> &values) {
        if (values.size() > count_) {
            throw Error(Status::OutOfRange);
        }
        if (!values.empty()) {
            context_->copy_to_device(addr(), values.data(), values.size() * sizeof(T));
        }
    }

    ElementProxy operator[](std::size_t index) {
        check_index(index);
        return ElementProxy(this, index);
    }

    T operator[](std::size_t index) const {
        return read(index);
    }

private:
    void check_index(std::size_t index) const {
        if (index >= count_) {
            throw Error(Status::OutOfRange);
        }
    }

    Context *context_ = nullptr;
    DeviceBuffer buffer_;
    std::size_t count_ = 0;
};

} // namespace minigpu

#endif
