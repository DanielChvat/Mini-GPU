#ifndef MINIGPU_KERNELS_HPP
#define MINIGPU_KERNELS_HPP

#include "minigpu_runtime.hpp"
#include "minigpu_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace minigpu::kernels {

/* Lightweight view of one device tensor argument. */
struct TensorView {
    DeviceAddress addr = 0;
    std::size_t elements = 0;
    std::string_view dtype;
};

/* Return the folder used for precompiled Mini-GPU kernel artifacts. */
std::string default_kernel_dir();

/* Return the default kernel YAML manifest path. */
std::string default_manifest_path();

/* Register all kernels listed in a YAML manifest. */
void register_manifest(Context &context, std::string_view manifest_path);

/* Register all kernels listed in the default runtime kernel manifest. */
void register_builtin_kernels(Context &context);

/* Resolve and launch a binary elementwise kernel using tensor addresses as args. */
void launch_elementwise_binary(
    Context &context,
    std::string_view op,
    const TensorView &a,
    const TensorView &b,
    const TensorView &out,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch a unary elementwise kernel using tensor addresses as args. */
void launch_elementwise_unary(
    Context &context,
    std::string_view op,
    const TensorView &a,
    const TensorView &out,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch a matrix-multiply kernel with row-major tensor arguments. */
void launch_matmul(
    Context &context,
    const TensorView &a,
    const TensorView &b,
    const TensorView &out,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch a matrix-vector multiply kernel. */
void launch_gemv(
    Context &context,
    const TensorView &a,
    const TensorView &x,
    const TensorView &out,
    std::uint32_t m,
    std::uint32_t n,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch a dot product kernel. */
void launch_dot(
    Context &context,
    const TensorView &a,
    const TensorView &b,
    const TensorView &out,
    std::uint32_t n,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch out = alpha * x. */
void launch_scal(
    Context &context,
    const TensorView &x,
    const TensorView &out,
    float alpha,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch out = alpha * x + y. */
void launch_axpy(
    Context &context,
    const TensorView &x,
    const TensorView &y,
    const TensorView &out,
    float alpha,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch out = beta * self + alpha * (a @ b). */
void launch_addmm(
    Context &context,
    const TensorView &self,
    const TensorView &a,
    const TensorView &b,
    const TensorView &out,
    float beta,
    float alpha,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch a row-major linear layer kernel. */
void launch_linear(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &out,
    std::uint32_t total,
    std::uint32_t out_features,
    std::uint32_t in_features,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch a row-major linear layer kernel with bias */
void launch_linear_bias(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    const TensorView &out,
    std::uint32_t total,
    std::uint32_t out_features,
    std::uint32_t in_features,
    const LaunchConfig *launch_config = nullptr); 

/* Resolve and launch a 1d conv kernel without bias */
void launch_conv1d(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &out,
    std::uint32_t total, 
    std::uint32_t batch_size,
    std::uint32_t in_channels,
    std::uint32_t out_channels, 
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint32_t kernel_width,
    std::uint32_t stride,
    std::uint32_t padding,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch a 1d conv kernel with bias */
void launch_conv1d_bias(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    const TensorView &out,
    std::uint32_t total, 
    std::uint32_t batch_size,
    std::uint32_t in_channels,
    std::uint32_t out_channels, 
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint32_t kernel_width,
    std::uint32_t stride,
    std::uint32_t padding,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch a 2d conv kernel without bias */
void launch_conv2d(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &out,
    std::uint32_t total,
    std::uint32_t batch_size,
    std::uint32_t in_channels,
    std::uint32_t out_channels,
    std::uint32_t input_height,
    std::uint32_t input_width, 
    std::uint32_t output_height,
    std::uint32_t output_width,
    std::uint32_t kernel_height,
    std::uint32_t kernel_width,
    std::uint32_t stride_h,
    std::uint32_t stride_w,
    std::uint32_t padding_h,
    std::uint32_t padding_w,
    const LaunchConfig *launch_config = nullptr);

/* Resolve and launch a 2d conv kernel with bias */
void launch_conv2d_bias(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    const TensorView &out,
    std::uint32_t total,
    std::uint32_t batch_size,
    std::uint32_t in_channels,
    std::uint32_t out_channels,
    std::uint32_t input_height,
    std::uint32_t input_width, 
    std::uint32_t output_height,
    std::uint32_t output_width,
    std::uint32_t kernel_height,
    std::uint32_t kernel_width,
    std::uint32_t stride_h,
    std::uint32_t stride_w,
    std::uint32_t padding_h,
    std::uint32_t padding_w,
    const LaunchConfig *launch_config = nullptr);

/* Return the Mini-GPU kernel dtype name for a C++ scalar type. */
template <typename T>
std::string_view dtype_name();

template <>
inline std::string_view dtype_name<std::int32_t>() {
    return "i32";
}

template <>
inline std::string_view dtype_name<std::uint32_t>() {
    return "i32";
}

template <>
inline std::string_view dtype_name<std::int16_t>() {
    return "i16";
}

template <>
inline std::string_view dtype_name<std::uint16_t>() {
    return "i16";
}

template <>
inline std::string_view dtype_name<std::int8_t>() {
    return "i8";
}

template <>
inline std::string_view dtype_name<std::uint8_t>() {
    return "i8";
}

template <>
inline std::string_view dtype_name<float>() {
    return "fp32";
}

/* Launch the manifest-resolved vector_add kernel for three runtime tensors. */
template <typename T>
void vector_add(Context &context, const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out) {
    launch_elementwise_binary(
        context,
        "vector_add",
        TensorView{a.addr(), a.size(), dtype_name<T>()},
        TensorView{b.addr(), b.size(), dtype_name<T>()},
        TensorView{out.addr(), out.size(), dtype_name<T>()});
}

} // namespace minigpu::kernels

#endif
