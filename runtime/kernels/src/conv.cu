#include "minigpu_linalg.h"

template <typename T>
__global__ void conv1d_kernel(T *input,
                              T *weight,
                              T *out,
                              int total,
                              int batch_size,
                              int in_channels,
                              int out_channels,
                              int input_width,
                              int output_width,
                              int kernel_width,
                              int stride,
    int padding) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int tmp = idx / output_width;
        int ow = idx - tmp * output_width;

        int tmp_n = tmp / out_channels;
        int oc = tmp - tmp_n * out_channels;
        int n = tmp_n;

        out[idx] = minigpu_conv1d_element(
            input, weight, (T)0, n, oc, ow, in_channels, input_width,
            kernel_width, stride, padding);
    }
}

template <typename T>
__global__ void conv1d_bias_kernel(T *input,
                                   T *weight,
                                   T *bias,
                                   T *out,
                                   int total,
                                   int batch_size,
                                   int in_channels,
                                   int out_channels,
                                   int input_width,
                                   int output_width,
                                   int kernel_width,
                                   int stride,
    int padding) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int tmp = idx / output_width;
        int ow = idx - tmp * output_width;

        int tmp_n = tmp / out_channels;
        int oc = tmp - tmp_n * out_channels;
        int n = tmp_n;

        out[idx] = minigpu_conv1d_element(
            input, weight, bias[oc], n, oc, ow, in_channels, input_width,
            kernel_width, stride, padding);
    }
}

template <typename T>
__global__ void conv2d_kernel(T *input,
                              T *weight,
                              T *out,
                              int total,
                              int batch_size,
                              int in_channels,
                              int out_channels,
                              int input_height,
                              int input_width,
                              int output_height,
                              int output_width,
                              int kernel_height,
                              int kernel_width,
                              int stride_h,
                              int stride_w,
                              int padding_h,
    int padding_w) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int tmp0 = idx / output_width;
        int ox = idx - tmp0 * output_width;

        int tmp1 = tmp0 / output_height;
        int oy = tmp0 - tmp1 * output_height;

        int n = tmp1 / out_channels;
        int oc = tmp1 - n * out_channels;

        out[idx] = minigpu_conv2d_element(
            input, weight, (T)0, n, oc, oy, ox, in_channels, input_height,
            input_width, kernel_height, kernel_width, stride_h, stride_w,
            padding_h, padding_w);
    }
}

template <typename T>
__global__ void conv2d_bias_kernel(T *input,
                                   T *weight,
                                   T *bias,
                                   T *out,
                                   int total,
                                   int batch_size,
                                   int in_channels,
                                   int out_channels,
                                   int input_height,
                                   int input_width,
                                   int output_height,
                                   int output_width,
                                   int kernel_height,
                                   int kernel_width,
                                   int stride_h,
                                   int stride_w,
                                   int padding_h,
    int padding_w) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int tmp0 = idx / output_width;
        int ox = idx - tmp0 * output_width;

        int tmp1 = tmp0 / output_height;
        int oy = tmp0 - tmp1 * output_height;

        int n = tmp1 / out_channels;
        int oc = tmp1 - n * out_channels;

        out[idx] = minigpu_conv2d_element(
            input, weight, bias[oc], n, oc, oy, ox, in_channels, input_height,
            input_width, kernel_height, kernel_width, stride_h, stride_w,
            padding_h, padding_w);
    }
}

MINIGPU_REGISTER_KERNEL("conv1d.fp32", "conv1d", "fp32", "conv1d_kernel_fp32")
MINIGPU_REGISTER_KERNEL("conv1d_bias.fp32", "conv1d_bias", "fp32", "conv1d_bias_kernel_fp32")
MINIGPU_REGISTER_KERNEL("conv2d.fp32", "conv2d", "fp32", "conv2d_kernel_fp32")
MINIGPU_REGISTER_KERNEL("conv2d_bias.fp32", "conv2d_bias", "fp32", "conv2d_bias_kernel_fp32")
