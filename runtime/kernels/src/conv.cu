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

__global__ void conv1d_backward_input_fp32(float *grad_out,
                                           float *weight,
                                           float *grad_input,
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
        int iw = idx % input_width;
        int tmp = idx / input_width;
        int ic = tmp % in_channels;
        int n = tmp / in_channels;
        float sum = 0.0f;
        int reduce_total = out_channels * output_width * kernel_width;

        for (int ridx = 0; ridx < reduce_total; ridx += 1) {
            int kw = ridx % kernel_width;
            int tmp_r0 = ridx / kernel_width;
            int ow = tmp_r0 % output_width;
            int oc = tmp_r0 / output_width;
            int source_iw = ow * stride + kw - padding;
            int grad_idx =
                n * out_channels * output_width +
                oc * output_width +
                ow;
            int weight_idx =
                oc * in_channels * kernel_width +
                ic * kernel_width +
                kw;
            if (source_iw == iw) {
                sum += grad_out[grad_idx] * weight[weight_idx];
            }
        }
        grad_input[idx] = sum;
    }
}

__global__ void conv1d_backward_weight_fp32(float *input,
                                            float *grad_out,
                                            float *grad_weight,
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
        int kw = idx % kernel_width;
        int tmp = idx / kernel_width;
        int ic = tmp % in_channels;
        int oc = tmp / in_channels;
        float sum = 0.0f;
        int reduce_total = batch_size * output_width;

        for (int ridx = 0; ridx < reduce_total; ridx += 1) {
            int ow = ridx % output_width;
            int n = ridx / output_width;
            int iw = ow * stride + kw - padding;
            int input_idx =
                n * in_channels * input_width +
                ic * input_width +
                iw;
            int grad_idx =
                n * out_channels * output_width +
                oc * output_width +
                ow;
            sum += input[input_idx] * grad_out[grad_idx];
        }
        grad_weight[idx] = sum;
    }
}

__global__ void conv1d_backward_bias_fp32(float *grad_out,
                                          float *grad_bias,
                                          int batch_size,
                                          int out_channels,
                                          int output_width) {
    for (int oc = threadIdx.x; oc < out_channels; oc += blockDim.x) {
        float sum = 0.0f;
        for (int n = 0; n < batch_size; n += 1) {
            for (int ow = 0; ow < output_width; ow += 1) {
                sum += grad_out[n * out_channels * output_width +
                                oc * output_width +
                                ow];
            }
        }
        grad_bias[oc] = sum;
    }
}

__global__ void conv2d_backward_input_fp32(float *grad_out,
                                           float *weight,
                                           float *grad_input,
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
        int ix = idx % input_width;
        int tmp0 = idx / input_width;
        int iy = tmp0 % input_height;
        int tmp1 = tmp0 / input_height;
        int ic = tmp1 % in_channels;
        int n = tmp1 / in_channels;
        float sum = 0.0f;
        int reduce_total = out_channels * output_height * output_width *
                           kernel_height * kernel_width;

        for (int ridx = 0; ridx < reduce_total; ridx += 1) {
            int kx = ridx % kernel_width;
            int tmp_r0 = ridx / kernel_width;
            int ky = tmp_r0 % kernel_height;
            int tmp_r1 = tmp_r0 / kernel_height;
            int ox = tmp_r1 % output_width;
            int tmp_r2 = tmp_r1 / output_width;
            int oy = tmp_r2 % output_height;
            int oc = tmp_r2 / output_height;
            int source_iy = oy * stride_h + ky - padding_h;
            int source_ix = ox * stride_w + kx - padding_w;
            int grad_idx =
                n * out_channels * output_height * output_width +
                oc * output_height * output_width +
                oy * output_width +
                ox;
            int weight_idx =
                oc * in_channels * kernel_height * kernel_width +
                ic * kernel_height * kernel_width +
                ky * kernel_width +
                kx;
            if ((source_iy == iy) && (source_ix == ix)) {
                sum += grad_out[grad_idx] * weight[weight_idx];
            }
        }
        grad_input[idx] = sum;
    }
}

__global__ void conv2d_backward_weight_fp32(float *input,
                                            float *grad_out,
                                            float *grad_weight,
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
        int kx = idx % kernel_width;
        int tmp0 = idx / kernel_width;
        int ky = tmp0 % kernel_height;
        int tmp1 = tmp0 / kernel_height;
        int ic = tmp1 % in_channels;
        int oc = tmp1 / in_channels;
        float sum = 0.0f;
        int reduce_total = batch_size * output_height * output_width;

        for (int ridx = 0; ridx < reduce_total; ridx += 1) {
            int ox = ridx % output_width;
            int tmp_r0 = ridx / output_width;
            int oy = tmp_r0 % output_height;
            int n = tmp_r0 / output_height;
            int iy = oy * stride_h + ky - padding_h;
            int ix = ox * stride_w + kx - padding_w;
            int input_idx =
                n * in_channels * input_height * input_width +
                ic * input_height * input_width +
                iy * input_width +
                ix;
            int grad_idx =
                n * out_channels * output_height * output_width +
                oc * output_height * output_width +
                oy * output_width +
                ox;
            sum += input[input_idx] * grad_out[grad_idx];
        }
        grad_weight[idx] = sum;
    }
}

__global__ void conv2d_backward_bias_fp32(float *grad_out,
                                          float *grad_bias,
                                          int batch_size,
                                          int out_channels,
                                          int output_height,
                                          int output_width) {
    for (int oc = threadIdx.x; oc < out_channels; oc += blockDim.x) {
        float sum = 0.0f;
        for (int n = 0; n < batch_size; n += 1) {
            for (int oy = 0; oy < output_height; oy += 1) {
                for (int ox = 0; ox < output_width; ox += 1) {
                    sum += grad_out[n * out_channels * output_height * output_width +
                                    oc * output_height * output_width +
                                    oy * output_width +
                                    ox];
                }
            }
        }
        grad_bias[oc] = sum;
    }
}

MINIGPU_REGISTER_KERNEL("conv1d.fp32", "conv1d", "fp32", "conv1d_kernel_fp32")
MINIGPU_REGISTER_KERNEL("conv1d_bias.fp32", "conv1d_bias", "fp32", "conv1d_bias_kernel_fp32")
MINIGPU_REGISTER_KERNEL("conv2d.fp32", "conv2d", "fp32", "conv2d_kernel_fp32")
MINIGPU_REGISTER_KERNEL("conv2d_bias.fp32", "conv2d_bias", "fp32", "conv2d_bias_kernel_fp32")
MINIGPU_REGISTER_KERNEL("conv1d_backward_input.fp32", "conv1d_backward_input", "fp32", "conv1d_backward_input_fp32")
MINIGPU_REGISTER_KERNEL("conv1d_backward_weight.fp32", "conv1d_backward_weight", "fp32", "conv1d_backward_weight_fp32")
MINIGPU_REGISTER_KERNEL("conv1d_backward_bias.fp32", "conv1d_backward_bias", "fp32", "conv1d_backward_bias_fp32")
MINIGPU_REGISTER_KERNEL("conv2d_backward_input.fp32", "conv2d_backward_input", "fp32", "conv2d_backward_input_fp32")
MINIGPU_REGISTER_KERNEL("conv2d_backward_weight.fp32", "conv2d_backward_weight", "fp32", "conv2d_backward_weight_fp32")
MINIGPU_REGISTER_KERNEL("conv2d_backward_bias.fp32", "conv2d_backward_bias", "fp32", "conv2d_backward_bias_fp32")
