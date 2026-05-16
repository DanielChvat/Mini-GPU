#include "minigpu_kernel.h"

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
        int ow = idx % output_width;

        int tmp = idx / output_width;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;

        T sum = 0;

        for (int ic = 0; ic < in_channels; ic += 1) {
            for (int kw = 0; kw < kernel_width; kw += 1) {
                int iw = ow * stride + kw - padding;

                if (iw >= 0 && iw < input_width) {
                    int input_idx =
                        n * in_channels * input_width +
                        ic * input_width +
                        iw;

                    int weight_idx =
                        oc * in_channels * kernel_width +
                        ic * kernel_width +
                        kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }

        out[idx] = sum;
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
        int ow = idx % output_width;

        int tmp = idx / output_width;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;

        T sum = bias[oc];

        for (int ic = 0; ic < in_channels; ic += 1) {
            for (int kw = 0; kw < kernel_width; kw += 1) {
                int iw = ow * stride + kw - padding;

                if (iw >= 0 && iw < input_width) {
                    int input_idx =
                        n * in_channels * input_width +
                        ic * input_width +
                        iw;

                    int weight_idx =
                        oc * in_channels * kernel_width +
                        ic * kernel_width +
                        kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }

        out[idx] = sum;
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
        int ox = idx % output_width;

        int tmp0 = idx / output_width;
        int oy = tmp0 % output_height;

        int tmp1 = tmp0 / output_height;
        int oc = tmp1 % out_channels;

        int n = tmp1 / out_channels;

        T sum = 0;

        for (int ic = 0; ic < in_channels; ic += 1) {
            for (int ky = 0; ky < kernel_height; ky += 1) {
                for (int kx = 0; kx < kernel_width; kx += 1) {
                    int iy = oy * stride_h + ky - padding_h;
                    int ix = ox * stride_w + kx - padding_w;

                    if (iy >= 0 && iy < input_height &&
                        ix >= 0 && ix < input_width) {
                        int input_idx =
                            n * in_channels * input_height * input_width +
                            ic * input_height * input_width +
                            iy * input_width +
                            ix;

                        int weight_idx =
                            oc * in_channels * kernel_height * kernel_width +
                            ic * kernel_height * kernel_width +
                            ky * kernel_width +
                            kx;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        out[idx] = sum;
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
        int ox = idx % output_width;

        int tmp0 = idx / output_width;
        int oy = tmp0 % output_height;

        int tmp1 = tmp0 / output_height;
        int oc = tmp1 % out_channels;

        int n = tmp1 / out_channels;

        T sum = bias[oc];

        for (int ic = 0; ic < in_channels; ic += 1) {
            for (int ky = 0; ky < kernel_height; ky += 1) {
                for (int kx = 0; kx < kernel_width; kx += 1) {
                    int iy = oy * stride_h + ky - padding_h;
                    int ix = ox * stride_w + kx - padding_w;

                    if (iy >= 0 && iy < input_height &&
                        ix >= 0 && ix < input_width) {
                        int input_idx =
                            n * in_channels * input_height * input_width +
                            ic * input_height * input_width +
                            iy * input_width +
                            ix;

                        int weight_idx =
                            oc * in_channels * kernel_height * kernel_width +
                            ic * kernel_height * kernel_width +
                            ky * kernel_width +
                            kx;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        out[idx] = sum;
    }
}