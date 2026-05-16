#ifndef MINIGPU_LINALG_H
#define MINIGPU_LINALG_H

#include "minigpu_kernel.h"

template <typename T>
MINIGPU_INLINE T minigpu_scal_value(T alpha, T x) {
    return alpha * x;
}

template <typename T>
MINIGPU_INLINE T minigpu_axpy_value(T alpha, T x, T y) {
    return alpha * x + y;
}

template <typename T>
MINIGPU_INLINE T minigpu_dot_strided(const T *a,
                                     int a_offset,
                                     int a_stride,
                                     const T *b,
                                     int b_offset,
                                     int b_stride,
                                     int n) {
    T sum = 0;
    for (int i = 0; i < n; i += 1) {
        sum += a[a_offset + i * a_stride] * b[b_offset + i * b_stride];
    }
    return sum;
}

template <typename T>
MINIGPU_INLINE T minigpu_gemv_element(const T *a,
                                      const T *x,
                                      int row,
                                      int n) {
    return minigpu_dot_strided(a, row * n, 1, x, 0, 1, n);
}

template <typename T>
MINIGPU_INLINE T minigpu_gemm_element(const T *a,
                                      const T *b,
                                      int row,
                                      int col,
                                      int n,
                                      int k) {
    return minigpu_dot_strided(a, row * k, 1, b, col, n, k);
}

template <typename T>
MINIGPU_INLINE T minigpu_addmm_element(const T *self,
                                       const T *a,
                                       const T *b,
                                       T beta,
                                       T alpha,
                                       int row,
                                       int col,
                                       int n,
                                       int k) {
    return beta * self[row * n + col] +
           alpha * minigpu_gemm_element(a, b, row, col, n, k);
}

template <typename T>
MINIGPU_INLINE T minigpu_linear_element(const T *input,
                                        const T *weight,
                                        int row,
                                        int col,
                                        int out_features,
                                        int in_features) {
    return minigpu_dot_strided(
        input, row * in_features, 1,
        weight, col * in_features, 1,
        in_features);
}

template <typename T>
MINIGPU_INLINE T minigpu_conv1d_element(const T *input,
                                        const T *weight,
                                        T bias,
                                        int n,
                                        int oc,
                                        int ow,
                                        int in_channels,
                                        int input_width,
                                        int kernel_width,
                                        int stride,
                                        int padding) {
    T sum = bias;
    for (int ic = 0; ic < in_channels; ic += 1) {
        for (int kw = 0; kw < kernel_width; kw += 1) {
            int iw = ow * stride + kw - padding;
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
    return sum;
}

template <typename T>
MINIGPU_INLINE T minigpu_conv2d_element(const T *input,
                                        const T *weight,
                                        T bias,
                                        int n,
                                        int oc,
                                        int oy,
                                        int ox,
                                        int in_channels,
                                        int input_height,
                                        int input_width,
                                        int kernel_height,
                                        int kernel_width,
                                        int stride_h,
                                        int stride_w,
                                        int padding_h,
                                        int padding_w) {
    T sum = bias;
    for (int ic = 0; ic < in_channels; ic += 1) {
        for (int ky = 0; ky < kernel_height; ky += 1) {
            for (int kx = 0; kx < kernel_width; kx += 1) {
                int iy = oy * stride_h + ky - padding_h;
                int ix = ox * stride_w + kx - padding_w;
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
    return sum;
}

#endif
