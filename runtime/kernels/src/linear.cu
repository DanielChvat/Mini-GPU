#include "minigpu_linalg.h"

template <typename T>
__global__ void linear_kernel(T *input, T *weight, T *out,
                              int total, int out_features, int in_features) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int row = idx / out_features;
        int col = idx - row * out_features;
        out[idx] = minigpu_linear_element(
            input, weight, row, col, out_features, in_features);
    }
}

template <typename T>
__global__ void linear_bias_kernel(T *input, T *weight, T *bias, T *out,
                                   int total, int out_features, int in_features) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int row = idx / out_features;
        int col = idx - row * out_features;
        out[idx] = bias[col] + minigpu_linear_element(
            input, weight, row, col, out_features, in_features);
    }
}

MINIGPU_REGISTER_TYPED_KERNELS("linear", "i32,fp32", "linear_kernel")
MINIGPU_REGISTER_TYPED_KERNELS("linear_bias", "i32,fp32", "linear_bias_kernel")
