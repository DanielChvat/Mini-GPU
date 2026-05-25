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

__global__ void linear_backward_input_fp32(float *grad_out,
                                           float *weight,
                                           float *grad_input,
                                           int batch,
                                           int out_features,
                                           int in_features) {
    for (int idx = threadIdx.x; idx < batch * in_features; idx += blockDim.x) {
        int row = idx / in_features;
        int in_col = idx - row * in_features;
        float sum = 0.0f;
        for (int out_col = 0; out_col < out_features; out_col += 1) {
            sum += grad_out[row * out_features + out_col] *
                   weight[out_col * in_features + in_col];
        }
        grad_input[idx] = sum;
    }
}

__global__ void linear_backward_weight_fp32(float *input,
                                            float *grad_out,
                                            float *grad_weight,
                                            int batch,
                                            int out_features,
                                            int in_features) {
    for (int idx = threadIdx.x; idx < out_features * in_features; idx += blockDim.x) {
        int out_col = idx / in_features;
        int in_col = idx - out_col * in_features;
        float sum = 0.0f;
        for (int row = 0; row < batch; row += 1) {
            sum += grad_out[row * out_features + out_col] *
                   input[row * in_features + in_col];
        }
        grad_weight[idx] = sum;
    }
}

__global__ void linear_backward_bias_fp32(float *grad_out,
                                          float *grad_bias,
                                          int batch,
                                          int out_features) {
    for (int out_col = threadIdx.x; out_col < out_features; out_col += blockDim.x) {
        float sum = 0.0f;
        for (int row = 0; row < batch; row += 1) {
            sum += grad_out[row * out_features + out_col];
        }
        grad_bias[out_col] = sum;
    }
}

MINIGPU_REGISTER_TYPED_KERNELS("linear", "i32,fp32", "linear_kernel")
MINIGPU_REGISTER_TYPED_KERNELS("linear_bias", "i32,fp32", "linear_bias_kernel")
MINIGPU_REGISTER_KERNEL("linear_backward_input.fp32", "linear_backward_input", "fp32", "linear_backward_input_fp32")
MINIGPU_REGISTER_KERNEL("linear_backward_weight.fp32", "linear_backward_weight", "fp32", "linear_backward_weight_fp32")
MINIGPU_REGISTER_KERNEL("linear_backward_bias.fp32", "linear_backward_bias", "fp32", "linear_backward_bias_fp32")
