#include "minigpu_kernel.h"

__global__ void softmax_backward_fp32(float *grad_out,
                                      float *output,
                                      float *out,
                                      int rows,
                                      int cols) {
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        int base = row * cols;
        float dot = 0.0f;
        for (int col = 0; col < cols; col += 1) {
            dot += grad_out[base + col] * output[base + col];
        }
        for (int col = 0; col < cols; col += 1) {
            out[base + col] = output[base + col] * (grad_out[base + col] - dot);
        }
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

MINIGPU_REGISTER_KERNEL("softmax_backward.fp32", "softmax_backward", "fp32", "softmax_backward_fp32")
MINIGPU_REGISTER_KERNEL("linear_backward_input.fp32", "linear_backward_input", "fp32", "linear_backward_input_fp32")
MINIGPU_REGISTER_KERNEL("linear_backward_weight.fp32", "linear_backward_weight", "fp32", "linear_backward_weight_fp32")
MINIGPU_REGISTER_KERNEL("linear_backward_bias.fp32", "linear_backward_bias", "fp32", "linear_backward_bias_fp32")
