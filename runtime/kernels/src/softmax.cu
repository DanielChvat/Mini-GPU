#include "minigpu_kernel.h"

__global__ void softmax_fp32(float *input, float *out, int rows, int cols) {
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        int base = row * cols;
        float sum = 0.0f;
        for (int col = 0; col < cols; col += 1) {
            float e = expf(input[base + col]);
            out[base + col] = e;
            sum += e;
        }

        float inv_sum = rcpf(sum);
        for (int col = 0; col < cols; col += 1) {
            out[base + col] = out[base + col] * inv_sum;
        }
    }
}

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

MINIGPU_REGISTER_KERNEL("softmax.fp32", "softmax", "fp32", "softmax_fp32")
MINIGPU_REGISTER_KERNEL("softmax_backward.fp32", "softmax_backward", "fp32", "softmax_backward_fp32")
