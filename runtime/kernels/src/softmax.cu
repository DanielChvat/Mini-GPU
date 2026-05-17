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

MINIGPU_REGISTER_KERNEL("softmax.fp32", "softmax", "fp32", "softmax_fp32")
