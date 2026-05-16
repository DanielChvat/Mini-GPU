#include "minigpu_kernel.h"

__global__ void reciprocal_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = rcpf(a[i]);
    }
}

__global__ void div_fp32(float *a, float *b, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = a[i] * rcpf(b[i]);
    }
}

MINIGPU_REGISTER_KERNEL("reciprocal.fp32", "reciprocal", "fp32", "reciprocal_fp32")
MINIGPU_REGISTER_KERNEL("div.fp32", "div", "fp32", "div_fp32")
