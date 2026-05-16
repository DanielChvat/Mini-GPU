#include "minigpu_kernel.h"

__global__ void log_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = logf(a[i]);
    }
}

__global__ void log2_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = log2f(a[i]);
    }
}

__global__ void log10_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = log10f(a[i]);
    }
}

__global__ void sin_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = sinf(a[i]);
    }
}

__global__ void cos_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = cosf(a[i]);
    }
}

__global__ void tan_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = tanf(a[i]);
    }
}

__global__ void pow_fp32(float *a, float *b, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = powf(a[i], b[i]);
    }
}

MINIGPU_REGISTER_KERNEL("log.fp32", "log", "fp32", "log_fp32")
MINIGPU_REGISTER_KERNEL("log2.fp32", "log2", "fp32", "log2_fp32")
MINIGPU_REGISTER_KERNEL("log10.fp32", "log10", "fp32", "log10_fp32")
MINIGPU_REGISTER_KERNEL("sin.fp32", "sin", "fp32", "sin_fp32")
MINIGPU_REGISTER_KERNEL("cos.fp32", "cos", "fp32", "cos_fp32")
MINIGPU_REGISTER_KERNEL("tan.fp32", "tan", "fp32", "tan_fp32")
MINIGPU_REGISTER_KERNEL("pow.fp32", "pow", "fp32", "pow_fp32")
