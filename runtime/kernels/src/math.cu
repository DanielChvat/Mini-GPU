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
