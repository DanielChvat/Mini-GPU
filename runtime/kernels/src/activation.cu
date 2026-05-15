#include "minigpu_kernel.h"

template <typename T>
__global__ void relu_kernel(T *a, T *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T v = a[i];
        if (v < 0) {
            v = 0;
        }
        out[i] = v;
    }
}

__global__ void exp_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = expf(a[i]);
    }
}

__global__ void sigmoid_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = sigmoidf(a[i]);
    }
}

__global__ void tanh_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = tanhf(a[i]);
    }
}
