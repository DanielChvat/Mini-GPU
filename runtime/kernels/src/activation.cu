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

__global__ void relu_backward_fp32(float *grad_out, float *self, float *out, float threshold, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = grad_out[i];
        if (self[i] <= threshold) {
            v = 0.0f;
        }
        out[i] = v;
    }
}

__global__ void sigmoid_backward_fp32(float *grad_out, float *output, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float y = output[i];
        out[i] = grad_out[i] * y * (1.0f - y);
    }
}

__global__ void tanh_backward_fp32(float *grad_out, float *output, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float y = output[i];
        out[i] = grad_out[i] * (1.0f - y * y);
    }
}

MINIGPU_REGISTER_TYPED_KERNELS("relu", "i32,i16,i8,fp32,fp16,fp8_e4m3fn", "relu_kernel")
MINIGPU_REGISTER_KERNEL("exp.fp32", "exp", "fp32", "exp_fp32")
MINIGPU_REGISTER_KERNEL("sigmoid.fp32", "sigmoid", "fp32", "sigmoid_fp32")
MINIGPU_REGISTER_KERNEL("tanh.fp32", "tanh", "fp32", "tanh_fp32")
MINIGPU_REGISTER_KERNEL("relu_backward.fp32", "relu_backward", "fp32", "relu_backward_fp32")
MINIGPU_REGISTER_KERNEL("sigmoid_backward.fp32", "sigmoid_backward", "fp32", "sigmoid_backward_fp32")
MINIGPU_REGISTER_KERNEL("tanh_backward.fp32", "tanh_backward", "fp32", "tanh_backward_fp32")
