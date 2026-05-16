#include "minigpu_linalg.h"

__global__ void scal_fp32(float *x, float *out, float alpha, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = minigpu_scal_value(alpha, x[i]);
    }
}

__global__ void axpy_fp32(float *x, float *y, float *out, float alpha, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = minigpu_axpy_value(alpha, x[i], y[i]);
    }
}

template <typename T>
__global__ void dot_kernel(T *a, T *b, T *out, int n) {
    T sum = 0;
    for (int i = 0; i < n; i += 1) {
        sum += a[i] * b[i];
    }
    out[0] = sum;
}

template <typename T>
__global__ void gemv_kernel(T *a, T *x, T *out, int m, int n) {
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        out[row] = minigpu_gemv_element(a, x, row, n);
    }
}

__global__ void addmm_fp32(float *self,
                           float *a,
                           float *b,
                           float *out,
                           float beta,
                           float alpha,
                           int m,
                           int n,
                           int k) {
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int row = idx / n;
        int col = idx - row * n;
        out[idx] = minigpu_addmm_element(self, a, b, beta, alpha, row, col, n, k);
    }
}

MINIGPU_REGISTER_KERNEL("scal.fp32", "scal", "fp32", "scal_fp32")
MINIGPU_REGISTER_KERNEL("axpy.fp32", "axpy", "fp32", "axpy_fp32")
MINIGPU_REGISTER_TYPED_KERNELS("dot", "i32,fp32", "dot_kernel")
MINIGPU_REGISTER_TYPED_KERNELS("gemv", "i32,fp32", "gemv_kernel")
MINIGPU_REGISTER_KERNEL("addmm.fp32", "addmm", "fp32", "addmm_fp32")
