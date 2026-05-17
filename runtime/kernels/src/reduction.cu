#include "minigpu_kernel.h"

__global__ void sum_fp32(float *a, float *out, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i += 1) {
        acc += a[i];
    }
    out[0] = acc;
}

__global__ void mean_fp32(float *a, float *out, int n, float inv_n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i += 1) {
        acc += a[i];
    }
    out[0] = acc * inv_n;
}

__global__ void amax_fp32(float *a, float *out, int n) {
    float best = a[0];
    for (int i = 1; i < n; i += 1) {
        if (minigpu_float_gt(a[i], best)) {
            best = a[i];
        }
    }
    out[0] = best;
}

__global__ void amin_fp32(float *a, float *out, int n) {
    float best = a[0];
    for (int i = 1; i < n; i += 1) {
        if (minigpu_float_lt(a[i], best)) {
            best = a[i];
        }
    }
    out[0] = best;
}

__global__ void argmax_fp32(float *a, int *out, int n) {
    int best_index = 0;
    float best = a[0];
    for (int i = 1; i < n; i += 1) {
        if (minigpu_float_gt(a[i], best)) {
            best = a[i];
            best_index = i;
        }
    }
    out[0] = best_index;
}

__global__ void argmin_fp32(float *a, int *out, int n) {
    int best_index = 0;
    float best = a[0];
    for (int i = 1; i < n; i += 1) {
        if (minigpu_float_lt(a[i], best)) {
            best = a[i];
            best_index = i;
        }
    }
    out[0] = best_index;
}

MINIGPU_REGISTER_KERNEL("sum.fp32", "sum", "fp32", "sum_fp32")
MINIGPU_REGISTER_KERNEL("mean.fp32", "mean", "fp32", "mean_fp32")
MINIGPU_REGISTER_KERNEL("amax.fp32", "amax", "fp32", "amax_fp32")
MINIGPU_REGISTER_KERNEL("amin.fp32", "amin", "fp32", "amin_fp32")
MINIGPU_REGISTER_KERNEL("argmax.fp32", "argmax", "fp32", "argmax_fp32")
MINIGPU_REGISTER_KERNEL("argmin.fp32", "argmin", "fp32", "argmin_fp32")
