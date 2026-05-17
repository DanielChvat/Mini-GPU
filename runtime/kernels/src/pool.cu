#include "minigpu_kernel.h"

__global__ void max_pool2d_fp32(float *input,
                                float *out,
                                int total,
                                int channels,
                                int input_height,
                                int input_width,
                                int output_height,
                                int output_width,
                                int stride_h,
                                int stride_w) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int tmp = idx / output_width;
        int ox = idx - tmp * output_width;
        int tmp2 = tmp / output_height;
        int oy = tmp - tmp2 * output_height;
        int n = tmp2 / channels;
        int c = tmp2 - n * channels;

        int base = n * channels * input_height * input_width +
                   c * input_height * input_width;
        int iy0 = oy * stride_h;
        int ix0 = ox * stride_w;
        float best = input[base + iy0 * input_width + ix0];

        float v1 = input[base + iy0 * input_width + ix0 + 1];
        float v2 = input[base + (iy0 + 1) * input_width + ix0];
        float v3 = input[base + (iy0 + 1) * input_width + ix0 + 1];
        if (v1 > best) {
            best = v1;
        }
        if (v2 > best) {
            best = v2;
        }
        if (v3 > best) {
            best = v3;
        }
        out[idx] = best;
    }
}

__global__ void avg_pool2d_fp32(float *input,
                                float *out,
                                int total,
                                int channels,
                                int input_height,
                                int input_width,
                                int output_height,
                                int output_width,
                                int stride_h,
                                int stride_w) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int tmp = idx / output_width;
        int ox = idx - tmp * output_width;
        int tmp2 = tmp / output_height;
        int oy = tmp - tmp2 * output_height;
        int n = tmp2 / channels;
        int c = tmp2 - n * channels;

        int base = n * channels * input_height * input_width +
                   c * input_height * input_width;
        int iy0 = oy * stride_h;
        int ix0 = ox * stride_w;
        float sum = input[base + iy0 * input_width + ix0];

        sum += input[base + iy0 * input_width + ix0 + 1];
        sum += input[base + (iy0 + 1) * input_width + ix0];
        sum += input[base + (iy0 + 1) * input_width + ix0 + 1];
        out[idx] = sum * 0.25f;
    }
}

__global__ void adaptive_avg_pool2d_fp32(float *input,
                                         float *out,
                                         int total,
                                         int channels,
                                         int input_height,
                                         int input_width,
                                         int output_height,
                                         int output_width) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int tmp = idx / output_width;
        int ox = idx - tmp * output_width;
        int tmp2 = tmp / output_height;
        int oy = tmp - tmp2 * output_height;
        int n = tmp2 / channels;
        int c = tmp2 - n * channels;

        int iy0 = oy * 2;
        int ix0 = ox * 2;
        int base = n * channels * input_height * input_width +
                   c * input_height * input_width;
        float sum = input[base + iy0 * input_width + ix0];
        sum += input[base + iy0 * input_width + ix0 + 1];
        sum += input[base + (iy0 + 1) * input_width + ix0];
        sum += input[base + (iy0 + 1) * input_width + ix0 + 1];
        out[idx] = sum * 0.25f;
    }
}

MINIGPU_REGISTER_KERNEL("max_pool2d.fp32", "max_pool2d", "fp32", "max_pool2d_fp32")
MINIGPU_REGISTER_KERNEL("avg_pool2d.fp32", "avg_pool2d", "fp32", "avg_pool2d_fp32")
MINIGPU_REGISTER_KERNEL("adaptive_avg_pool2d.fp32", "adaptive_avg_pool2d", "fp32", "adaptive_avg_pool2d_fp32")
