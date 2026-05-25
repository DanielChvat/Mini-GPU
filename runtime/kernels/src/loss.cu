#include "minigpu_kernel.h"

MINIGPU_INLINE float cross_entropy_rcp_positive(float x) {
    uint32_t ix = __float_as_uint(x);
    float y = __uint_as_float(0x7ef311c3u - ix);
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}

__global__ void mse_loss_fp32(float *input, float *target, float *out, int n, float inv_n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i += 1) {
        float diff = input[i] - target[i];
        acc += diff * diff;
    }
    out[0] = acc * inv_n;
}

__global__ void mse_loss_backward_fp32(
    float *grad_out,
    float *input,
    float *target,
    float *out,
    int n,
    float scale) {
    float g = grad_out[0] * scale;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = g * (input[i] - target[i]);
    }
}

__global__ void l1_loss_fp32(float *input, float *target, float *out, int n, float inv_n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i += 1) {
        acc += fabsf(input[i] - target[i]);
    }
    out[0] = acc * inv_n;
}

__global__ void l1_loss_backward_fp32(
    float *grad_out,
    float *input,
    float *target,
    float *out,
    int n,
    float scale) {
    float g = grad_out[0] * scale;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float diff = input[i] - target[i];
        float sign = 0.0f;
        if (minigpu_float_gt(diff, 0.0f)) {
            sign = 1.0f;
        }
        if (minigpu_float_lt(diff, 0.0f)) {
            sign = -1.0f;
        }
        out[i] = g * sign;
    }
}

__global__ void l2_loss_fp32(float *input, float *target, float *out, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i += 1) {
        float diff = input[i] - target[i];
        acc += diff * diff;
    }
    out[0] = 0.5f * acc;
}

__global__ void l2_loss_backward_fp32(
    float *grad_out,
    float *input,
    float *target,
    float *out,
    int n) {
    float g = grad_out[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = g * (input[i] - target[i]);
    }
}

__global__ void cross_entropy_stats_fp32(float *logits, int *target, float *stats, int rows, int cols) {
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        int base = row * cols;
        float max_value = logits[base];
        for (int col = 1; col < cols; col += 1) {
            float value = logits[base + col];
            if (value > max_value) {
                max_value = value;
            }
        }

        int label = target[row];
        float exp_sum = 0.0f;
        float target_logit = logits[base];
        for (int col = 0; col < cols; col += 1) {
            float value = logits[base + col];
            exp_sum += expf(value - max_value);
            if (col == label) {
                target_logit = value;
            }
        }

        int stats_base = row * 3;
        stats[stats_base + 0] = max_value;
        stats[stats_base + 1] = exp_sum;
        stats[stats_base + 2] = target_logit;
    }
}

__global__ void cross_entropy_finish_fp32(float *stats, float *out, int rows, float inv_rows) {
    float total = 0.0f;
    for (int row = 0; row < rows; row += 1) {
        int stats_base = row * 3;
        float log_sum = logf(stats[stats_base + 1]);
        total += log_sum + stats[stats_base + 0] - stats[stats_base + 2];
    }
    out[0] = total * inv_rows;
}

__global__ void cross_entropy_debug_finish_fp32(float *stats, float *out, int rows) {
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        int stats_base = row * 3;
        float max_value = stats[stats_base + 0];
        float exp_sum = stats[stats_base + 1];
        float target_logit = stats[stats_base + 2];
        float log_sum = logf(exp_sum);
        int out_base = row * 5;
        out[out_base + 0] = max_value;
        out[out_base + 1] = exp_sum;
        out[out_base + 2] = log_sum;
        out[out_base + 3] = target_logit;
        out[out_base + 4] = log_sum + max_value - target_logit;
    }
}

__global__ void cross_entropy_debug_fp32(float *logits, int *target, float *out, int rows, int cols) {
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        int base = row * cols;
        float max_value = logits[base];
        for (int col = 1; col < cols; col += 1) {
            float value = logits[base + col];
            if (value > max_value) {
                max_value = value;
            }
        }

        int label = target[row];
        float exp_sum = 0.0f;
        float target_logit = logits[base];
        for (int col = 0; col < cols; col += 1) {
            float value = logits[base + col];
            exp_sum += expf(value - max_value);
            if (col == label) {
                target_logit = value;
            }
        }

        float x = exp_sum;
        float log_sum = logf(x);
        int out_base = row * 5;
        out[out_base + 0] = max_value;
        out[out_base + 1] = exp_sum;
        out[out_base + 2] = log_sum;
        out[out_base + 3] = target_logit;
        out[out_base + 4] = log_sum + max_value - target_logit;
    }
}

__global__ void cross_entropy_backward_fp32(
    float *grad_out,
    float *logits,
    int *target,
    float *out,
    int rows,
    int cols,
    float inv_rows) {
    float g = grad_out[0] * inv_rows;
    int n = rows * cols;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int row = i / cols;
        int col = i - row * cols;
        int base = row * cols;

        float max_value = logits[base];
        for (int scan = 1; scan < cols; scan += 1) {
            float value = logits[base + scan];
            if (value > max_value) {
                max_value = value;
            }
        }

        float exp_sum = 0.0f;
        for (int scan = 0; scan < cols; scan += 1) {
            exp_sum += expf(logits[base + scan] - max_value);
        }

        float prob = expf(logits[i] - max_value) * cross_entropy_rcp_positive(exp_sum);
        if (col == target[row]) {
            prob -= 1.0f;
        }
        out[i] = g * prob;
    }
}

MINIGPU_REGISTER_KERNEL("mse_loss.fp32", "mse_loss", "fp32", "mse_loss_fp32")
MINIGPU_REGISTER_KERNEL("mse_loss_backward.fp32", "mse_loss_backward", "fp32", "mse_loss_backward_fp32")
MINIGPU_REGISTER_KERNEL("l1_loss.fp32", "l1_loss", "fp32", "l1_loss_fp32")
MINIGPU_REGISTER_KERNEL("l1_loss_backward.fp32", "l1_loss_backward", "fp32", "l1_loss_backward_fp32")
MINIGPU_REGISTER_KERNEL("l2_loss.fp32", "l2_loss", "fp32", "l2_loss_fp32")
MINIGPU_REGISTER_KERNEL("l2_loss_backward.fp32", "l2_loss_backward", "fp32", "l2_loss_backward_fp32")
MINIGPU_REGISTER_KERNEL("cross_entropy_stats.fp32", "cross_entropy_stats", "fp32", "cross_entropy_stats_fp32")
MINIGPU_REGISTER_KERNEL("cross_entropy_finish.fp32", "cross_entropy_finish", "fp32", "cross_entropy_finish_fp32")
MINIGPU_REGISTER_KERNEL("cross_entropy_debug.fp32", "cross_entropy_debug", "fp32", "cross_entropy_debug_fp32")
MINIGPU_REGISTER_KERNEL("cross_entropy_debug_finish.fp32", "cross_entropy_debug_finish", "fp32", "cross_entropy_debug_finish_fp32")
MINIGPU_REGISTER_KERNEL("cross_entropy_backward.fp32", "cross_entropy_backward", "fp32", "cross_entropy_backward_fp32")
