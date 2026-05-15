#include "minigpu_kernel.h"

__global__ void sqrt_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = sqrtf(a[i]);
    }
}
