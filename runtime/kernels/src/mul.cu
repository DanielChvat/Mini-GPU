#include "minigpu_kernel.h"

template <typename T>
__global__ void mul_kernel(T *a, T *b, T *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = a[i] * b[i];
    }
}
