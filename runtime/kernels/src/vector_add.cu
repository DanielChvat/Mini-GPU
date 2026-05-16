#include "minigpu_kernel.h"

template <typename T>
__global__ void vector_add_kernel(T *a, T *b, T *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = a[i] + b[i];
    }
}

MINIGPU_REGISTER_TYPED_KERNELS("vector_add", "i32,i16,i8,fp32,fp16,fp8_e4m3fn", "vector_add_kernel")
