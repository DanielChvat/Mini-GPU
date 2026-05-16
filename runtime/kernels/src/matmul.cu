#include "minigpu_linalg.h"

template <typename T>
__global__ void matmul_kernel(T *a, T *b, T *out, int m, int n, int k) {
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int row = idx / n;
        int col = idx - row * n;
        out[idx] = minigpu_gemm_element(a, b, row, col, n, k);
    }
}

MINIGPU_REGISTER_TYPED_KERNELS("matmul", "i32,fp32", "matmul_kernel")
