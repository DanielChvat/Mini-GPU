#include "minigpu_kernel.h"

template <typename T>
__global__ void matmul_kernel(T *a, T *b, T *out, int m, int n, int k) {
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int row = idx / n;
        int col = idx - row * n;
        T sum = 0;

        for (int inner = 0; inner < k; inner += 1) {
            sum += a[row * k + inner] * b[inner * n + col];
        }

        out[idx] = sum;
    }
}
