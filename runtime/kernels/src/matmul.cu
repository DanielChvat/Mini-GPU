__global__ void matmul(int *a, int *b, int *out, int m, int n, int k) {
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int row = idx / n;
        int col = idx - row * n;
        int sum = 0;

        for (int inner = 0; inner < k; inner += 1) {
            sum += a[row * k + inner] * b[inner * n + col];
        }

        out[idx] = sum;
    }
}

__global__ void matmul_fp32(float *a, float *b, float *out, int m, int n, int k) {
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int row = idx / n;
        int col = idx - row * n;
        float sum = 0;

        for (int inner = 0; inner < k; inner += 1) {
            sum += a[row * k + inner] * b[inner * n + col];
        }

        out[idx] = sum;
    }
}
