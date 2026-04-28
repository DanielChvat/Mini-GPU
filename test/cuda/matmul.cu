__global__ void matmul(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid / n;
    int col = tid - row * n;

    if (tid < n * n) {
        int sum = 0;

        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }

        c[tid] = sum;
    }
}
