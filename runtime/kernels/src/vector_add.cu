__global__ void vector_add(int *a, int *b, int *c, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_i16(int16_t *a, int16_t *b, int16_t *c, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_i8(int8_t *a, int8_t *b, int8_t *c, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_fp32(float *a, float *b, float *c, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_fp16(half *a, half *b, half *c, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_fp8_e4m3fn(
    fp8_e4m3fn *a,
    fp8_e4m3fn *b,
    fp8_e4m3fn *c,
    int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        c[i] = a[i] + b[i];
    }
}
