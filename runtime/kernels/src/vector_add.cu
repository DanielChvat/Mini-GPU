__global__ void vector_add(int *a, int *b, int *c) {
    int tid = threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

__global__ void vector_add_i16(int16_t *a, int16_t *b, int16_t *c) {
    int tid = threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

__global__ void vector_add_i8(int8_t *a, int8_t *b, int8_t *c) {
    int tid = threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

__global__ void vector_add_fp32(float *a, float *b, float *c) {
    int tid = threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

__global__ void vector_add_fp16(half *a, half *b, half *c) {
    int tid = threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

__global__ void vector_add_fp8(fp8_e4m3 *a, fp8_e4m3 *b, fp8_e4m3 *c) {
    int tid = threadIdx.x;
    c[tid] = a[tid] + b[tid];
}
