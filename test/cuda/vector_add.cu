__global__ void vector_add_i8(int8_t *a, int8_t *b, int8_t *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void vector_add_i16(int16_t *a, int16_t *b, int16_t *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void vector_add_fp32(float *a, float *b, float *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void vector_add_fp16(half *a, half *b, half *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void vector_add_fp8(fp8_e4m3 *a, fp8_e4m3 *b, fp8_e4m3 *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
