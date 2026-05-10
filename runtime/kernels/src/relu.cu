__global__ void relu(int *a, int *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int v = a[i];
        if (v < 0) {
            v = 0;
        }
        out[i] = v;
    }
}

__global__ void relu_i16(int16_t *a, int16_t *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int16_t v = a[i];
        if (v < 0) {
            v = 0;
        }
        out[i] = v;
    }
}

__global__ void relu_i8(int8_t *a, int8_t *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int8_t v = a[i];
        if (v < 0) {
            v = 0;
        }
        out[i] = v;
    }
}

__global__ void relu_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = a[i];
        if (v < 0) {
            v = 0;
        }
        out[i] = v;
    }
}

__global__ void relu_fp16(half *a, half *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        half v = a[i];
        if (v < 0) {
            v = 0;
        }
        out[i] = v;
    }
}

__global__ void relu_fp8_e4m3fn(fp8_e4m3fn *a, fp8_e4m3fn *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        fp8_e4m3fn v = a[i];
        if (v < 0) {
            v = 0;
        }
        out[i] = v;
    }
}
