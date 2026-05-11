__global__ void reciprocal_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        uint32_t ix = minigpu_as_u32(a[i]);
        uint32_t sign = ix & 0x80000000u;
        uint32_t mag = ix & 0x7fffffffu;
        uint32_t iy = sign | (0x7ef311c3u - mag);
        float x = a[i];
        float y = minigpu_as_f32(iy);
        float two = minigpu_as_f32(0x40000000u);

        y = y * (two - x * y);
        y = y * (two - x * y);

        out[i] = y;
    }
}

__global__ void div_fp32(float *a, float *b, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        uint32_t ib = minigpu_as_u32(b[i]);
        uint32_t sign = ib & 0x80000000u;
        uint32_t bm = ib & 0x7fffffffu;
        uint32_t iy = sign | (0x7ef311c3u - bm);
        float x = b[i];
        float y = minigpu_as_f32(iy);
        float two = minigpu_as_f32(0x40000000u);

        y = y * (two - x * y);
        y = y * (two - x * y);

        out[i] = a[i] * y;
    }
}
