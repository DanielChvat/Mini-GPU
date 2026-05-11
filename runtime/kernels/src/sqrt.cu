__global__ void sqrt_fp32(float *a, float *out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float x = a[i];
        uint32_t ix = minigpu_as_u32(x);
        uint32_t seed = 0x5f3759dfu - (ix >> 1);
        float y = minigpu_as_f32(seed);
        float half = minigpu_as_f32(0x3f000000u);
        float three_halves = minigpu_as_f32(0x3fc00000u);

        y = y * (three_halves - half * x * y * y);
        y = y * (three_halves - half * x * y * y);

        out[i] = x * y;
    }
}
