__global__ void linear(int *input, int *weight, int *out,
                       int total, int out_features, int in_features) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int row = idx / out_features;
        int col = idx - row * out_features;
        int sum = 0;

        for (int inner = 0; inner < in_features; inner += 1) {
            sum += input[row * in_features + inner] *
                   weight[col * in_features + inner];
        }

        out[idx] = sum;
    }
}

__global__ void linear_fp32(float *input, float *weight, float *out,
                            int total, int out_features, int in_features) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int row = idx / out_features;
        int col = idx - row * out_features;
        float sum = 0;

        for (int inner = 0; inner < in_features; inner += 1) {
            float lhs = input[row * in_features + inner];
            float rhs = weight[col * in_features + inner];
            if (lhs != 0) {
                if (rhs != 0) {
                    sum += lhs * rhs;
                }
            }
        }

        out[idx] = sum;
    }
}
