#include "minigpu_kernel.h"

template <typename T>
__global__ void linear_kernel(T *input, T *weight, T *out,
                              int total, int out_features, int in_features) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int row = idx / out_features;
        int col = idx - row * out_features;
        T sum = 0;

        for (int inner = 0; inner < in_features; inner += 1) {
            sum += input[row * in_features + inner] *
                   weight[col * in_features + inner];
        }

        out[idx] = sum;
    }
}
