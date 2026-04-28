__global__ void relu(int *input, int *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int value = input[i];

        if (value < 0) {
            output[i] = 0;
        } else {
            output[i] = value;
        }
    }
}
