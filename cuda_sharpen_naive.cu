#include <cuda_runtime.h>

__global__ void sharpen_naive_kernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        for (int c = 0; c < channels; ++c) {
            int sum = 0;
            int filter[3][3] = {
                { 0, -1, 0 },
                { -1, 5, -1 },
                { 0, -1, 0 }
            };
            for (int fy = -1; fy <= 1; ++fy) {
                for (int fx = -1; fx <= 1; ++fx) {
                    int idx = ((y + fy) * width + (x + fx)) * channels + c;
                    sum += input[idx] * filter[fy + 1][fx + 1];
                }
            }
            int out_idx = (y * width + x) * channels + c;
            output[out_idx] = (unsigned char)min(max(sum, 0), 255);
        }
    }
}

void cuda_sharpen_naive(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    unsigned char *d_input, *d_output;
    size_t img_size = width * height * channels;

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);

    cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    sharpen_naive_kernel<<<grid, block>>>(d_input, d_output, width, height, channels);

    cudaMemcpy(output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
