#include <cuda_runtime.h>

__global__ void sharpen_shared_kernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    __shared__ unsigned char tile[18][18][4];

    int x = blockIdx.x * 16 + threadIdx.x - 1;
    int y = blockIdx.y * 16 + threadIdx.y - 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int c = 0; c < channels; ++c) {
        int value = 0;
        if (x >= 0 && x < width && y >= 0 && y < height)
            value = input[(y * width + x) * channels + c];
        tile[ty][tx][c] = value;
    }

    __syncthreads();

    if (tx > 0 && tx < 17 && ty > 0 && ty < 17) {
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
                        sum += tile[ty + fy][tx + fx][c] * filter[fy + 1][fx + 1];
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)min(max(sum, 0), 255);
            }
        }
    }
}

void cuda_sharpen_shared(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    unsigned char *d_input, *d_output;
    size_t img_size = width * height * channels;

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);

    cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);

    dim3 block(18, 18);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    sharpen_shared_kernel<<<grid, block>>>(d_input, d_output, width, height, channels);

    cudaMemcpy(output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
