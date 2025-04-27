#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <chrono>
#include "stb_image.h"
#include "stb_image_write.h"
#include "cpu_sharpen.h"
#include <cuda_runtime.h>

void cuda_sharpen_naive(unsigned char* input, unsigned char* output, int width, int height, int channels);
void cuda_sharpen_shared(unsigned char* input, unsigned char* output, int width, int height, int channels);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>\n";
        return -1;
    }

    const char* input_filename = argv[1];

    int width, height, channels;
    unsigned char* img = stbi_load(input_filename, &width, &height, &channels, 0);
    if (!img) {
        std::cerr << "Failed to load image: " << input_filename << "\n";
        return -1;
    }

    size_t img_size = width * height * channels;
    unsigned char* cpu_result = new unsigned char[img_size];
    unsigned char* cuda_naive_result = new unsigned char[img_size]; // Corrected typo
    unsigned char* cuda_shared_result = new unsigned char[img_size];

    // CPU sharpen
    auto start = std::chrono::high_resolution_clock::now();
    cpu_sharpen(img, cpu_result, width, height, channels);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU sharpen time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms\n";

    // CUDA Naive sharpen
    start = std::chrono::high_resolution_clock::now();
    cuda_sharpen_naive(img, cuda_naive_result, width, height, channels);
    cudaDeviceSynchronize();  // Ensure kernel finishes before timing
    end = std::chrono::high_resolution_clock::now();
    std::cout << "CUDA Naive sharpen time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms\n";

    // CUDA Shared sharpen
    start = std::chrono::high_resolution_clock::now();
    cuda_sharpen_shared(img, cuda_shared_result, width, height, channels);
    cudaDeviceSynchronize();  // Ensure kernel finishes before timing
    end = std::chrono::high_resolution_clock::now();
    std::cout << "CUDA Shared sharpen time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms\n";

    // Save images
    stbi_write_png("cpu_output.png", width, height, channels, cpu_result, width * channels);
    stbi_write_png("cuda_naive_output.png", width, height, channels, cuda_naive_result, width * channels);
    stbi_write_png("cuda_shared_output.png", width, height, channels, cuda_shared_result, width * channels);

    // Clean up
    stbi_image_free(img);
    delete[] cpu_result;
    delete[] cuda_naive_result;
    delete[] cuda_shared_result;
    
    return 0;
}
