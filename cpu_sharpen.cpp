#include "cpu_sharpen.h"
#include <algorithm>

void cpu_sharpen(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int filter[3][3] = {
        { 0, -1, 0 },
        { -1, 5, -1 },
        { 0, -1, 0 }
    };

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            for (int c = 0; c < channels; ++c) {
                int sum = 0;
                for (int fy = -1; fy <= 1; ++fy) {
                    for (int fx = -1; fx <= 1; ++fx) {
                        int img_idx = ((y + fy) * width + (x + fx)) * channels + c;
                        sum += input[img_idx] * filter[fy + 1][fx + 1];
                    }
                }
                int idx = (y * width + x) * channels + c;
                output[idx] = static_cast<unsigned char>(std::min(std::max(sum, 0), 255));
            }
        }
    }
}
