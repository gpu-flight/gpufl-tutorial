#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(status) << std::endl;     \
            return 1;                                                           \
        }                                                                       \
    } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    constexpr int element_count = 1 << 20;
    constexpr int iterations = 50;
    constexpr int threads_per_block = 256;
    const int blocks = (element_count + threads_per_block - 1) / threads_per_block;
    const size_t bytes = static_cast<size_t>(element_count) * sizeof(float);

    std::vector<float> host_a(element_count);
    std::vector<float> host_b(element_count);
    std::vector<float> host_c(element_count, 0.0f);

    for (int i = 0; i < element_count; ++i) {
        host_a[i] = static_cast<float>(i) * 0.5f;
        host_b[i] = static_cast<float>(i) * 2.0f;
    }

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;

    CHECK_CUDA(cudaMalloc(&dev_a, bytes));
    CHECK_CUDA(cudaMalloc(&dev_b, bytes));
    CHECK_CUDA(cudaMalloc(&dev_c, bytes));

    CHECK_CUDA(cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < iterations; ++i) {
        vector_add<<<blocks, threads_per_block>>>(dev_a, dev_b, dev_c, element_count);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < element_count; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (std::fabs(host_c[i] - expected) > 1e-5f) {
            ok = false;
            std::cerr << "Validation failed at index " << i << ": expected "
                      << expected << ", got " << host_c[i] << std::endl;
            break;
        }
    }

    CHECK_CUDA(cudaFree(dev_a));
    CHECK_CUDA(cudaFree(dev_b));
    CHECK_CUDA(cudaFree(dev_c));

    if (!ok) {
        return 1;
    }

    std::cout << "Vector add completed successfully: " << iterations
              << " kernel launches, " << element_count << " elements" << std::endl;
    return 0;
}
