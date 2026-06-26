#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

constexpr int kStreamCount = 4;
constexpr int kChunksPerIteration = 8;
constexpr int kIterations = 25;
constexpr int kChunkElements = 1 << 18;
constexpr int kThreadsPerBlock = 256;
constexpr int kSyntheticWork = 32;

void check_cuda(cudaError_t status, const char* call, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                             std::to_string(line) + " while running " + call +
                             " - " + cudaGetErrorString(status));
}

#define CHECK_CUDA(call) check_cuda((call), #call, __FILE__, __LINE__)

struct CudaResources {
    float* host_a = nullptr;
    float* host_b = nullptr;
    float* host_c = nullptr;
    std::array<float*, kStreamCount> dev_a{};
    std::array<float*, kStreamCount> dev_b{};
    std::array<float*, kStreamCount> dev_c{};
    std::array<cudaStream_t, kStreamCount> streams{};

    ~CudaResources() {
        for (cudaStream_t stream : streams) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
        }

        for (int i = 0; i < kStreamCount; ++i) {
            cudaFree(dev_a[i]);
            cudaFree(dev_b[i]);
            cudaFree(dev_c[i]);
        }

        cudaFreeHost(host_a);
        cudaFreeHost(host_b);
        cudaFreeHost(host_c);
    }
};

__global__ void transform_kernel(const float* a,
                                 const float* b,
                                 float* c,
                                 int element_count,
                                 int iteration,
                                 int synthetic_work) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= element_count) {
        return;
    }

    float value = a[idx];
    const float bias = b[idx] * 0.000001f;

    for (int work = 0; work < synthetic_work; ++work) {
        value = fmaf(value, 1.000001f, bias);
    }

    c[idx] = value + b[idx] + static_cast<float>(iteration) * 0.01f;
}

float expected_value(float a, float b, int iteration) {
    float value = a;
    const float bias = b * 0.000001f;

    for (int work = 0; work < kSyntheticWork; ++work) {
        value = std::fma(value, 1.000001f, bias);
    }

    return value + b + static_cast<float>(iteration) * 0.01f;
}

void initialize_inputs(float* host_a, float* host_b, float* host_c, int element_count) {
    for (int i = 0; i < element_count; ++i) {
        host_a[i] = static_cast<float>(i % 1024) * 0.5f;
        host_b[i] = static_cast<float>((i * 7) % 2048) * 0.25f;
        host_c[i] = 0.0f;
    }
}

bool validate_results(const float* host_a,
                      const float* host_b,
                      const float* host_c,
                      int element_count,
                      int final_iteration) {
    constexpr float kTolerance = 1e-2f;

    for (int i = 0; i < element_count; i += 97) {
        const float expected = expected_value(host_a[i], host_b[i], final_iteration);
        const float actual = host_c[i];

        if (std::fabs(actual - expected) > kTolerance) {
            std::cerr << "Validation failed at index " << i << ": expected "
                      << expected << ", got " << actual << std::endl;
            return false;
        }
    }

    return true;
}

int main() {
    try {
        constexpr int total_elements = kChunksPerIteration * kChunkElements;
        constexpr size_t total_bytes =
            static_cast<size_t>(total_elements) * sizeof(float);
        constexpr size_t chunk_bytes =
            static_cast<size_t>(kChunkElements) * sizeof(float);
        constexpr int blocks =
            (kChunkElements + kThreadsPerBlock - 1) / kThreadsPerBlock;

        int device = 0;
        cudaDeviceProp props{};
        CHECK_CUDA(cudaGetDevice(&device));
        CHECK_CUDA(cudaGetDeviceProperties(&props, device));

        CudaResources resources;

        CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&resources.host_a),
                                  total_bytes));
        CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&resources.host_b),
                                  total_bytes));
        CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&resources.host_c),
                                  total_bytes));

        for (int i = 0; i < kStreamCount; ++i) {
            CHECK_CUDA(cudaStreamCreateWithFlags(&resources.streams[i],
                                                 cudaStreamNonBlocking));
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&resources.dev_a[i]),
                                  chunk_bytes));
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&resources.dev_b[i]),
                                  chunk_bytes));
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&resources.dev_c[i]),
                                  chunk_bytes));
        }

        initialize_inputs(resources.host_a, resources.host_b, resources.host_c,
                          total_elements);

        const auto start = std::chrono::steady_clock::now();

        for (int iteration = 0; iteration < kIterations; ++iteration) {
            for (int chunk = 0; chunk < kChunksPerIteration; ++chunk) {
                const int stream_index = chunk % kStreamCount;
                const int offset = chunk * kChunkElements;
                cudaStream_t stream = resources.streams[stream_index];

                CHECK_CUDA(cudaMemcpyAsync(resources.dev_a[stream_index],
                                           resources.host_a + offset,
                                           chunk_bytes,
                                           cudaMemcpyHostToDevice,
                                           stream));
                CHECK_CUDA(cudaMemcpyAsync(resources.dev_b[stream_index],
                                           resources.host_b + offset,
                                           chunk_bytes,
                                           cudaMemcpyHostToDevice,
                                           stream));

                transform_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
                    resources.dev_a[stream_index],
                    resources.dev_b[stream_index],
                    resources.dev_c[stream_index],
                    kChunkElements,
                    iteration,
                    kSyntheticWork);
                CHECK_CUDA(cudaGetLastError());

                CHECK_CUDA(cudaMemcpyAsync(resources.host_c + offset,
                                           resources.dev_c[stream_index],
                                           chunk_bytes,
                                           cudaMemcpyDeviceToHost,
                                           stream));
            }
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        const int final_iteration = kIterations - 1;
        if (!validate_results(resources.host_a, resources.host_b, resources.host_c,
                              total_elements, final_iteration)) {
            return 1;
        }

        const int kernel_launches = kIterations * kChunksPerIteration;
        const int async_copies = kernel_launches * 3;

        std::cout << "Explicit CUDA streams completed successfully" << std::endl;
        std::cout << "  gpu:                " << props.name << std::endl;
        std::cout << "  streams:            " << kStreamCount << std::endl;
        std::cout << "  chunks/iteration:   " << kChunksPerIteration << std::endl;
        std::cout << "  elements/chunk:     " << kChunkElements << std::endl;
        std::cout << "  iterations:         " << kIterations << std::endl;
        std::cout << "  kernel launches:    " << kernel_launches << std::endl;
        std::cout << "  async copies:       " << async_copies << std::endl;
        std::cout << "  concurrent kernels: "
                  << (props.concurrentKernels ? "yes" : "no") << std::endl;
        std::cout << "  async copy engines: " << props.asyncEngineCount << std::endl;
        std::cout << "  elapsed:            " << std::fixed << std::setprecision(2)
                  << elapsed_ms << " ms" << std::endl;

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        cudaDeviceReset();
        return 1;
    }
}
