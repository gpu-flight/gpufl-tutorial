#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <iostream>


__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vectorMul(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void vectorScale(int* a, int scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = a[idx];
        for (int i = 0; i < 2048; ++i) {
            val = val * scale + i;
        }
        a[idx] = val;
    }
}

int main() {
    std::cout << "=== GPUFl Block-Style API Demo ===" << std::endl;
    
    const int n = 1 << 22;  // 4M elements
    const size_t bytes = n * sizeof(int);

    // Allocate memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int* h_a = new int[n];
    int* h_b = new int[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    // DIAG: full-grid coverage (was grid(4) = only 1024 threads / 32 warps =
    // near-zero occupancy). Cover all n elements so every SM has resident warps
    // for the PC sampler to catch — tests injection + HEAVY workload.
    dim3 grid((n + block.x - 1) / block.x);

    std::cout << "Running heavy monitored scope..." << std::endl;
    // NVTX scope around the whole heavy region. gpufl pairs the push/pop into
    // one NvtxMarkerEvent (start/end/name), so this region shows up as the
    // "heavy_monitored_scope" scope in gpufl's scope channel — use it to verify
    // the scope correctly brackets the kernels under injection.
    nvtxRangePushA("heavy_monitored_scope");
    for (int i = 0; i < 500; ++i) {  // DIAG: 500 (was 2000) — full grid is far heavier; keep runtime to a few seconds
        // Per-iteration nested range: confirms push/pop pairing + nesting are
        // captured under injection (each launch becomes a "vectorScale_iter"
        // child scope). Drop this inner range if the output is too noisy.
        nvtxRangePushA("vectorScale_iter");
        vectorScale<<<grid, block>>>(d_a, 3, n);
        nvtxRangePop();  // vectorScale_iter
    }
    cudaDeviceSynchronize();
    nvtxRangePop();  // heavy_monitored_scope


    // Cleanup
    delete[] h_a;
    delete[] h_b;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "\n=== Demo Complete ===" << std::endl;

    return 0;
}
