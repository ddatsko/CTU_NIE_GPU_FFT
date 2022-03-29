//
// Created by mrs on 27.03.22.
//

#include "fft_cuda.hpp"
#include <utility>
#include <iostream>

const float PI = 3.14159265358979323846;

const int BLOCK_SIZE = 1024;

#define GPU_ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

//TODO: check this formula and try to optimize
__device__ int reverse_id(int id, int N) {
    int bit = 0;
    N >>= 1;
    int res = 0;
    while (N != 0) {
        if (N & id) {
            res ^= (1 << bit);
        }
        bit++;
        N >>= 1;
    }
    return res;
}

__global__ void index_swap_kernel(cf* arr, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_to_swap = reverse_id(idx, N);
    if (idx > idx_to_swap) {
        cf tmp = arr[idx];
        arr[idx] = arr[idx_to_swap];
        arr[idx_to_swap] = tmp;
    }
}

// Assuming that N <= 1024 and all the threads are located in one block
__global__ void fft_kernel(cf *a, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ cf shm[BLOCK_SIZE];
    shm[tid] = a[idx];
    __syncthreads();


    for (int len = 2; len <= N; len <<= 1) {
        int j = tid % len;
        // lower part of the chunk takes care about combining two values
        if (j < len / 2) {
            int tid_comb = tid + len / 2;
            float ang = (2.0f * PI / static_cast<float>(len)) * static_cast<float>(j);
            cf w{__cosf(ang), __sinf(ang)};

            cf u = shm[tid], v = cuCmulf(shm[tid_comb], w);

//            printf("Cuda len = %d, Combination on %d and %d with w = %f,%f u=%f,%f, v=%f,%f\n", len, tid, tid_comb, w.x, w.y, u.x, u.y, v.x, v.y);

            shm[tid] = cuCaddf(v, u);
            shm[tid_comb] = cuCsubf(u, v);
        }
        __syncthreads();
    }
    a[idx] = shm[tid];
}

__global__ void fft_kernel_global(cf *a, int len) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int j = idx % len;
    if (j < len / 2) {
        int idx_comb = idx + len / 2;
        float ang = (2.0f * PI * static_cast<float>(j)) / static_cast<float>(len) ;
        cf w{__cosf(ang), __sinf(ang)};

        cf u = a[idx], v = cuCmulf(a[idx_comb], w);
        a[idx] = cuCaddf(v, u);
        a[idx_comb] = cuCsubf(u, v);
    }
}


void fft_cuda(cf* arr, int N) {
    cf *arr_cuda;
    GPU_ERROR_CHECK(cudaMalloc(&arr_cuda, sizeof(cf) * N));
    GPU_ERROR_CHECK(cudaMemcpy(arr_cuda, arr, sizeof(cf) * N, cudaMemcpyHostToDevice));

    if (N <= BLOCK_SIZE) {
        index_swap_kernel<<<1, N>>>(arr_cuda, N);
        fft_kernel<<<1, N>>>(arr_cuda, N);
    } else {
        index_swap_kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(arr_cuda, N);
        // N is power of 2, so a multiple of 1024
        // Firstly, run kernels for each 1024 elements
        fft_kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(arr_cuda, BLOCK_SIZE);
        // Now, we need to finish the execution of last loop iterations, but do it using only global memory
        for (int len = BLOCK_SIZE * 2; len <= N; len <<= 1) {
            fft_kernel_global<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(arr_cuda, len);
        }
    }

    GPU_ERROR_CHECK(cudaDeviceSynchronize());
    GPU_ERROR_CHECK(cudaMemcpy(arr, arr_cuda, sizeof(cf) * N, cudaMemcpyDeviceToHost));
}