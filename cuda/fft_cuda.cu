//
// Created by mrs on 27.03.22.
//

#include "fft_cuda.hpp"
#include <utility>
#include <iostream>

#define PI 3.14159265358979323846f

#define GPU_ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
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

// Assuming that N <= 1024 and all the threads are located in one block
__global__ void fft_kernel(cf* a, int N) {
    int tid = threadIdx.x;

    __shared__ cf shm[1024];
    shm[tid] = a[tid];
    __syncthreads();

    int to_swap = reverse_id(tid, N);
    // Thread with bigger id manages the swapping

    if (to_swap < tid) {
//        printf("Cuda Swapping %d and %d\n", tid, to_swap);
        cf tmp = shm[tid];
        shm[tid] = shm[to_swap];
        shm[to_swap] = tmp;
    }
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


    a[tid] = shm[tid];
}


void fft_cuda(cf* arr, int N) {
    if (N > 1024) {
        std::cerr << "Cuda implementation of the algorithm currently works only with arrays of size <. 1024" << std::endl;
        return;
    }
    cf *arr_cuda;
    GPU_ERROR_CHECK(cudaMalloc(&arr_cuda, sizeof(cf) * N));
    GPU_ERROR_CHECK(cudaMemcpy(arr_cuda, arr, sizeof(cf) * N, cudaMemcpyHostToDevice));

    // Run kernel assuming that the size of array if less than 1025
    fft_kernel<<<1, N>>>(arr_cuda, N);

    GPU_ERROR_CHECK(cudaDeviceSynchronize());
    GPU_ERROR_CHECK(cudaMemcpy(arr, arr_cuda, sizeof(cf) * N, cudaMemcpyDeviceToHost));
}