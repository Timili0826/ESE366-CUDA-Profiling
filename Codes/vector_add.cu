#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do {                                      \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error at line %d: %s\n",        \
                __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while (0)

__global__ void vecAdd(const float *a, const float *b,
                       float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Host allocation failed\n");
        free(h_a);
        free(h_b);
        free(h_c);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = 2.0f * i;
    }

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, bytes));
    CHECK(cudaMalloc((void **)&d_b, bytes));
    CHECK(cudaMalloc((void **)&d_c, bytes));

    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    printf("Elements: %d\n", n);
    printf("Blocks: %d, threads per block: %d\n",
           blocks, threads);
    printf("Kernel time: %.6f ms\n", ms);
    printf("c[100] = %f\n", h_c[100]);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
