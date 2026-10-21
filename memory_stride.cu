#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void check(cudaError_t err)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void strideRead(const float *a, float *b,
                           int n, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        b[i] = a[(size_t)i * stride] + 1.0f;
}

int main()
{
    const int n = 1 << 20;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    const size_t input_bytes =
        (size_t)n * 32 * sizeof(float);
    const size_t output_bytes =
        (size_t)n * sizeof(float);

    float *d_a, *d_b;
    check(cudaMalloc((void **)&d_a, input_bytes));
    check(cudaMalloc((void **)&d_b, output_bytes));

    check(cudaMemset(d_a, 0, input_bytes));

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    int strides[] = {1, 32};
    float times[2];

    for (int k = 0; k < 2; k++) {
        int stride = strides[k];

        // Warm-up: exclude first-launch setup from timing.
        strideRead<<<blocks, threads>>>(d_a, d_b, n, stride);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());

        check(cudaEventRecord(start));

        strideRead<<<blocks, threads>>>(d_a, d_b, n, stride);
        check(cudaGetLastError());

        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        check(cudaEventElapsedTime(&times[k], start, stop));

        float result;
        check(cudaMemcpy(&result, d_b + 100,
                         sizeof(float),
                         cudaMemcpyDeviceToHost));

        printf("Stride %d: %.6f ms, b[100] = %.1f\n",
               stride, times[k], result);
    }

    printf("Stride 32 / stride 1 time ratio: %.2fx\n",
           times[1] / times[0]);

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(d_a));
    check(cudaFree(d_b));

    return 0;
}
