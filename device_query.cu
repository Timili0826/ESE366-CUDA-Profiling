#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    cudaDeviceProp prop;
    size_t free_mem, total_mem;

    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printf("Device query failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        printf("Memory query failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n",
           prop.major, prop.minor);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n",
           prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n",
           prop.maxThreadsPerMultiProcessor);
    printf("Registers per SM: %d\n",
           prop.regsPerMultiprocessor);
    printf("Shared memory per block: %zu bytes\n",
           prop.sharedMemPerBlock);
    printf("Free memory: %.2f MiB\n",
           free_mem / (1024.0 * 1024.0));
    printf("Total memory: %.2f MiB\n",
           total_mem / (1024.0 * 1024.0));

    return 0;
}
