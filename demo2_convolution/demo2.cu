#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define THREADS_PER_BLK 128

__global__ void convolve(int N, float* input, float* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;  // thread local variable
    float result = 0.0f;  // thread-local variable

    for (int i=0; i<3; i++)
        result += input[index + i];

    output[index] = result / 3.f;
}

__global__ void convolve_shared_memory(int N, float* input, float* output) {
    __shared__ float support[THREADS_PER_BLK+2];        // per-block allocation
    int index = blockIdx.x * blockDim.x + threadIdx.x;  // thread local variable

    support[threadIdx.x] = input[index];
    if (threadIdx.x < 2) {
    support[THREADS_PER_BLK + threadIdx.x] = input[index+THREADS_PER_BLK];
    }

    __syncthreads();

    float result = 0.0f;  // thread-local variable
    for (int i=0; i<3; i++)
        result += support[threadIdx.x + i];

    output[index] = result / 3.f;
}

void my_launcher(bool useSharedMemory) {
    // const int N = 1024*1024;
    const int N = 10;
    float* input = new float[N+2];
    for (int i=0; i<N+2; i++) input[i] = i%2;

    int threadsPerBlock = THREADS_PER_BLK;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* devInput = nullptr;
    float* devOutput = nullptr;

    cudaMalloc(&devInput, sizeof(float) * (N+2) );  // allocate input array in device memory
    cudaMalloc(&devOutput, sizeof(float) * N);      // allocate output array in device memory

    cudaMemcpy(devInput, input, (N+2) * sizeof(float), cudaMemcpyHostToDevice);

    // properly initialize contents of devInput here ...
    double startTime = CycleTimer::currentSeconds();
    if(!useSharedMemory)    convolve<<<numBlocks, threadsPerBlock>>>(N, devInput, devOutput);
    else    convolve_shared_memory<<<numBlocks, threadsPerBlock>>>(N, devInput, devOutput);
    double kernelCallEndTime = CycleTimer::currentSeconds();

    cudaDeviceSynchronize();
    double kernelCompleteTime = CycleTimer::currentSeconds();

    float result[N];
    cudaMemcpy(result, devOutput, N * sizeof(float), cudaMemcpyDeviceToHost);

    // for(int i=0; i<N+2; i++)  printf("%f ", input[i]);
    // printf("\n");
    // for(int i=0; i<N; i++)  printf("%f ", result[i]);
    // printf("\n");
    printf("Kernel Call Time       = %.3f\nKernel Completion Time = %.3f \n", kernelCallEndTime - startTime, kernelCompleteTime - startTime);

    delete[] input;
    cudaFree(devInput);
    cudaFree(devOutput);
}

void printCudaInfo() {

    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}