#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

__global__ void my_kernel() {
    int myThreadID = blockIdx.x *blockDim.x + threadIdx.x;
    printf("ThreadID %d: I am thread %d from block %d \n", myThreadID, threadIdx.x, blockIdx.x);
}

void my_launcher() {

    int numBlocks = 2;
    int threadsPerBlock = 2;            // 10000000
    my_kernel<<<numBlocks, threadsPerBlock>>>();

    cudaError_t code = cudaGetLastError();
    if( code != cudaSuccess)    printf("%s \n", cudaGetErrorString(code));
    cudaDeviceSynchronize();
}

__global__ void my_kernel_2D() {
    int numCols = blockDim.x * gridDim.x;   // Nx
    int numRows = blockDim.y * gridDim.y;   // Ny
    
    int columnID = blockIdx.x * blockDim.x + threadIdx.x;
    int rowID    = blockIdx.y * blockDim.y + threadIdx.y;

    // Row Major threadID
    int myThreadID = rowID * numCols + columnID;
    printf("rowID %d colID %d blockDim.x %d, blockDim.y %d gridDim.x %d gridDim.y %d numRows %d numCols %d myThreadID %d\n", rowID, columnID, blockDim.x, blockDim.y, gridDim.x, gridDim.y, numRows, numCols, myThreadID);
}

void my_launcher_2D() {
    
    const int Nx = 4;
    const int Ny = 3;
    dim3 threadsPerBlock(4, 3);
    dim3 numBlocks(Nx/threadsPerBlock.x, Ny/threadsPerBlock.y);

    my_kernel_2D<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
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