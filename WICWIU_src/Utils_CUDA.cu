#ifdef __CUDNN__

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>
#include <math.h>

// #include <windows.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "Utils.h"

int gNoCudaDevice = 0;
cudaDeviceProp gCudaDeviceProp[MAX_CUDA_DEVICE];

int gNoActiveCudaDevice = 0;
int gActiveCudaDeviceIndex[MAX_CUDA_DEVICE];

int GetCurrentCudaDevice() {
    int curDevice = -1;

    cudaGetDevice(&curDevice);  // for thread-safety

    return curDevice;
    // return gCurDeviceId;				// not thread safe
}

void GetKernelParameters(int totalThread, int *pNoBlock, int *pThreadsPerBlock, int blockSize) {
    // printf("GetKernelParameters\n");
    int curDevice = GetCurrentCudaDevice();
    // printf("curDevice : %d\n", curDevice);

    if (totalThread < 32) {
        *pThreadsPerBlock = totalThread;
        *pNoBlock         = 1;
				// printf("pThreadsPerBlock : %d pNoBlock : %d \n", *pThreadsPerBlock, *pNoBlock);
    } else if (totalThread < gCudaDeviceProp[curDevice].multiProcessorCount * 2) {
        *pThreadsPerBlock = 1;
        *pNoBlock         = totalThread;
				// printf("pThreadsPerBlock : %d pNoBlock : %d \n", *pThreadsPerBlock, *pNoBlock);
    } else {
        if (blockSize > 0) {
            *pNoBlock         = (totalThread + blockSize - 1) / blockSize;
            *pThreadsPerBlock = min(totalThread, blockSize);
						// printf("pThreadsPerBlock : %d pNoBlock : %d \n", *pThreadsPerBlock, *pNoBlock);
        } else {
            *pNoBlock         = gCudaDeviceProp[curDevice].multiProcessorCount * 2;
            *pThreadsPerBlock = (totalThread + *pNoBlock - 1) / *pNoBlock;
						// printf("pThreadsPerBlock : %d pNoBlock : %d \n", *pThreadsPerBlock, *pNoBlock);
        }

        if (*pThreadsPerBlock > gCudaDeviceProp[curDevice].maxThreadsPerBlock) {
            *pThreadsPerBlock = gCudaDeviceProp[curDevice].maxThreadsPerBlock;
            *pNoBlock         = (totalThread + *pThreadsPerBlock - 1) / *pThreadsPerBlock;
						// printf("pThreadsPerBlock : %d pNoBlock : %d \n", *pThreadsPerBlock, *pNoBlock);
        }

        if (*pNoBlock > gCudaDeviceProp[curDevice].maxGridSize[0]) *pNoBlock = gCudaDeviceProp[curDevice].maxGridSize[0];
    }
}

#endif  // ifdef __CUDNN__
