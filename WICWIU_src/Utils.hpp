#ifndef UTILS_H_
#define UTILS_H_    value

#ifdef __CUDNN__

#define MAX_CUDA_DEVICE 32

int  GetCurrentCudaDevice();

void GetKernelParameters(int totalThread, int *pNoBlock, int *pThreadsPerBlock, int blockSize = 128);

#endif  // ifdef __CUDNN__


#endif  // ifndef UTILS_H_
