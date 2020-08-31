#ifndef UTILS_H_
#define UTILS_H_

#include <vector>

#ifndef MIN
#define MIN(X, Y) ((X) <= (Y) ? (X) : (Y))
#endif // MIN

#ifndef MAX
#define MAX(X, Y) ((X) >= (Y) ? (X) : (Y))
#endif // MAX

int LogMessageF(const char* logFile, int bOverwrite, const char* format, ...);
int DisplayFeature(int dim, float* data, int width = 0);
int LogFeature(const char* fileName, int bOverwrite, int dim, float* data, int width = 0);

// only for gray images (0.0F ~ 1.0F)
int DisplayImage(int width, int height, float* data);
int LogImage(const char* fileName, int bOverwrite, int width, int height, float* data);

void AllocFeatureVector(int dim, int noSample, std::vector<float*>& vFeature);
void DeleteFeatureVector(std::vector<float*>& vFeature);
void MyPause(const char* message = NULL);

float GetSquareDistance(int dim, float* pVec1, float* pVec2);

#ifdef __CUDNN__

#define MAX_CUDA_DEVICE 32

int GetCurrentCudaDevice();

void GetKernelParameters(int totalThread, int* pNoBlock, int* pThreadsPerBlock,
                         int blockSize = 128);

#endif // ifdef __CUDNN__

#endif // ifndef UTILS_H_
