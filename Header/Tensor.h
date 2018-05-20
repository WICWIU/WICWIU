#ifndef TENSOR_H_
#define TENSOR_H_

#include <time.h>
#include <math.h>
#include <chrono>
#include <random>

#include "Shape.h"
#include "Data.h"

template<typename DTYPE> class Tensor {
private:
    Shape *m_aShape;
    Data<DTYPE> *m_aData;

    Device m_Device;

public:
    Tensor();
    Tensor(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize);  // For 5D-Tensor
    Tensor(Shape *pShape);
    Tensor(Tensor<DTYPE> *pTensor);  // Copy Constructor

    virtual ~Tensor();

    int                      Alloc(Shape *pShape);
    int                      Alloc(Tensor *pTensor);
    void                     Delete();

    Shape                  * GetShape();
    Data<DTYPE>            * GetData();

    int                      GetTimeSize();
    int                      GetBatchSize();
    int                      GetChannelSize();
    int                      GetRowSize();
    int                      GetColSize();

    int                      GetCapacity();

    DTYPE                  * GetHostData(unsigned int pTime = 0);

#ifdef __CUDNN__
    DTYPE                  * GetDeviceData(unsigned int pTime = 0);
    void                     MemcpyDeviceToHost();
    void                     MemcpyHostToDevice();

    cudnnTensorDescriptor_t& GetDescriptor();
#endif  // if __CUDNN__

    ///////////////////////////////////////////////////////////////////

    int  Reshape(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize);

    void Reset();
#ifdef __CUDNN__
    void Reset(cudnnHandle_t& pCudnnHandle);
#endif  // ifdef __CUDNN__


    ///////////////////////////////////////////////////////////////////

    DTYPE& operator[](unsigned int index);

    ///////////////////////////////////////////////////////////////////

    static Tensor<DTYPE>* Truncated_normal(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, float mean, float stddev);

    static Tensor<DTYPE>* Zeros(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize);

    static Tensor<DTYPE>* Constants(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, DTYPE constant);
};

///////////////////////////////////////////////////////////////////

inline unsigned int Index5D(Shape *pShape, int ti, int ba, int ch, int ro, int co) {
    return (((ti * (*pShape)[1] + ba) * (*pShape)[2] + ch) * (*pShape)[3] + ro) * (*pShape)[4] + co;
}

inline unsigned int Index4D(Shape *pShape, int ba, int ch, int ro, int co) {
    return ((ba * (*pShape)[2] + ch) * (*pShape)[3] + ro) * (*pShape)[4] + co;
}

#endif  // TENSOR_H_
