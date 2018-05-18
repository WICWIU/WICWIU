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
    // #ifdef __DATA_CLASS__
    // Data<DTYPE> *m_aData;
    // #else // ifdef __DATA_CLASS__
    // DTYPE *m_aData;
    // #endif  // __DATA_CLASS__

public:
    Tensor();
    Tensor(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize);  // For 5D-Tensor
    Tensor(Shape *pShape);
    Tensor(Tensor<DTYPE> *pTensor);  // Copy Constructor

    virtual ~Tensor();

    int          Alloc(Shape *pShape);
    int          Alloc(Tensor *pTensor);
    void         Delete();

    Shape      * GetShape();
    Data<DTYPE>* GetData();

    int          GetTimeSize();
    int          GetBatchSize();
    int          GetChannelSize();
    int          GetRowSize();
    int          GetColSize();

    int          GetCapacity();

    DTYPE      * GetLowData(unsigned int pTime = 0);

    ///////////////////////////////////////////////////////////////////

    int  Reshape(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize);
    void Reset();

    ///////////////////////////////////////////////////////////////////

    DTYPE& operator[](unsigned int index);
    // DTYPE& GetDatum(int ti, int ba, int ch, int ro, int co);

    ///////////////////////////////////////////////////////////////////

    static Tensor<DTYPE>* Truncated_normal(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, float mean, float stddev);

    static Tensor<DTYPE>* Zeros(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize);

    static Tensor<DTYPE>* Constants(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, DTYPE constant);

    ///////////////////////////////////////////////////////////////////

    static Tensor<DTYPE>* Add(Tensor<DTYPE> *pRightTensor, Tensor<DTYPE> *pLeftTensor, Tensor<DTYPE> *pDestTensor = NULL);

    static Tensor<DTYPE>* BroadcastAdd(Tensor<DTYPE> *pLeftTensor, Tensor<DTYPE> *pRightTensor, Tensor<DTYPE> *pDestTensor = NULL, int is_inverse = FALSE);

    static Tensor<DTYPE>* Multiply(Tensor<DTYPE> *pLeftTensor, float pMultiplier, Tensor<DTYPE> *pDestTensor = NULL);

    // static Tensor<DTYPE>* Matmul(Tensor<DTYPE> *pRightTensor, Tensor<DTYPE> *pLeftTensor, Tensor<DTYPE> *pDestTensor=NULL);
    //
    // static Tensor<DTYPE>* BroadcastMatmul(Tensor<DTYPE> *pLeftTensor, Tensor<DTYPE> *pRightTensor, Tensor<DTYPE> *pDestTensor=NULL);
};

///////////////////////////////////////////////////////////////////

inline unsigned int Index5D(Shape *pShape, int ti, int ba, int ch, int ro, int co) {
    return (((ti * (*pShape)[1] + ba) * (*pShape)[2] + ch) * (*pShape)[3] + ro) * (*pShape)[4] + co;
}

inline unsigned int Index4D(Shape *pShape, int ba, int ch, int ro, int co) {
    return ((ba * (*pShape)[2] + ch) * (*pShape)[3] + ro) * (*pShape)[4] + co;
}

#endif  // TENSOR_H_
