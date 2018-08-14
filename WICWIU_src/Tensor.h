#ifndef TENSOR_H_
#define TENSOR_H_

#include "Shape.h"
#include "LongArray.h"

enum IsUseTime {
    UseTime,
    NoUseTime
};

template<typename DTYPE> class Tensor {
private:
    Shape *m_aShape;
    LongArray<DTYPE> *m_aLongArray;
    Device m_Device;
    int m_idOfDevice = -1;
    IsUseTime m_IsUseTime;

private:
    int  Alloc(Shape *pShape, IsUseTime pAnswer);
    int  Alloc(Tensor *pTensor);
    void Delete();

public:
    Tensor(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer = UseTime);  // For 5D-Tensor
    Tensor(int pSize0, int pSize1, int pSize2, int pSize3, IsUseTime pAnswer = UseTime);  // For 4D-Tensor
    Tensor(int pSize0, int pSize1, int pSize2, IsUseTime pAnswer = UseTime);  // For 3D-Tensor
    Tensor(int pSize0, int pSize1, IsUseTime pAnswer = UseTime);  // For 2D-Tensor
    Tensor(int pSize0, IsUseTime pAnswer = UseTime);  // For 1D-Tensor
    Tensor(Shape *pShape, IsUseTime pAnswer = UseTime);
    Tensor(Tensor<DTYPE> *pTensor);  // Copy Constructor

    virtual ~Tensor();

    Shape                  * GetShape();
    int                      GetRank();
    int                      GetDim(int pRanknum);
    LongArray<DTYPE>       * GetLongArray();
    int                      GetCapacity();
    int                      GetElement(unsigned int index);
    DTYPE                  & operator[](unsigned int index);
    Device                   GetDevice();
    IsUseTime                GetIsUseTime();
    DTYPE                  * GetCPULongArray(unsigned int pTime = 0);

    int                      GetTimeSize(); // 추후 LongArray의 Timesize 반환
    int                      GetBatchSize(); // 삭제 예정
    int                      GetChannelSize(); // 삭제 예정
    int                      GetRowSize(); // 삭제 예정
    int                      GetColSize(); // 삭제 예정


    int                      ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4);
    int                      ReShape(int pSize0, int pSize1, int pSize2, int pSize3);
    int                      ReShape(int pSize0, int pSize1, int pSize2);
    int                      ReShape(int pSize0, int pSize1);
    int                      ReShape(int pSize0);

    void                     Reset();


    void                     SetDeviceCPU();

    int                      Save(FILE *fileForSave);
    int                      Load(FILE *fileForLoad);
#ifdef __CUDNN__
    void                     SetDeviceGPU(unsigned int idOfDevice);

    DTYPE                  * GetGPUData(unsigned int pTime = 0);
    cudnnTensorDescriptor_t& GetDescriptor();

    void                     Reset(cudnnHandle_t& pCudnnHandle);


#endif  // if __CUDNN__


    static Tensor<DTYPE>* Random_normal(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, float mean, float stddev, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Random_normal(Shape *pShape, float mean, float stddev, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Zeros(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Zeros(Shape *pShape, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Constants(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, DTYPE constant, IsUseTime pAnswer = UseTime);
    static Tensor<DTYPE>* Constants(Shape *pShape, DTYPE constant, IsUseTime pAnswer = UseTime);
};


inline unsigned int Index5D(Shape *pShape, int ti, int ba, int ch, int ro, int co) {
    return (((ti * (*pShape)[1] + ba) * (*pShape)[2] + ch) * (*pShape)[3] + ro) * (*pShape)[4] + co;
}

inline unsigned int Index4D(Shape *pShape, int ba, int ch, int ro, int co) {
    return ((ba * (*pShape)[1] + ch) * (*pShape)[2] + ro) * (*pShape)[3] + co;
}

inline unsigned int Index3D(Shape *pShape, int ch, int ro, int co) {
    return (ch * (*pShape)[1] + ro) * (*pShape)[2] + co;
}

inline unsigned int Index2D(Shape *pShape, int ro, int co) {
    return ro * (*pShape)[1] + co;
}

#endif  // TENSOR_H_
