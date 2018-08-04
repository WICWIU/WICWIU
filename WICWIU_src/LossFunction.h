#ifndef LossFunction_H_
#define LossFunction_H_

#include "Module_utils.h"

template<typename DTYPE> class LossFunction {
private:
    Tensor<DTYPE> *m_aResult;
    Tensor<DTYPE> *m_aGradient;

    Operator<DTYPE> *m_pInputOperator;
    Tensor<DTYPE> *m_pInputTensor;

    Operator<DTYPE> *m_pLabel;

    std::string m_name;

    Device m_Device;

#ifdef __CUDNN__
    cudnnHandle_t m_pCudnnHandle;
#endif  // if __CUDNN__

public:
    LossFunction(std::string pName = "NO NAME");
    LossFunction(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName = "NO NAME");

    virtual ~LossFunction();

    virtual int            Alloc(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel);
    virtual void           Delete();

    void                   SetResult(Tensor<DTYPE> *pTensor);
    void                   SetGradient(Tensor<DTYPE> *pTensor);

    Tensor<DTYPE>        * GetResult() const;
    Tensor<DTYPE>        * GetGradient() const;
    Operator<DTYPE>      * GetOperator() const;
    Tensor<DTYPE>        * GetTensor() const;
    Operator<DTYPE>      * GetLabel() const;
    std::string            GetName() const;

    // For Propagate
    virtual Tensor<DTYPE>* ForwardPropagate(int pTime = 0);
    virtual Tensor<DTYPE>* BackPropagate(int pTime = 0);

#ifdef __CUDNN__
    virtual Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0);
    virtual Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0);
#endif  // if __CUDNN__

    DTYPE                & operator[](unsigned int index);

    virtual void           SetDeviceCPU();
#ifdef __CUDNN__

    // Setting Supporter
    virtual int    SetResultOnCPU();
    virtual int    SetGradientOnCPU();

    virtual void   SetDeviceGPU();
    virtual void   SetDeviceGPU(cudnnHandle_t& pCudnnHandle);
    virtual void   InitializeAttributeForGPU();

    cudnnHandle_t& GetCudnnHandle();

    // Setting Supporter
    virtual int    SetResultOnGPU();
    virtual int    SetGradientOnGPU();

#endif  // if __CUDNN__

    virtual Device GetDevice() {
        return m_Device;
    }

    // reset value
    int ResetResult();
    int ResetGradient();
};

#endif  // LossFunction_H_
