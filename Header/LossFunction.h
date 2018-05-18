#ifndef LossFunction_H_
#define LossFunction_H_

#include "Operator_utils.h"

template<typename DTYPE> class LossFunction {
private:
    Tensor<DTYPE> *m_aResult;
    Tensor<DTYPE> *m_aGradient;

    Operator<DTYPE> *m_pInputOperator;
    Tensor<DTYPE> *m_pInputTensor;

    Operator<DTYPE> *m_pLabel;

    std::string m_name;

    Device m_Device;

    int m_numOfThread;

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
    virtual Tensor<DTYPE>* ForwardPropagate();
    virtual Tensor<DTYPE>* ForwardPropagate(int pTime, int pThreadNum); //

    // For BackPropagate
    virtual Tensor<DTYPE>* BackPropagate();
    virtual Tensor<DTYPE>* BackPropagate(int pTime, int pThreadNum); //

    DTYPE                & operator[](unsigned int index);

    virtual void SetDeviceCPU(); //
    virtual void SetDeviceCPU(int pNumOfThread); //
#ifdef __CUDNN__
    virtual void SetDeviceGPU(); //

#endif  // if __CUDNN__

    virtual Device GetDevice() {
        return m_Device;
    }

    int GetNumOfThread() {
        return m_numOfThread;
    }

    // reset value
    int ResetResult();
    int ResetGradient();
};

#endif  // LossFunction_H_
