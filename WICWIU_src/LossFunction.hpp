#ifndef LossFunction_H_
#define LossFunction_H_

#include "Module_utils.hpp"

template<typename DTYPE> class LossFunction {
private:
    Tensor<DTYPE> *m_aResult;
    Tensor<DTYPE> *m_aGradient;

    Operator<DTYPE> *m_pInputOperator;
    Tensor<DTYPE> *m_pInputTensor;

    Operator<DTYPE> *m_pLabel;

    std::string m_name;

    Device m_Device;
    int m_idOfDevice;

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
    virtual Device         GetDevice();
    virtual int            GetDeviceID();

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

    // virtual void   SetDeviceGPU(unsigned int idOfDevice);
    virtual void   SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice);
    virtual void   InitializeAttributeForGPU(unsigned int idOfDevice);

    cudnnHandle_t& GetCudnnHandle();

    // Setting Supporter
    virtual int    SetResultOnGPU(unsigned int idOfDevice);
    virtual int    SetGradientOnGPU(unsigned int idOfDevice);

#endif  // if __CUDNN__

    // reset value
    int ResetResult();
    int ResetGradient();
};

template<typename DTYPE> LossFunction<DTYPE>::LossFunction(std::string pName) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::LossFunction()" << '\n';
    #endif  // __DEBUG__
    m_aResult        = NULL;
    m_aGradient      = NULL;
    m_pInputOperator = NULL;
    m_pInputTensor   = NULL;
    m_pLabel         = NULL;
    m_name           = pName;
    m_Device         = CPU;
    m_idOfDevice     = -1;
}

template<typename DTYPE> LossFunction<DTYPE>::LossFunction(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::LossFunction()" << '\n';
    #endif  // __DEBUG__
    m_aResult        = NULL;
    m_aGradient      = NULL;
    m_pInputOperator = NULL;
    m_pInputTensor   = NULL;
    m_pLabel         = NULL;
    m_name           = pName;
    m_Device         = CPU;
    m_idOfDevice     = -1;
    Alloc(pOperator, pLabel);
}

template<typename DTYPE> LossFunction<DTYPE>::~LossFunction() {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::~LossFunction()" << '\n';
    #endif  // __DEBUG__
    this->Delete();
}

template<typename DTYPE> int LossFunction<DTYPE>::Alloc(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    #endif  // __DEBUG__

    m_pInputOperator = pOperator;
    m_pInputTensor   = m_pInputOperator->GetResult();

    m_pLabel = pLabel;
    return TRUE;
}

template<typename DTYPE> void LossFunction<DTYPE>::Delete() {
    if (m_aResult) {
        delete m_aResult;
        m_aResult = NULL;
    }

    if (m_aGradient) {
        delete m_aGradient;
        m_aGradient = NULL;
    }
}

template<typename DTYPE> void LossFunction<DTYPE>::SetResult(Tensor<DTYPE> *pTensor) {
    m_aResult = pTensor;
}

template<typename DTYPE> void LossFunction<DTYPE>::SetGradient(Tensor<DTYPE> *pTensor) {
    m_aGradient = pTensor;
}

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::GetResult() const {
    return m_aResult;
}

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::GetGradient() const {
    return m_aGradient;
}

template<typename DTYPE> Operator<DTYPE> *LossFunction<DTYPE>::GetOperator() const {
    return m_pInputOperator;
}

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::GetTensor() const {
    return m_pInputTensor;
}

template<typename DTYPE> Operator<DTYPE> *LossFunction<DTYPE>::GetLabel() const {
    return m_pLabel;
}

template<typename DTYPE> std::string LossFunction<DTYPE>::GetName() const {
    return m_name;
}

template<typename DTYPE> Device LossFunction<DTYPE>::GetDevice() {
    return m_Device;
}

template<typename DTYPE> int LossFunction<DTYPE>::GetDeviceID() {
    return m_idOfDevice;
}

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::ForwardPropagate(int pTime) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::ForwardPropagate(int pTime)" << '\n';
    std::cout << this->GetName() << '\n';
    #endif  // __DEBUG__
    return NULL;
}

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::BackPropagate(int pTime) {
    #ifdef __DEBUG__
    std::cout << "LossFunction<DTYPE>::BackPropagate(int pTime)" << '\n';
    std::cout << this->GetName() << '\n';
    #endif  // __DEBUG__
    return NULL;
}

#ifdef __CUDNN__

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return NULL;
}

template<typename DTYPE> Tensor<DTYPE> *LossFunction<DTYPE>::BackPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return NULL;
}

#endif  // __CUDNN__


template<typename DTYPE> DTYPE& LossFunction<DTYPE>::operator[](unsigned int index) {
    return (*m_aResult)[index];
}

template<typename DTYPE> void LossFunction<DTYPE>::SetDeviceCPU() {
    m_Device = CPU;

#ifdef __CUDNN__
    this->SetResultOnCPU();
    this->SetGradientOnCPU();
#endif  // __CUDNN__
}

#ifdef __CUDNN__
template<typename DTYPE> int LossFunction<DTYPE>::SetResultOnCPU() {
    if (m_aResult) m_aResult->SetDeviceCPU();

    return TRUE;
}

template<typename DTYPE> int LossFunction<DTYPE>::SetGradientOnCPU() {
    if (m_aGradient) m_aGradient->SetDeviceCPU();

    return TRUE;
}

// template<typename DTYPE> void LossFunction<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
// m_Device = GPU;
// this->SetResultOnGPU(idOfDevice);
// this->SetGradientOnGPU(idOfDevice);
// }

template<typename DTYPE> void LossFunction<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) {
    checkCudaErrors(cudaSetDevice(idOfDevice));
    m_Device       = GPU;
    m_idOfDevice   = idOfDevice;
    m_pCudnnHandle = pCudnnHandle;
    this->SetResultOnGPU(idOfDevice);
    this->SetGradientOnGPU(idOfDevice);
    this->InitializeAttributeForGPU(idOfDevice);
}

template<typename DTYPE> void LossFunction<DTYPE>::InitializeAttributeForGPU(unsigned int idOfDevice) {}

template<typename DTYPE> int LossFunction<DTYPE >::SetResultOnGPU(unsigned int idOfDevice) {
    if (m_aResult) m_aResult->SetDeviceGPU(idOfDevice);

    return TRUE;
}

template<typename DTYPE> int LossFunction<DTYPE>::SetGradientOnGPU(unsigned int idOfDevice) {
    if (m_aGradient) m_aGradient->SetDeviceGPU(idOfDevice);

    return TRUE;
}

template<typename DTYPE> cudnnHandle_t& LossFunction<DTYPE>::GetCudnnHandle() {
    return m_pCudnnHandle;
}

#endif  // __CUDNN__


template<typename DTYPE> int LossFunction<DTYPE>::ResetResult() {
    if (m_Device == CPU) {
        if (m_aResult) m_aResult->Reset();
    }

#ifdef __CUDNN__
    else if (m_Device == GPU) {
        if (m_aResult) m_aResult->Reset(this->GetCudnnHandle());
    }
#endif  // if __CUDNN__

    else return FALSE;

    return TRUE;
}

template<typename DTYPE> int LossFunction<DTYPE>::ResetGradient() {
    if (m_Device == CPU) {
        if (m_aGradient) m_aGradient->Reset();
    }

#ifdef __CUDNN__
    else if (m_Device == GPU) {
        if (m_aGradient) m_aGradient->Reset(this->GetCudnnHandle());
    }
#endif  // if __CUDNN__

    else return FALSE;

    return TRUE;
}

#endif  // LossFunction_H_
