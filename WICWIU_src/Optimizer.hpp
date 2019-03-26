#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_    value

#include "LossFunction_utils.hpp"

enum OptimizeDirection {
    MAXIMIZE,
    MINIMIZE
};

template<typename DTYPE> class Optimizer {
private:
    float m_LearningRate;
    int m_OptimizeDirection;  // 1 or -1
    float m_weightDecayRate;

    Container<Operator<DTYPE> *> *m_ppParameters;
    int m_numOfParameter;

    int m_idOfDevice;

#ifdef __CUDNN__
    cudnnHandle_t m_pCudnnHandle;
#endif  // if __CUDNN__

public:
    Optimizer(Operator<DTYPE> **pParameters, float pLearningRate, OptimizeDirection pOptimizeDirection);
    Optimizer(Container<Operator<DTYPE> *> *pParameters, float pLearningRate, OptimizeDirection pOptimizeDirection);
    Optimizer(Container<Operator<DTYPE> *> *pParameters, float pLearningRate, float pWeightDecayRate, OptimizeDirection pOptimizeDirection);

    virtual ~Optimizer();

    int                           Alloc(Container<Operator<DTYPE> *> *pParameters, float pLearningRate, OptimizeDirection pOptimizeDirection);
    int                           Alloc(Container<Operator<DTYPE> *> *pParameters, float pLearningRate, float pWeightDecayRate, OptimizeDirection pOptimizeDirection);
    int                           Delete();

    virtual int                   UpdateParameter();
    virtual int                   UpdateParameter(Operator<DTYPE> *pTrainableTensor) = 0;

    void                          SetLearningRate(float pLearningRate);
    void                          SetTrainableTensorDegree(int pTrainableTensorDegree);
    void                          SetWeightDecayRate(int pWeightDecayRate);

    float                         GetLearningRate() const;
    int                           GetOptimizeDirection() const;
    Container<Operator<DTYPE> *>* GetTrainableTensor();
    int                           GetTrainableTensorDegree() const;
    float                         GetWeightDecayRate() const;

    int                           ResetParameterGradient();

#ifdef __CUDNN__
    void                          SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice);
    virtual void                  InitializeAttributeForGPU(unsigned int idOfDevice) = 0;
    virtual void                  SetCudnnHandle(cudnnHandle_t& pCudnnHandle);
    virtual int                   UpdateParameterOnGPU();
    virtual int                   UpdateParameterOnGPU(Operator<DTYPE> *pTrainableTensor) = 0;

    cudnnHandle_t& GetCudnnHandle();
    int            GetDeviceID();

#endif  // if __CUDNN__
};


/*!
 * @brief Optimizer 클래스 생성자
 * @details 멤버 변수들을 0 또는 NULL로 초기화하고,
 * @details 전달받은 매개변수를 매개변수로 하여 Optimizer의 Alloc 메소드를 호출한다.
 * @param pParameters Optimizer 클래스의 alloc 메소드의 파라미터로 전달할 Trainable Tensor container
 * @param pLearningRate Optimizer 클래스의 alloc 메소드의 파라미터로 전달할 learning Rate
 * @param pOptimizeDirection Optimizer 클래스의 alloc 메소드의 파라미터로 전달할 optimize Direction
 * @return 없음
 * @see Optimizer<DTYPE>::Alloc(Container<Operator<DTYPE> *> *pParameters, float pLearningRate, OptimizeDirection pOptimizeDirection)
 */
template<typename DTYPE> Optimizer<DTYPE>::Optimizer(Container<Operator<DTYPE> *> *pParameters, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    #ifdef __DEBUG__
    std::cout << "Optimizer::Optimizer(Operator<DTYPE> *, float, OptimizeDirection)" << '\n';
    #endif  // __DEBUG__
    m_LearningRate          = 0.f;
    m_OptimizeDirection     = 1;
    m_ppParameters          = NULL;
    m_numOfParameter = 0;
    m_weightDecayRate       = 0.f;
    m_idOfDevice            = -1;

    Alloc(pParameters, pLearningRate, pOptimizeDirection);
}

/*!
 * @brief Optimizer 클래스 생성자
 * @details 멤버 변수들을 0 또는 NULL로 초기화하고,
 * @details 전달받은 매개변수를 매개변수로 하여 Optimizer의 Alloc 메소드를 호출한다.
 * @param pParameters Optimizer 클래스의 alloc 메소드의 파라미터로 전달할 Trainable Tensor container
 * @param pLearningRate Optimizer 클래스의 alloc 메소드의 파라미터로 전달할 learning Rate
 * @param pWeightDecayRate Optimizer 클래스의 alloc 메소드의 파라미터로 전달할 Weight Decay Rate
 * @param pOptimizeDirection Optimizer 클래스의 alloc 메소드의 파라미터로 전달할 optimize Direction
 * @return 없음
 * @see Optimizer<DTYPE>::Alloc(Container<Operator<DTYPE> *> *pParameters, float pLearningRate, float pWeightDecayRate, OptimizeDirection pOptimizeDirection)
 */
template<typename DTYPE> Optimizer<DTYPE>::Optimizer(Container<Operator<DTYPE> *> *pParameters, float pLearningRate, float pWeightDecayRate, OptimizeDirection pOptimizeDirection) {
    #ifdef __DEBUG__
    std::cout << "Optimizer::Optimizer(Operator<DTYPE> *, float, OptimizeDirection)" << '\n';
    #endif  // __DEBUG__
    m_LearningRate          = 0.f;
    m_OptimizeDirection     = 1;
    m_ppParameters          = NULL;
    m_numOfParameter = 0;
    m_weightDecayRate       = 0.f;
    m_idOfDevice            = -1;

    Alloc(pParameters, pLearningRate, pWeightDecayRate, pOptimizeDirection);
}

/*!
 * @brief Optimizer 클래스 소멸자
 * @details Optimizer<DTYPE>::Delete() 메소드를 호출하고 클래스를 소멸시킨다.
 * @return 없음
 */
template<typename DTYPE> Optimizer<DTYPE>::~Optimizer() {
    #ifdef __DEBUG__
    std::cout << "Optimizer::~Optimizer()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

/*!
 * @brief Optimizer 클래스의 멤버 변수들에 값을 할당하는 메소드
 * @details 매개변수로 전달 받은 값들을 각각 Trainable Tensor Conatiner, learning rate, Optimize Direction, Weight Decay Rate 멤버 변수에 할당한다.
 * @param pParameters Optimizer 클래스에의 Trainable Tensor container 멤버 변수
 * @param pLearningRate Optimizer 클래스의 learning Rate 멤버 변수
 * @param pOptimizeDirection Optimizer 클래스의 optimize Direction 멤버 변수
 * @return TRUE
 */
template<typename DTYPE> int Optimizer<DTYPE>::Alloc(Container<Operator<DTYPE> *> *pParameters, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    #ifdef __DEBUG__
    std::cout << "Optimizer::Alloc(Container<Operator<DTYPE> *> *, float , OptimizeDirection )" << '\n';
    #endif  // __DEBUG__
    m_ppParameters          = pParameters;
    m_numOfParameter = pParameters->GetSize();

    m_LearningRate = pLearningRate;

    if (pOptimizeDirection == MAXIMIZE) m_OptimizeDirection = 1;
    else if (pOptimizeDirection == MINIMIZE) m_OptimizeDirection = -1;

    return TRUE;
}

/*!
 * @brief Optimizer 클래스의 멤버 변수들에 값을 할당하는 메소드
 * @details 매개변수로 전달 받은 값들을 각각 Trainable Tensor Conatiner, learning rate, Optimize Direction, Weight Decay Rate 멤버 변수에 할당한다.
 * @param pParameters Optimizer 클래스에의 Trainable Tensor container 멤버 변수
 * @param pLearningRate Optimizer 클래스의 learning Rate 멤버 변수
 * @param pWeightDecayRate Optimizer 클래스의 Weight Decay Rate 멤버 변수
 * @param pOptimizeDirection Optimizer 클래스의 optimize Direction 멤버 변수
 * @return TRUE
 */
template<typename DTYPE> int Optimizer<DTYPE>::Alloc(Container<Operator<DTYPE> *> *pParameters, float pLearningRate, float pWeightDecayRate, OptimizeDirection pOptimizeDirection) {
    #ifdef __DEBUG__
    std::cout << "Optimizer::Alloc(Container<Operator<DTYPE> *> *, float , OptimizeDirection )" << '\n';
    #endif  // __DEBUG__
    m_ppParameters          = pParameters;
    m_numOfParameter = pParameters->GetSize();

    m_LearningRate = pLearningRate;

    if (pOptimizeDirection == MAXIMIZE) m_OptimizeDirection = 1;
    else if (pOptimizeDirection == MINIMIZE) m_OptimizeDirection = -1;

    m_weightDecayRate = pWeightDecayRate;
    // std::cout << "m_weightDecayRate" << m_weightDecayRate << '\n';

    return TRUE;
}

template<typename DTYPE> int Optimizer<DTYPE>::Delete() {
    return TRUE;
}

/*!
 * @brief Trainable Tensor Container의 Operator들의 파라미터들을 순서대로 업데이트하는 메소드
 * @details 파생 클래스에서 오버라이드해서 사용하는 메소드
 * @return TRUE
 */
template<typename DTYPE> int Optimizer<DTYPE>::UpdateParameter() {
    for (int i = 0; i < m_numOfParameter; i++) {
        if((*m_ppParameters)[i]->GetIsTrainable()) UpdateParameter((*m_ppParameters)[i]);
    }
    return TRUE;
}

#ifdef __CUDNN__

template<typename DTYPE> void Optimizer<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) {
    checkCudaErrors(cudaSetDevice(idOfDevice));
    SetCudnnHandle(pCudnnHandle);
    m_idOfDevice = idOfDevice;
    InitializeAttributeForGPU(idOfDevice);
}

template<typename DTYPE> void Optimizer<DTYPE>::SetCudnnHandle(cudnnHandle_t& pCudnnHandle) {
    m_pCudnnHandle = pCudnnHandle;
}

template<typename DTYPE> int Optimizer<DTYPE>::GetDeviceID() {
    return m_idOfDevice;
}

template<typename DTYPE> cudnnHandle_t& Optimizer<DTYPE>::GetCudnnHandle() {
    return m_pCudnnHandle;
}

/*!
 * @brief GPU를 활용해 Trainable Tensor Container의 Operator들의 파라미터들을 순서대로 업데이트하는 메소드
 * @details 파생 클래스에서 오버라이드해서 사용하는 메소드
 * @return TRUE
 */
template<typename DTYPE> int Optimizer<DTYPE>::UpdateParameterOnGPU() {
    for (int i = 0; i < m_numOfParameter; i++) {
        if((*m_ppParameters)[i]->GetIsTrainable()) UpdateParameterOnGPU((*m_ppParameters)[i]);
    }
    return TRUE;
}

#endif  // if __CUDNN__

template<typename DTYPE> void Optimizer<DTYPE>::SetLearningRate(float pLearningRate) {
    m_LearningRate = pLearningRate;
}

template<typename DTYPE> void Optimizer<DTYPE>::SetTrainableTensorDegree(int pTrainableTensorDegree) {
    m_numOfParameter = pTrainableTensorDegree;
}

template<typename DTYPE> void Optimizer<DTYPE>::SetWeightDecayRate(int pWeightDecayRate) {
    m_weightDecayRate = pWeightDecayRate;
}

template<typename DTYPE> float Optimizer<DTYPE>::GetLearningRate()  const {
    return m_LearningRate;
}

template<typename DTYPE> int Optimizer<DTYPE>::GetOptimizeDirection() const {
    return m_OptimizeDirection;
}

template<typename DTYPE> float Optimizer<DTYPE>::GetWeightDecayRate() const {
    return m_weightDecayRate;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Optimizer<DTYPE>::GetTrainableTensor() {
    return m_ppParameters;
}

template<typename DTYPE> int Optimizer<DTYPE>::GetTrainableTensorDegree() const {
    return m_numOfParameter;
}

/*!
 * @brief Trainable Tensor Container의 Operator들의 Gradient를 초기화하는 메소드
 * @return TRUE
 * @ref Operator<DTYPE>::ResetGradient()
 */
template<typename DTYPE> int Optimizer<DTYPE>::ResetParameterGradient() {
    for (int i = 0; i < m_numOfParameter; i++) {
        (*m_ppParameters)[i]->ResetGradient();
    }

    return TRUE;
}

#endif  // OPTIMIZER_H_
