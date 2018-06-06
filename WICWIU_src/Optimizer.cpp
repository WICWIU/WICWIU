#include "Optimizer.h"

template class Optimizer<int>;
template class Optimizer<float>;
template class Optimizer<double>;

template<typename DTYPE> Optimizer<DTYPE>::Optimizer(Container<Tensorholder<DTYPE> *> *pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    #if __DEBUG__
    std::cout << "Optimizer::Optimizer(Operator<DTYPE> *, float, OptimizeDirection)" << '\n';
    #endif  // __DEBUG__
    m_LearningRate          = 0.f;
    m_OptimizeDirection     = 1;
    m_ppTrainableTensors    = NULL;
    m_TrainableTensorDegree = 0;

    Alloc(pTrainableTensors, pLearningRate, pOptimizeDirection);
}

template<typename DTYPE> Optimizer<DTYPE>::~Optimizer() {
    #if __DEBUG__
    std::cout << "Optimizer::~Optimizer()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

template<typename DTYPE> int Optimizer<DTYPE>::Alloc(Container<Tensorholder<DTYPE> *> *pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    #if __DEBUG__
    std::cout << "Optimizer::Alloc(Container<Tensorholder<DTYPE> *> *, float , OptimizeDirection )" << '\n';
    #endif  // __DEBUG__
    m_ppTrainableTensors    = pTrainableTensors;
    m_TrainableTensorDegree = pTrainableTensors->GetSize();

    m_LearningRate = pLearningRate;

    if (pOptimizeDirection == MAXIMIZE) m_OptimizeDirection = 1;
    else if (pOptimizeDirection == MINIMIZE) m_OptimizeDirection = -1;

    return TRUE;
}

template<typename DTYPE> int Optimizer<DTYPE>::Delete() {
    return TRUE;
}

template<typename DTYPE> int Optimizer<DTYPE>::UpdateVariable() {
    for (int i = 0; i < m_TrainableTensorDegree; i++) {
        UpdateVariable((*m_ppTrainableTensors)[i]);
    }
    return TRUE;
}

#if __CUDNN__

template<typename DTYPE> void Optimizer<DTYPE>::SetCudnnHandle(cudnnHandle_t& pCudnnHandle) {
    m_pCudnnHandle = pCudnnHandle;
}

template<typename DTYPE> cudnnHandle_t& Optimizer<DTYPE>::GetCudnnHandle() {
    return m_pCudnnHandle;
}

template<typename DTYPE> int Optimizer<DTYPE>::UpdateVariableOnGPU() {
    for (int i = 0; i < m_TrainableTensorDegree; i++) {
        UpdateVariableOnGPU((*m_ppTrainableTensors)[i]);
    }
    return TRUE;
}

#endif  // if __CUDNN__

template<typename DTYPE> void Optimizer<DTYPE>::SetLearningRate(float pLearningRate) {
    m_LearningRate = pLearningRate;
}

template<typename DTYPE> void Optimizer<DTYPE>::SetTrainableTensorDegree(int pTrainableTensorDegree) {
    m_TrainableTensorDegree = pTrainableTensorDegree;
}

template<typename DTYPE> float Optimizer<DTYPE>::GetLearningRate()  const {
    return m_LearningRate;
}

template<typename DTYPE> int Optimizer<DTYPE>::GetOptimizeDirection() const {
    return m_OptimizeDirection;
}

template<typename DTYPE> Container<Tensorholder<DTYPE> *> *Optimizer<DTYPE>::GetTrainableTensor() {
    return m_ppTrainableTensors;
}

template<typename DTYPE> int Optimizer<DTYPE>::GetTrainableTensorDegree() const {
    return m_TrainableTensorDegree;
}

template<typename DTYPE> int Optimizer<DTYPE>::ResetParameterGradient() {
    for (int i = 0; i < m_TrainableTensorDegree; i++) {
        (*m_ppTrainableTensors)[i]->ResetGradient();
    }

    return TRUE;
}
