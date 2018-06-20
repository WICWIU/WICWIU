#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_    value

#include "LossFunction_utils.h"

enum OptimizeDirection {
    MAXIMIZE,
    MINIMIZE
};

template<typename DTYPE> class Optimizer {
private:
    float m_LearningRate;
    int m_OptimizeDirection;  // 1 or -1

    Container<Operator<DTYPE> *> *m_ppTrainableTensors;
    int m_TrainableTensorDegree;

#ifdef __CUDNN__
    cudnnHandle_t m_pCudnnHandle;
#endif  // if __CUDNN__

public:
    Optimizer(Operator<DTYPE> **pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection);
    Optimizer(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection);

    virtual ~Optimizer();

    int                           Alloc(Container<Operator<DTYPE> *> *pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection);
    int                           Delete();

    virtual int                   UpdateParameter();
    virtual int                   UpdateParameter(Operator<DTYPE> *pTrainableTensor) = 0;

    void                          SetLearningRate(float pLearningRate);
    void                          SetTrainableTensorDegree(int pTrainableTensorDegree);

    float                         GetLearningRate() const;
    int                           GetOptimizeDirection() const;
    Container<Operator<DTYPE> *>* GetTrainableTensor();
    int                           GetTrainableTensorDegree() const;

    int                           ResetParameterGradient();

#ifdef __CUDNN__

    virtual void   SetCudnnHandle(cudnnHandle_t& pCudnnHandle);
    cudnnHandle_t& GetCudnnHandle();
    virtual int    UpdateParameterOnGPU();
    virtual int    UpdateParameterOnGPU(Operator<DTYPE> *pTrainableTensor) = 0;

#endif  // if __CUDNN__
};

#endif  // OPTIMIZER_H_
