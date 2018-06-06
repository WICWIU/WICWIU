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

    Container<Tensorholder<DTYPE> *> *m_ppTrainableTensors;
    int m_TrainableTensorDegree;

#if __CUDNN__
    cudnnHandle_t m_pCudnnHandle;
#endif  // if __CUDNN__

public:
    Optimizer(Operator<DTYPE> **pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection);
    Optimizer(Container<Tensorholder<DTYPE> *> *pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection);


    virtual ~Optimizer();

    // ===============

    int Alloc(Container<Tensorholder<DTYPE> *> *pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection);

    int Delete();

    // int AddTrainableTensor(Operator<DTYPE> **pTrainableTensors);
    // int AddTrainableTensor(Operator<DTYPE> *pTrainableTensor);

    // ===============
    virtual int UpdateVariable();

    virtual int UpdateVariable(Tensorholder<DTYPE> *pTrainableTensor) = 0;

#if __CUDNN__

    virtual void   SetCudnnHandle(cudnnHandle_t& pCudnnHandle);

    cudnnHandle_t& GetCudnnHandle();

    virtual int    UpdateVariableOnGPU();

    virtual int    UpdateVariableOnGPU(Tensorholder<DTYPE> *pTrainableTensor) = 0;

#endif  // if __CUDNN__
    // ===============

    void                              SetLearningRate(float pLearningRate);

    void                              SetTrainableTensorDegree(int pTrainableTensorDegree);

    float                             GetLearningRate() const;

    int                               GetOptimizeDirection() const;

    Container<Tensorholder<DTYPE> *>* GetTrainableTensor();

    int                               GetTrainableTensorDegree() const;

    int                               ResetParameterGradient();
};

#endif  // OPTIMIZER_H_
