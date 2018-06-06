#ifndef GRADIENTDESCENTOPTIMIZER_H_
#define GRADIENTDESCENTOPTIMIZER_H_    value

#include "../Optimizer.h"

template<typename DTYPE> class GradientDescentOptimizer : public Optimizer<DTYPE>{
private:
    Container<Tensorholder<DTYPE> *> *m_ppParameter;
    Container<Tensor<DTYPE> *> *m_aaVelocity;

    int m_numOfParameter;
    float m_momentum;

public:
    GradientDescentOptimizer(Container<Tensorholder<DTYPE> *> *pParameterContainer, float pLearningRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #if __DEBUG__
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc();
    }

    GradientDescentOptimizer(Container<Tensorholder<DTYPE> *> *pParameterContainer, float pLearningRate, float momentum, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #if __DEBUG__
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc(momentum);
    }

    ~GradientDescentOptimizer() {
        #if __DEBUG__
        std::cout << "GradientDescentOptimizer::~GradientDescentOptimizer()" << '\n';
        #endif  // __DEBUG__
    }

    int Alloc() {
        m_ppParameter    = this->GetTrainableTensor();
        m_numOfParameter = this->GetTrainableTensorDegree();

        return TRUE;
    }

    int Alloc(float momentum) {
        Alloc();
        m_aaVelocity = new Container<Tensor<DTYPE> *>();

        Shape *pParameterGradShape = NULL;

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();

            // std::cout << (*m_ppParameter)[i]->GetName() << '\n';
            // std::cout << pParameterGradShape << '\n';

            m_aaVelocity->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            pParameterGradShape = NULL;
        }

        m_momentum = momentum;

        return TRUE;
    }

    virtual int UpdateVariable() {
        if (m_momentum == 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateVariable((*m_ppParameter)[i]);
            }
        } else {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateVariable((*m_ppParameter)[i], (*m_aaVelocity)[i]);
            }
        }

        return TRUE;
    }

    int UpdateVariable(Tensorholder<DTYPE> *pParameter) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*trainable_data)[i] += learning_rate * (*gradient)[i];
        }

        return TRUE;
    }

    int UpdateVariable(Tensorholder<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*pVelocity)[i]       = m_momentum * (*pVelocity)[i] + learning_rate * (*gradient)[i];
            (*trainable_data)[i] += (*pVelocity)[i];
        }

        return TRUE;
    }

#if __CUDNN__


    virtual int UpdateVariableOnGPU() {
        if (m_momentum == 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateVariableOnGPU((*m_ppParameter)[i]);
            }
        } else {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateVariableOnGPU((*m_ppParameter)[i], (*m_aaVelocity)[i]);
            }
        }

        return TRUE;
    }

    int UpdateVariableOnGPU(Tensorholder<DTYPE> *pParameter) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        cudnnTensorDescriptor_t dataDesc = trainable_data->GetDescriptor();
        cudnnTensorDescriptor_t gradDesc = gradient->GetDescriptor();

        DTYPE *m_pDevData = trainable_data->GetDeviceData();
        DTYPE *m_pDevGrad = gradient->GetDeviceData();

        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        float alpha = 1.f;
        float beta  = learning_rate;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &beta, gradDesc, m_pDevGrad,
                                  &alpha, dataDesc, m_pDevData));

        return TRUE;
    }

    int UpdateVariableOnGPU(Tensorholder<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        cudnnTensorDescriptor_t dataDesc = trainable_data->GetDescriptor();
        cudnnTensorDescriptor_t gradDesc = gradient->GetDescriptor();
        cudnnTensorDescriptor_t veloDesc = pVelocity->GetDescriptor();

        DTYPE *m_pDevData = trainable_data->GetDeviceData();
        DTYPE *m_pDevGrad = gradient->GetDeviceData();
        DTYPE *m_pDevVelo = pVelocity->GetDeviceData();

        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        float alpha = 1.f;
        float beta  = learning_rate;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &beta, gradDesc, m_pDevGrad,
                                  &m_momentum, veloDesc, m_pDevVelo));

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &alpha, veloDesc, m_pDevVelo,
                                  &alpha, dataDesc, m_pDevData));

        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // GRADIENTDESCENTOPTIMIZER_H_
