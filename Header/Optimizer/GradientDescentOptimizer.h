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
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(Objective<DTYPE> *, float, OptimizeDirection)" << '\n';
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc();
    }

    GradientDescentOptimizer(Container<Tensorholder<DTYPE> *> *pParameterContainer, float pLearningRate, float momentum, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(Objective<DTYPE> *, float, OptimizeDirection)" << '\n';
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc(momentum);
    }

    ~GradientDescentOptimizer() {
        std::cout << "GradientDescentOptimizer::~GradientDescentOptimizer()" << '\n';
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

        int batchsize = gradient->GetBatchSize();

        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*trainable_data)[i] += learning_rate * (*gradient)[i] / batchsize;
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
};


#endif  // GRADIENTDESCENTOPTIMIZER_H_
