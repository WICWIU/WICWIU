#ifndef NAGOPTIMIZER_H_
#define NAGOPTIMIZER_H_    value

#include "../Optimizer.h"

template<typename DTYPE> class NagOptimizer : public Optimizer<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_ppParameter;
    Container<Tensor<DTYPE> *> *m_aaVelocity;

    int m_numOfParameter;
    float m_momentum;

public:

    NagOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float momentum, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "NagOptimizer::NagOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc(momentum);
    }

    NagOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float momentum, float weightDecayRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, weightDecayRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "NagOptimizer::NagOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc(momentum);
    }

    ~NagOptimizer() {
        #ifdef __DEBUG__
        std::cout << "NagOptimizer::~NagOptimizer()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    virtual void Delete(){
      if (m_aaVelocity) {
          delete m_aaVelocity;
          m_aaVelocity = NULL;
      }
    }

    int Alloc(float momentum) {
        m_ppParameter    = this->GetTrainableTensor();
        m_numOfParameter = this->GetTrainableTensorDegree();
        m_aaVelocity = new Container<Tensor<DTYPE> *>();

        Shape *pParameterGradShape = NULL;

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();

            m_aaVelocity->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));

            pParameterGradShape = NULL;
        }

        m_momentum = momentum;

        return TRUE;
    }

    virtual int UpdateParameter() {
        if (m_momentum != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                  UpdateParameter((*m_ppParameter)[i], (*m_aaVelocity)[i]);
            }
        }else{
            std::cout << "Don't execute UpdateParameter On CPU" << '\n';
            }

        return TRUE;
    }

    int UpdateParameter(Operator<DTYPE> *pParameter){
      return TRUE;
    }

    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();
        Tensor<DTYPE> *prev_Velocity  = pVelocity;

        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*prev_Velocity)[i] = (*pVelocity)[i];
            (*pVelocity)[i]       = (m_momentum * (*pVelocity)[i]) + (learning_rate * (*gradient)[i]);
            (*trainable_data)[i] +=  -m_momentum * (*prev_Velocity)[i] + ((1.f + m_momentum) * (*pVelocity)[i]);
        }

        return TRUE;
    }

#ifdef __CUDNN__

    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        if (m_momentum != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                (*m_aaVelocity)[i]->SetDeviceGPU(idOfDevice);
            }
        }
    }

    virtual int UpdateParameterOnGPU() {
        if (m_momentum != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaVelocity)[i]);
            }
        }else{
            std::cout << "Don't execute UpdataParameterOnGPU" << '\n';
            }

        return TRUE;
    }


    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter){
      return TRUE;
    }

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity);


#endif  // if __CUDNN__
};


#endif  // GRADIENTDESCENTOPTIMIZER_H_
