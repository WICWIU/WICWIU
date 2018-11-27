#ifndef ADAMOPTIMIZER_H_
#define ADAMOPTIMIZER_H_    value

#include "../Optimizer.hpp"

template<typename DTYPE> class AdamOptimizer : public Optimizer<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_ppParameter;
    Container<Tensor<DTYPE> *> *m_aaFirstVelocity;
    Container<Tensor<DTYPE> *> *m_aaFirstMomentum;
    Container<Tensor<DTYPE> *> *m_aaUnbiasedVelocity;
    Container<Tensor<DTYPE> *> *m_aaUnbiasedMomentum;

    int m_numOfParameter;

    float m_Beta1;
    float m_Beta2;

    float m_epsilon;

public:

    AdamOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float Beta1, float Beta2, float epsilon, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "AdamOptimizer::AdamOptimizer(LossFunction<DTYPE> *, float, float, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter        = NULL;
        m_aaFirstVelocity    = NULL;
        m_aaFirstMomentum    = NULL;
        m_aaUnbiasedVelocity = NULL;
        m_aaUnbiasedMomentum = NULL;
        m_Beta1              = 0.f;
        m_Beta2              = 0.f;
        m_numOfParameter     = 0;
        m_epsilon            = 0.f;

        Alloc(Beta1, Beta2, epsilon);
    }

    AdamOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float Beta1, float Beta2, float epsilon, float weightDecayRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, weightDecayRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "AdamOptimizer::AdamOptimizer(LossFunction<DTYPE> *, float, float, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter        = NULL;
        m_aaFirstVelocity    = NULL;
        m_aaFirstMomentum    = NULL;
        m_aaUnbiasedVelocity = NULL;
        m_aaUnbiasedMomentum = NULL;
        m_Beta1              = 0.f;
        m_Beta2              = 0.f;
        m_numOfParameter     = 0;
        m_epsilon            = 0.f;

        Alloc(Beta1, Beta2, epsilon);
    }

    ~AdamOptimizer() {
        #ifdef __DEBUG__
        std::cout << "AdamOptimizer::~AdamOptimizer()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    virtual void Delete() {
        if (m_aaFirstVelocity) {
            delete m_aaFirstVelocity;
            m_aaFirstVelocity = NULL;
        }

        if (m_aaFirstMomentum) {
            delete m_aaFirstMomentum;
            m_aaFirstMomentum = NULL;
        }

        if (m_aaUnbiasedVelocity) {
            delete m_aaUnbiasedVelocity;
            m_aaUnbiasedVelocity = NULL;
        }

        if (m_aaUnbiasedMomentum) {
            delete m_aaUnbiasedMomentum;
            m_aaUnbiasedMomentum = NULL;
        }
    }

    int Alloc(float Beta1, float Beta2, float epsilon) {

        m_aaFirstVelocity    = new Container<Tensor<DTYPE> *>();
        m_aaUnbiasedVelocity = new Container<Tensor<DTYPE> *>();
        m_aaFirstMomentum    = new Container<Tensor<DTYPE> *>();
        m_aaUnbiasedMomentum = new Container<Tensor<DTYPE> *>();

        Shape *pParameterGradShape = NULL;

        m_ppParameter    = this->GetTrainableTensor();
        m_numOfParameter = this->GetTrainableTensorDegree();

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();

            m_aaFirstVelocity->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_aaUnbiasedVelocity->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_aaFirstMomentum->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_aaUnbiasedMomentum->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));

            pParameterGradShape = NULL;
        }

        m_Beta1   = Beta1;
        m_Beta2   = Beta2;
        m_epsilon = epsilon;

        return TRUE;
    }

    virtual int UpdateParameter() {
      if( m_Beta1 != 0.f ) {
          for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameter((*m_ppParameter)[i], (*m_aaFirstMomentum)[i], (*m_aaFirstVelocity)[i], (*m_aaUnbiasedMomentum)[i], (*m_aaUnbiasedVelocity)[i]);
          }
      }else{
            std::cout << "Don't execute UpdateParameter On CPU" << '\n';
      }
        return TRUE;
    }

    int UpdateParameter(Operator<DTYPE> *pParameter){
      return TRUE;
    }

    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pFirstMomentum, Tensor<DTYPE> *pFirstVelocity, Tensor<DTYPE> *pUnbiasedMomentum, Tensor<DTYPE> *pUnbiasedVelocity) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*pFirstMomentum)[i]    = (m_Beta1 * (*pFirstMomentum)[i]) + ((1.f - m_Beta1) * (*gradient)[i]);
            (*pFirstVelocity)[i]    = (m_Beta2 * (*pFirstVelocity)[i]) + ((1.f - m_Beta2) * ((*gradient)[i] * (*gradient)[i]));
            (*pUnbiasedMomentum)[i] = (*pFirstMomentum)[i] / (1.f - m_Beta1);
            (*pUnbiasedVelocity)[i] = (*pFirstVelocity)[i] / (1.f - m_Beta2);
            (*trainable_data)[i] += ((signed_learning_rate * (*pUnbiasedMomentum)[i]) / (std::sqrt((*pUnbiasedVelocity)[i]) + m_epsilon));
        }

        return TRUE;
    }

#ifdef __CUDNN__

    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        if (m_Beta1 != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                (*m_aaFirstMomentum)[i]->SetDeviceGPU(idOfDevice);
                (*m_aaFirstVelocity)[i]->SetDeviceGPU(idOfDevice);
                (*m_aaUnbiasedMomentum)[i]->SetDeviceGPU(idOfDevice);
                (*m_aaUnbiasedVelocity)[i]->SetDeviceGPU(idOfDevice);

            }
        }
    }

    virtual int UpdateParameterOnGPU() {
          if (m_Beta1 != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaFirstMomentum)[i], (*m_aaFirstVelocity)[i], (*m_aaUnbiasedMomentum)[i], (*m_aaUnbiasedVelocity)[i]);
            }
          }else{
            std::cout << "Don't execute UpdataParameterOnGPU" << '\n';
          }

        return TRUE;
    }

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter){
      return TRUE;
    }

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pFirstMomentum, Tensor<DTYPE> *pFirstVelocity, Tensor<DTYPE> *pUnbiasedMomentum, Tensor<DTYPE> *pUnbiasedVelocity);


 #endif  // if __CUDNN__
};


#endif  // ADAMOPTIMIZER_H_
