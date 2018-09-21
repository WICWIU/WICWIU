#ifndef RMSPROPOPTIMIZER_H_
#define RMSPROPOPTIMIZER_H_   value

/*
============================논문 버젼 ========================================================
reference :
https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
rmsprop algorithm [tieleman2012rmsprop]

explain:
centered version보다는 Accuracy측면에서는 성능이 떨어지나 속도가 빠름.
A detailed description of rmsprop.
- maintain a moving (discounted) average of the square of gradients
- divide gradient by the root of this average

m_aaMeanSqueared = decay * mean_square{t-1} + (1 - decay) * gradient ** 2

trainable_data = momentum * trainable_data{t-1} + learning_rate * g_t / sqrt(m_aaMeanSqueared + epsilon)

delta = - trainable_data

==========================centered version=======================================================
reference :
https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
https://arxiv.org/pdf/1308.0850v5.pdf //논문

notice :
centered version 사용시 반드시 learning rate는 0.0001로 주어야 한다. 논문참조
값이 크면 학습이 잘 안된다.

explain:
 Accuracy 측면에서는 좋으나, 수식이 많아 속도감이 떨어짐.

If True, gradients are normalized by the estimated variance of the gradient;
if False, by the uncentered second moment. Setting this to True may help with training,
but is slightly more expensive in terms of computation and memory. Defaults to False.
This implementation of RMSProp uses plain momentum, not Nesterov momentum.
The centered version additionally maintains a moving (discounted) average of the
gradients, and uses that average to estimate the variance:

mean_grad = decay * mean_square{t-1} + (1-decay) * gradient

m_aaMeanSqueared  = decay * mean_square{t-1} + (1-decay) * gradient ** 2

trainable_data = momentum * mom{t-1} + learning_rate * g_t /  sqrt(mean_square - mean_grad**2 + epsilon)

delta = - mom

==================================================================================
*/

#include "../Optimizer.h"


template<typename DTYPE> class RMSPropOptimizer : public Optimizer<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_ppParameter;
    Container<Tensor<DTYPE> *> *m_aaMeanSquared;  //  = decay * mean_square{t-1} + (1-decay) * gradient ** 2
    Container<Tensor<DTYPE> *> *m_aaMeanGrad;

    int m_numOfParameter;

    float m_decay; //decay = 0.9
    float m_epsilon; //epsilon = 0.001

    bool m_centered;

public:
    RMSPropOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "RMSPropOptimizer::RMSPropOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter       = NULL;
        m_aaMeanSquared     = NULL;
        m_aaMeanGrad        = NULL;
        m_numOfParameter    = 0;
        m_decay             = 0.f;
        m_epsilon           = 0.f;

        Alloc();
    }

    RMSPropOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float decay, float epsilon, bool centered, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "RMSPropOptimizer::RMSPropOptimizer(LossFunction<DTYPE> *, float, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter       = NULL;
        m_aaMeanSquared     = NULL;
        m_aaMeanGrad        = NULL;
        m_numOfParameter    = 0;
        m_decay             = 0.f;
        m_epsilon           = 0.f;
        m_centered          = FALSE;

        Alloc(decay, epsilon, centered);
    }

    ~RMSPropOptimizer() {
        #ifdef __DEBUG__
        std::cout << "RMSPropOptimizer::~RMSPropOptimizer()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    virtual void Delete() {
        if (m_aaMeanSquared) {
            delete m_aaMeanSquared;
            m_aaMeanSquared = NULL;
        }
        if (m_aaMeanGrad) {
            delete m_aaMeanGrad;
            m_aaMeanGrad = NULL;
        }
    }

    int Alloc() {
        m_ppParameter    = this->GetTrainableTensor();
        m_numOfParameter = this->GetTrainableTensorDegree();

        return TRUE;
    }

    int Alloc(float decay, float epsilon, bool centered) {

        Alloc();

        m_aaMeanSquared = new Container<Tensor<DTYPE> *>();
        m_aaMeanGrad    = new Container<Tensor<DTYPE> *>();

        Shape *pParameterGradShape = NULL;

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();
            m_aaMeanSquared->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_aaMeanGrad->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            pParameterGradShape = NULL;
        }

        m_decay = decay;
        m_epsilon = epsilon;
        m_centered = centered;

        return TRUE;
    }

    virtual int UpdateParameter() {
      //std::cout << typeid(m_decay).name() << 0.f << std::endl;
         if(m_decay == 0.f){          //trainable data = learning_rate * g_t / sqrt(m_aaMeanSqueared + epsilon)
             for (int i = 0; i < m_numOfParameter; i++) {
                 UpdateParameter((*m_ppParameter)[i]);
             }
         }
         else if(m_centered == TRUE){
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameter((*m_ppParameter)[i], (*m_aaMeanSquared)[i], (*m_aaMeanGrad)[i]);

            }
          }
        else
        for (int i = 0; i < m_numOfParameter; i++) {
            UpdateParameter((*m_ppParameter)[i], (*m_aaMeanSquared)[i]);
        }
        return TRUE;
    }

    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        if (m_decay != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                (*m_aaMeanSquared)[i]->SetDeviceGPU(idOfDevice);
            }
        }
    }

    int UpdateParameter(Operator<DTYPE> *pParameter) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate(); //minimizer = -1

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*trainable_data)[i] += signed_learning_rate * (*gradient)[i];
        }

        return TRUE;
    }

    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pMeanSquared) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();//minimizer = -1
        int capacity = trainable_data->GetCapacity();


        for (int i = 0; i < capacity; i++) {
            (*m_pMeanSquared)[i]       = (m_decay * (*m_pMeanSquared)[i]) + (( 1.f - m_decay) * ((*gradient)[i] * (*gradient)[i]));
            (*trainable_data)[i]    += ((signed_learning_rate * (*gradient)[i]) / std::sqrt((*m_pMeanSquared)[i] + m_epsilon));
        }

        return TRUE;
    }

    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pMeanSquared, Tensor<DTYPE> *m_pMeanGrad){
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();//minimizer = -1
        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*m_pMeanGrad)[i]     = (m_decay * (*m_pMeanGrad)[i]) + ((1.f - m_decay) * (*gradient)[i]);
            (*m_pMeanSquared)[i]  = (m_decay * (*m_pMeanSquared)[i]) + (( 1.f - m_decay) * ((*gradient)[i] * (*gradient)[i]));
            (*trainable_data)[i] += ((signed_learning_rate * (*gradient)[i]) / std::sqrt(((*m_pMeanSquared)[i] - ((*m_pMeanGrad)[i] * (*m_pMeanGrad)[i]))+ m_epsilon));
        }

        return TRUE;
    }

#ifdef __CUDNN__


    virtual int UpdateParameterOnGPU() {
        if (m_decay == 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i]);
            }
        } else {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaMeanSquared)[i]);
            }
        }

        return TRUE;
    }

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        cudnnTensorDescriptor_t dataDesc = trainable_data->GetDescriptor();
        cudnnTensorDescriptor_t gradDesc = gradient->GetDescriptor();

        DTYPE *m_pDevData = trainable_data->GetGPUData();
        DTYPE *m_pDevGrad = gradient->GetGPUData();

        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        float alpha = 1.f;
        float beta  = learning_rate;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &beta, gradDesc, m_pDevGrad,
                                  &alpha, dataDesc, m_pDevData));

        return TRUE;
    }

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pMeanSquared) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        cudnnTensorDescriptor_t dataDesc = trainable_data->GetDescriptor();
        cudnnTensorDescriptor_t gradDesc = gradient->GetDescriptor();
        cudnnTensorDescriptor_t MeanSquearedDesc = pMeanSquared->GetDescriptor();

        DTYPE *m_pDevData = trainable_data->GetGPUData();
        DTYPE *m_pDevGrad = gradient->GetGPUData();
        DTYPE *m_pMeanSquared = pMeanSquared->GetGPUData();

        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        float alpha = 1.f;
        float beta  = learning_rate;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &beta, gradDesc, m_pDevGrad,
                                  &m_decay, MeanSquearedDesc, m_pMeanSquared));

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &alpha, MeanSquearedDesc, m_pMeanSquared,
                                  &alpha, dataDesc, m_pDevData));

        return TRUE;
    }

#endif  // if __CUDNN__
 };


#endif  // RMSPROPOPTIMIZER_H_
