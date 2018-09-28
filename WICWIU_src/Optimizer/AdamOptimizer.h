#ifndef ADAMOPTIMIZER_H_
#define ADAMOPTIMIZER_H_   value

#include "../Optimizer.h"


template<typename DTYPE> class AdamOptimizer : public Optimizer<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_ppParameter;
    Container<Tensor<DTYPE> *> *m_aaFirstVelocity;
    Container<Tensor<DTYPE> *> *m_aaFirstMomentum;
    Container<Tensor<DTYPE> *> *m_aaUnbiasedVelocity;
    Container<Tensor<DTYPE> *> *m_aaUnbiasedMomentum;
    Container<Tensor<DTYPE> *> *m_rsqrt;


    int m_numOfParameter;

    float m_Beta1;
    float m_Beta2;

    float m_epsilon;

public:
    AdamOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "AdamOptimizer::AdamOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter          = NULL;
        m_aaFirstVelocity      = NULL;
        m_aaFirstMomentum     = NULL;
        m_aaUnbiasedVelocity  = NULL;
        m_rsqrt               =NULL;
        m_Beta1                = 0.f;
        m_Beta2               = 0.f;
        m_numOfParameter     = 0;
        m_epsilon            = 0.f;

        Alloc();
    }

    AdamOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float Beta1, float Beta2, float epsilon, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "AdamOptimizer::AdamOptimizer(LossFunction<DTYPE> *, float, float, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter       = NULL;
        m_aaFirstVelocity   = NULL;
        m_aaFirstMomentum   = NULL;
        m_aaUnbiasedVelocity  = NULL;
        m_aaUnbiasedMomentum  = NULL;
        m_rsqrt=NULL;
        m_Beta1             = 0.f;
        m_Beta2             = 0.f;
        m_numOfParameter    = 0;
        m_epsilon           = 0.f;

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

    int Alloc() {
        m_ppParameter    = this->GetTrainableTensor();
        m_numOfParameter = this->GetTrainableTensorDegree();

        return TRUE;
    }

    int Alloc(float Beta1, float Beta2, float epsilon) {

        Alloc();

        m_aaFirstVelocity   = new Container<Tensor<DTYPE> *>();
        m_aaUnbiasedVelocity  = new Container<Tensor<DTYPE> *>();
        m_aaFirstMomentum   = new Container<Tensor<DTYPE> *>();
        m_aaUnbiasedMomentum  = new Container<Tensor<DTYPE> *>();
        m_rsqrt  = new Container<Tensor<DTYPE> *>();

        Shape *pParameterGradShape = NULL;

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();

            // std::cout << (*m_ppParameter)[i]->GetName() << '\n';
            // std::cout << pParameterGradShape << '\n';

            m_aaFirstVelocity->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_aaUnbiasedVelocity->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_aaFirstMomentum->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_aaUnbiasedMomentum->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_rsqrt->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));

            pParameterGradShape = NULL;
        }

        m_Beta1 = Beta1;
        m_Beta2 = Beta2;
        m_epsilon = epsilon;

        return TRUE;
    }


    virtual int UpdateParameter() {
        if(m_Beta1 == 0.f){
            for (int i = 0; i < m_numOfParameter; i++){
                UpdateParameter((*m_ppParameter)[i]);
           }
        }
        else{
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameter((*m_ppParameter)[i], (*m_aaFirstMomentum)[i], (*m_aaFirstVelocity)[i], (*m_aaUnbiasedMomentum)[i], (*m_aaUnbiasedVelocity)[i], (*m_rsqrt)[i]);
            }
        }
        return TRUE;
    }



    int UpdateParameter(Operator<DTYPE> *pParameter) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();


        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*trainable_data)[i] += learning_rate * (*gradient)[i];
        }

        return TRUE;
    }


    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pFirstMomentum, Tensor<DTYPE> *pFirstVelocity, Tensor<DTYPE> *pUnbiasedMomentum, Tensor<DTYPE> *pUnbiasedVelocity, Tensor<DTYPE> *pRsqrt) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();


        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
          (*pFirstMomentum)[i]  = (m_Beta1 * (*pFirstMomentum)[i]) + ((1.f - m_Beta1) * (*gradient)[i]);
          (*pFirstVelocity)[i]  = (m_Beta2 * (*pFirstVelocity)[i]) + ((1.f - m_Beta2) * ((*gradient)[i] * (*gradient)[i])) ;
          (*pUnbiasedMomentum)[i]  =  (*pFirstMomentum)[i] / ( 1.f - m_Beta1 );
          (*pUnbiasedVelocity)[i]  =  (*pFirstVelocity)[i] / ( 1.f - m_Beta2 );
          // std::cout << "UV:" <<'\n';
          // std::cout <<(*pUnbiasedVelocity)[i]  << '\n';
          (*pRsqrt)[i] = (1.f / (std::sqrt((*pUnbiasedVelocity)[i]) + m_epsilon));
          // std::cout << "rsrt:" <<'\n';
          // std::cout << (*pRsqrt)[i]  << '\n';
          (*trainable_data)[i] += ((signed_learning_rate * (*pUnbiasedMomentum)[i]) * (*pRsqrt)[i]);
         //(*trainable_data)[i] += ((signed_learning_rate * (*pUnbiasedMomentum)[i]) / (std::sqrt((*pUnbiasedVelocity)[i]) + m_epsilon));
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
            (*m_rsqrt)[i]->SetDeviceGPU(idOfDevice);
        }
    }
}

    virtual int UpdateParameterOnGPU() {
        if (m_Beta1 == 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameter((*m_ppParameter)[i]);
            }
        } else {
            for (int i = 0; i < m_numOfParameter; i++) {
              UpdateParameter((*m_ppParameter)[i], (*m_aaFirstMomentum)[i], (*m_aaFirstVelocity)[i], (*m_aaUnbiasedMomentum)[i], (*m_aaUnbiasedVelocity)[i], (*m_rsqrt)[i]);
            }
        }

        return TRUE;
    }

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter) {
        // Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        // Tensor<DTYPE> *gradient       = pParameter->GetGradient();
        //
        // cudnnTensorDescriptor_t dataDesc = trainable_data->GetDescriptor();
        // cudnnTensorDescriptor_t gradDesc = gradient->GetDescriptor();
        //
        // DTYPE *m_pDevData = trainable_data->GetGPUData();
        // DTYPE *m_pDevGrad = gradient->GetGPUData();
        //
        // float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();
        //
        // float alpha = 1.f;
        // float beta  = learning_rate;
        //
        // checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
        //                           &beta, gradDesc, m_pDevGrad,
        //                           &alpha, dataDesc, m_pDevData));

        return TRUE;
     }

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pFirstMomentum, Tensor<DTYPE> *pFirstVelocity, Tensor<DTYPE> *pUnbiasedMomentum, Tensor<DTYPE> *pUnbiasedVelocity, Tensor<DTYPE> *pRsqrt)  {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        cudnnTensorDescriptor_t dataDesc = trainable_data->GetDescriptor();
        cudnnTensorDescriptor_t gradDesc = gradient->GetDescriptor();
        cudnnTensorDescriptor_t firstMDesc = pFirstMomentum->GetDescriptor();
        cudnnTensorDescriptor_t firstVDesc = pFirstVelocity->GetDescriptor();
        cudnnTensorDescriptor_t unbiasMDesc = pUnbiasedMomentum->GetDescriptor();
        cudnnTensorDescriptor_t unbiasVDesc = pUnbiasedVelocity->GetDescriptor();
        cudnnTensorDescriptor_t rsqrtDesc = pRsqrt->GetDescriptor();


        DTYPE *m_pDevData = trainable_data->GetGPUData();
        DTYPE *m_pDevGrad = gradient->GetGPUData();
        DTYPE *m_pDevFirstMomentum = pFirstMomentum->GetGPUData();
        DTYPE *m_pDevFirstVelocity = pFirstVelocity->GetGPUData();
        DTYPE *m_pDevUnbiasedMomentum = pUnbiasedMomentum->GetGPUData();
        DTYPE *m_pDevUnbiasedVelocity = pUnbiasedVelocity->GetGPUData();
        DTYPE *m_pDevRsqrt = pRsqrt->GetGPUData();

        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        float alpha = 1.f;
        float zero = 0.f;

        float beta  = signed_learning_rate;

          std::cout << "/* message */" << '\n';


        cudnnOpTensorDescriptor_t opTensorDescMul;
        checkCUDNN(cudnnCreateOpTensorDescriptor(&opTensorDescMul));
        checkCUDNN(cudnnSetOpTensorDescriptor(opTensorDescMul, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));


        checkCUDNN(cudnnOpTensor(this->GetCudnnHandle(), opTensorDescMul,
                                 &beta, unbiasMDesc, m_pDevUnbiasedMomentum,
                                 &alpha, rsqrtDesc, m_pDevRsqrt,
                                 &alpha, dataDesc, m_pDevData));



        return TRUE;
    }


 #endif  // if __CUDNN__
};


#endif  // ADAMOPTIMIZER_H_
