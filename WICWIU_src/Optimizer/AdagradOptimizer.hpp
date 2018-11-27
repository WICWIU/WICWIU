#ifndef ADAGRADOPTIMIZER_H_
#define ADAGRADOPTIMIZER_H_   value

#include "../Optimizer.hpp"


template<typename DTYPE> class AdagradOptimizer : public Optimizer<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_ppParameter;
    Container<Tensor<DTYPE> *> *m_aaGradientSquared;
    Tensor<DTYPE> *shape;

    int m_numOfParameter;
    float m_epsilon;


public:
    AdagradOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "AdagradOptimizer::AdagradOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter          = NULL;
        m_aaGradientSquared      = NULL;

        m_numOfParameter       = 0;
        m_epsilon              = 0.f;

        Alloc();
    }

    AdagradOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float epsilon, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "AdagradOptimizer::AdagradOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter          = NULL;
        m_aaGradientSquared      = NULL;

        m_numOfParameter       = 0;
        m_epsilon              = 0.f;

        Alloc(epsilon);
    }

    ~AdagradOptimizer() {
        #ifdef __DEBUG__
        std::cout << "AdagradOptimizer::~AdagradOptimizer()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    void Delete(){
      if (m_aaGradientSquared) {
          delete m_aaGradientSquared;
          m_aaGradientSquared = NULL;
      }
    }

    int Alloc() {
      m_ppParameter    = this->GetTrainableTensor();
      m_numOfParameter = this->GetTrainableTensorDegree();

      return TRUE;
    }

    int Alloc(float epsilon){

      Alloc();

      m_aaGradientSquared = new Container<Tensor<DTYPE> *>();

      Shape *pParameterGradShape = NULL;

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();
            m_aaGradientSquared->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            pParameterGradShape = NULL;
        }

        m_epsilon = epsilon;

        return TRUE;
    }

    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        if (m_epsilon != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                (*m_aaGradientSquared)[i]->SetDeviceGPU(idOfDevice);
            }
        }
    }

    virtual int UpdateParameter() {

      if(m_epsilon == 0.f){
        for (int i = 0; i < m_numOfParameter; i++) {
            UpdateParameter((*m_ppParameter)[i]);
        }
      }else{
        for (int i = 0; i < m_numOfParameter; i++) {
              UpdateParameter((*m_ppParameter)[i], (*m_aaGradientSquared)[i]);
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

    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pGradientSquared) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*m_pGradientSquared)[i] = ((*gradient)[i] * (*gradient)[i]);
            (*trainable_data)[i]    += (signed_learning_rate * (*gradient)[i]) / std::sqrt((*m_pGradientSquared)[i] + m_epsilon);
        }

        return TRUE;
    }


#ifdef __CUDNN__

    virtual int UpdateParameterOnGPU() {
      if (m_epsilon == 0.f) {
          for (int i = 0; i < m_numOfParameter; i++) {
              UpdateParameterOnGPU((*m_ppParameter)[i]);
          }
      } else {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaGradientSquared)[i]);
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


    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pGradientSquared) {
        // Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        // Tensor<DTYPE> *gradient       = pParameter->GetGradient();
        //
        // Shape *shapeOfParam          = pGradientSquared->GetShape();
        // shape = Tensor<DTYPE>::Constants(shapeOfParam, 1.f);
        //
        // cudnnTensorDescriptor_t dataDesc = trainable_data->GetDescriptor();
        // cudnnTensorDescriptor_t gradDesc = gradient->GetDescriptor();
        // cudnnTensorDescriptor_t gradsquaredDesc = pGradientSquared->GetDescriptor();
        // cudnnTensorDescriptor_t shapeDesc = shape->GetDescriptor();
        //
        // DTYPE *m_pDevData = trainable_data->GetGPUData();
        // DTYPE *m_pDevGrad = gradient->GetGPUData();
        // DTYPE *m_pDevGradSquared = pGradientSquared->GetGPUData();
        // DTYPE *m_pShape = shape->GetGPUData();
        //
        // float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();
        //
        // float alpha  = 1.f;
        // float beta   = 0.f;
        // float slr    = signed_learning_rate;
        // float eps    = m_epsilon;
        //
        // cudnnOpTensorDescriptor_t opTensorDescMul;
        // checkCUDNN(cudnnCreateOpTensorDescriptor(&opTensorDescMul));
        // checkCUDNN(cudnnSetOpTensorDescriptor(opTensorDescMul, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
        //
        // //(*m_pGradientSquared)[i]       = ((*gradient)[i] * (*gradient)[i]);
        // checkCUDNN(cudnnOpTensor(this->GetCudnnHandle(), opTensorDescMul,
        //                           &alpha, gradDesc, m_pDevGrad,
        //                           &alpha, gradDesc, m_pDevGrad,
        //                           &beta, gradsquaredDesc, m_pDevGradSquared));
        // //m_pDevData =                        +alpha * m_pDevData
        // // (*trainable_data)[i]   +=
        // //((signed_learning_rate * (*gradient)[i])
        // // / std::sqrt((*m_pGradientSquared)[i] + m_epsilon));
        // checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
        //                           &eps, shapeDesc, m_pShape,
        //                           &alpha, gradsquaredDesc, m_pDevGradSquared));

        // int batch       = m_pGradientSquared->GetBatchSize();
        // int channel     = m_pGradientSquared->GetChannelSize();
        // int row         = m_pGradientSquared->GetRowSize();
        // int col         = m_pGradientSquared->GetColSize();
        //
        //
        // int capacity = batch * channel * row * col;
        // rsquare<<<batch,capacity>>>(m_pDevGradSquared);

        //
        // checkCUDNN(cudnnOpTensor(this->GetCudnnHandle(),opTensorDescMul,
        //                           &slr, gradDesc, m_pDevGrad,
        //                           &alpha, gradsquaredDesc, m_pDevGradSquared,
        //                           &beta, gradsquaredDesc,m_pDevGradSquared ));

        // checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
        //                           &alpha, gradsquaredDesc, m_pDevGradSquared,
        //                           &alpha, dataDesc, m_pDevData));


        return TRUE;
    }

   //  __global__ void rsquare(DTYPE *myArrayGPU) {
   //   myArrayGPU[capacity] = pow((float) capacity, -0.5);
   // }

#endif  // if __CUDNN__
  };


#endif  // ADAGRADOPTIMIZER_H_
