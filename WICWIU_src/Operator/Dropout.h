#ifndef DROPOUT_H_
#define DROPOUT_H_    value

#include "../Operator.h"

template<typename DTYPE> class Dropout : public Operator<DTYPE>{
private:
  float m_keeprate;
  Mode  m_mode;
  Tensor<unsigned char> *m_noiseshape; //1D Tensor, type: unsigned char

#ifdef __CUDNN__
    cudnnTensorDescriptor_t m_aInOutDesc, m_aDeltaDesc, m_aInputDeltaDesc;

    cudnnDropoutDescriptor_t m_aDropoutDesc;

    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta, *m_pRandNumGenerator, *m_pReserveSpace;

    float m_droprate;


    size_t m_dataSizeInBytes   = 0;
    size_t m_spaceSizeInBytes  = 0;

#endif  // __CUDNN__

public:
    Dropout(Operator<DTYPE> *pInput) : Operator<DTYPE>(pInput) {
        #ifdef __DEBUG__
        std::cout << "Dropout::Dropout(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput);
    }

    Dropout(Operator<DTYPE> *pInput, float pKeeprate, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "Dropout::Dropout(Operator<DTYPE> * float *)" << '\n';
        #endif  // __DEBUG__
        m_keeprate = 0.f;
        m_mode = TRAINING;

        this->Alloc(pInput, pKeeprate);
    }

    ~Dropout() {
        #ifdef __DEBUG__
        std::cout << "Dropout::~Dropout()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, float pKeeprate) {
        #ifdef __DEBUG__
        std::cout << "Dropout::Alloc(Operator<DTYPE> *, float *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(Tensor<DTYPE>::Zeros(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetDelta(Tensor<DTYPE>::Zeros(timesize, batchsize, channelsize, rowsize, colsize));

        m_noiseshape = new Tensor<unsigned char>(batchsize);
        m_keeprate = pKeeprate;
        m_mode = TRAINING;

        return TRUE;
    }


#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        Operator<DTYPE> *pInput = this->GetInput()[0];

        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_droprate = 1.f - m_keeprate;
        m_mode = TRAINING;

        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInOutDesc));

        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputDeltaDesc));

        checkCUDNN(cudnnCreateDropoutDescriptor(&m_aDropoutDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInOutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));
        checkCUDNN(cudnnSetTensor4dDescriptor(m_aDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));
        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnDropoutGetStatesSize(this->GetCudnnHandle(), &m_dataSizeInBytes));

        checkCUDNN(cudnnDropoutGetReserveSpaceSize(m_aInOutDesc, &m_spaceSizeInBytes));

        checkCudaErrors(cudaMalloc((void **)&m_pRandNumGenerator, m_dataSizeInBytes));
        checkCudaErrors(cudaMalloc((void **)&m_pReserveSpace, m_spaceSizeInBytes));

        checkCUDNN(cudnnSetDropoutDescriptor(m_aDropoutDesc, this->GetCudnnHandle(),
                                             m_droprate, m_pRandNumGenerator, m_dataSizeInBytes, time(NULL)));


    }

#endif  // if __CUDNN__

  void Delete() {
#ifdef __CUDNN__

        if (m_aInOutDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aInOutDesc));
        m_aInOutDesc = NULL;

        if (m_aDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aDeltaDesc));
        m_aDeltaDesc = NULL;

        if (m_aInputDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputDeltaDesc));
        m_aInputDeltaDesc = NULL;

        if (m_pRandNumGenerator) checkCudaErrors(cudaFree(m_pRandNumGenerator));
        m_pRandNumGenerator = NULL;

        if (m_pReserveSpace) checkCudaErrors(cudaFree(m_pReserveSpace));
        m_pReserveSpace = NULL;


        checkCudaErrors(cudaThreadSynchronize());
#endif  // if __CUDNN__
    }

int ForwardPropagate(int pTime = 0) {
    Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();
    Tensor<unsigned char> *noiseshape   = m_noiseshape;

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    float randNum = 0.f;

    Shape *resultTenShape = result->GetShape();

    int ti = pTime;

      switch(m_mode){

        case TRAINING:
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                          randNum = rand()/(float)RAND_MAX;
                          if(randNum <= m_keeprate){
                          (*noiseshape)[ba] = (1.f / m_keeprate);
                          (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                          = (*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)] * (*noiseshape)[ba]; //scale up

                          }

                          else
                          (*noiseshape)[ba] = 0.f;
                          (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                          = (*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)] * (*noiseshape)[ba];

                      }
                    }
                  }
                  break;
          }

        case INFERENCING:

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                              (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                              +=(*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)];
                            }
                          }
                        }
                      }
        break;

        default:
        break;
      }
      return TRUE;
}

    int BackPropagate(int pTime = 0) {

        Tensor<DTYPE> *result       = this->GetResult();
        Tensor<DTYPE> *this_delta   = this->GetGradient();
        Tensor<DTYPE> *input_delta  = this->GetInput()[0]->GetDelta();
        Tensor<unsigned char> *noiseshape         = m_noiseshape;

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                            (*input_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                              += (*this_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)] * (*noiseshape)[ba];
                        }
                    }
                }
            }

        return TRUE;
    }


#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        float alpha = 1.f;
        float beta = 0.f;

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        switch(m_mode){
        case TRAINING:

        checkCUDNN(cudnnDropoutForward(this->GetCudnnHandle(),
                                      m_aDropoutDesc,
                                      m_aInOutDesc,
                                      m_pDevInput,
                                      m_aInOutDesc,
                                      m_pDevOutput,
                                      m_pReserveSpace,
                                      m_spaceSizeInBytes));


        break;
        case INFERENCING:

        // checkCUDNN(cudnnTransformTensor(this->GetCudnnHandle(),
        //                                 &alpha,
        //                                 m_aInOutDesc,
        //                                 m_pDevInput,
        //                                 &beta,
        //                                 m_aInOutDesc,
        //                                 m_pDevOutput));
        this->ForwardPropagate(pTime);
        break;
      }

        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {

      float alpha = 1.f;
      float beta = 0.f;

        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        m_pDevInput      = input->GetGPUData(pTime);
        m_pDevOutput     = result->GetGPUData(pTime);
        m_pDevDelta      = this_delta->GetGPUData(pTime);
        m_pDevInputDelta = input_delta->GetGPUData(pTime);


        checkCUDNN(cudnnDropoutBackward(this->GetCudnnHandle(),
                        m_aDropoutDesc,
                        m_aInOutDesc,
                        m_pDevDelta  ,
                        m_aInOutDesc,
                        m_pDevInputDelta,
                        m_pReserveSpace,
                        m_spaceSizeInBytes));

//m_pDevInputDelta
        checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

      int SetModeTraining() {
          m_mode = TRAINING;
          return TRUE;
        }

      int SetModeInferencing() {
          m_mode = INFERENCING;
          return TRUE;
        }

#endif  // if __CUDNN__
};

#endif  // DROPOUT_H_
