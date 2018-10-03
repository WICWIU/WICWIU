#ifndef DROPOUT_H_
#define DROPOUT_H_    value

#include "../Operator.h"

template<typename DTYPE> class Dropout : public Operator<DTYPE>{
private:
  float m_droprate;
  Mode  m_mode;
  Tensor<unsigned char> *m_activeshape; //1D Tensor, type: unsigned char

#ifdef __CUDNN__
    cudnnTensorDescriptor_t m_aInOutDesc, m_aDeltaDesc, m_aInputDeltaDesc;

    cudnnDropoutDescriptor_t m_aDropoutDesc;

    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta, *m_pRandNumGenerator, *m_pReserveSpace;

    size_t m_dataSizeInBytes   = 0;
    size_t m_spaceSizeInBytes  = 0;

#endif  // __CUDNN__

public:
    Dropout(Operator<DTYPE> *pInput) : Operator<DTYPE>(pInput) {
        #ifdef __DEBUG__
        std::cout << "Dropout::Dropout(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
    }
    

    Dropout(Operator<DTYPE> *pInput, float pDroprate, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "Dropout::Dropout(Operator<DTYPE> * float *)" << '\n';
        #endif  // __DEBUG__
        m_droprate = 0.f;
        m_mode = TRAIN;

        this->Alloc(pInput, pDroprate);
    }

    ~Dropout() {
        #ifdef __DEBUG__
        std::cout << "Dropout::~Dropout()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, float pDroprate) {
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

        m_activeshape = new Tensor<unsigned char>(batchsize);
        m_droprate = pDroprate;
        m_mode = TRAIN;

        return TRUE;
    }


#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        Operator<DTYPE> *pInput = this->GetInput()[0];

        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_mode = TRAIN;

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
    Tensor<unsigned char> *activeshape   = m_activeshape;

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    float randNum = 0.f;

    Shape *resultTenShape = result->GetShape();

    int ti = pTime;

      switch(m_mode){

        case TRAIN:
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                          randNum = rand()/(float)RAND_MAX;
                          if(randNum > m_droprate){
                          (*activeshape)[ba] = (1.f / (1.f - m_droprate));
                          (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                          = (*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)] * (*activeshape)[ba]; //scale up

                          }
                          else
                          (*activeshape)[ba] = 0.f;
                          (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                          = (*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)] * (*activeshape)[ba];

                      }
                    }
                  }
                  break;
          }

        case INFERENCE:

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
        Tensor<unsigned char> *activeshape         = m_activeshape;

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
                              += (*this_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)] * (*activeshape)[ba];
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

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        switch(m_mode){
        case TRAIN:

        checkCUDNN(cudnnDropoutForward(this->GetCudnnHandle(),
                                      m_aDropoutDesc,
                                      m_aInOutDesc,
                                      m_pDevInput,
                                      m_aInOutDesc,
                                      m_pDevOutput,
                                      m_pReserveSpace,
                                      m_spaceSizeInBytes));


        break;
        case INFERENCE:

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &alpha,
                                  m_aInOutDesc,
                                  m_pDevInput,
                                  &alpha,
                                  m_aInOutDesc,
                                  m_pDevOutput));
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

        checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

    int SetModeTrain() {
        m_mode = TRAIN;
        return TRUE;
      }

    int SetModeInference() {
        m_mode = INFERENCE;
        return TRUE;
      }

#endif  // if __CUDNN__
};

#endif  // DROPOUT_H_
