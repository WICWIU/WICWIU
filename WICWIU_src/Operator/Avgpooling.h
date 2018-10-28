#ifndef __AVGPOOLING__
#define __AVGPOOLING__    value

#include "../Operator.h"

template<typename DTYPE> class GlobalAvaragePooling2D : public Operator<DTYPE>{
private:
    int m_timesize;
    int m_batchsize;
    int m_channelsize;
    int m_rowsize;
    int m_colsize;

    int m_divisor;

#ifdef __CUDNN__
    cudnnTensorDescriptor_t m_aInputTensorDesc, m_aOutputTensorDesc, m_aDeltaDesc, m_aInputDeltaDesc;
    cudnnPoolingDescriptor_t m_aPoolingDesc;
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;
    // DTYPE *m_aHostInput, *m_aHostOutput, *m_aHostInputDelta, *m_aHostDelta;

    float m_alpha;
    float m_beta;
    double m_coef;
#endif  // __CUDNN__

public:
    GlobalAvaragePooling2D(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "GlobalAvaragePooling2D::GlobalAvaragePooling2D(Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        Alloc(pInput);
    }

    virtual ~GlobalAvaragePooling2D() {}

    int Alloc(Operator<DTYPE> *pInput) {
        Shape *pInputTenShape = pInput->GetResult()->GetShape();

        m_timesize    = (*pInputTenShape)[0];
        m_batchsize   = (*pInputTenShape)[1];
        m_channelsize = (*pInputTenShape)[2];
        m_rowsize     = (*pInputTenShape)[3];
        m_colsize     = (*pInputTenShape)[4];

        m_divisor = m_rowsize * m_colsize;

        this->AddResult(new Tensor<DTYPE>(new Shape(m_timesize, m_batchsize, m_channelsize, 1, 1)));
        this->AddGradient(new Tensor<DTYPE>(new Shape(m_timesize, m_batchsize, m_channelsize, 1, 1)));

        return TRUE;
    }

#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        m_alpha = 1.f;
        m_beta  = 0.f;

        checkCUDNN(cudnnCreatePoolingDescriptor(&m_aPoolingDesc));

        checkCUDNN(cudnnSetPooling2dDescriptor(m_aPoolingDesc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_PROPAGATE_NAN,
                                               m_rowsize, m_colsize,  // mask
                                               0, 0, // padding
                                               m_rowsize, m_colsize // stride
                                             ));

        checkCUDNN(cudnnGetPooling2dForwardOutputDim(m_aPoolingDesc, input->GetDescriptor(),
                                                     &batchsize, &channelsize, &rowsize, &colsize));
    }

#endif  // if __CUDNN__

    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < m_batchsize; ba++) {
            for (int ch = 0; ch < m_channelsize; ch++) {
                for (int ro = 0; ro < m_rowsize; ro++) {
                    for (int co = 0; co < m_colsize; co++) {
                        (*result)[Index5D(resultTenShape, ti, ba, ch, 0, 0)]
                            += (*input)[Index5D(inputTenShape, ti, ba, ch, ro, co)];
                    }
                }
                (*result)[Index5D(resultTenShape, ti, ba, ch, 0, 0)] /= m_divisor;
            }
        }


        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner         = this->GetInputContainer();
        Container<Tensor<DTYPE> *>   *input_gradient_container = (*input_contatiner)[0]->GetGradientContainer();
        Container<Tensor<DTYPE> *>   *this_gradient_container  = this->GetGradientContainer();

        Tensor<DTYPE> *this_grad  = (*this_gradient_container)[0];
        Tensor<DTYPE> *input_grad = (*input_gradient_container)[0];

        Shape *resultTenShape = this_grad->GetShape();
        Shape *inputTenShape  = input_grad->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < m_batchsize; ba++) {
            for (int ch = 0; ch < m_channelsize; ch++) {
                for (int ro = 0; ro < m_rowsize; ro++) {
                    for (int co = 0; co < m_colsize; co++) {
                        (*input_grad)[Index5D(inputTenShape, ti, ba, ch, ro, co)]
                            += (*this_grad)[Index5D(resultTenShape, ti, ba, ch, 0, 0)] / m_divisor;
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

          m_pDevInput         = input->GetGPUData(pTime);
          m_aInputTensorDesc  = input->GetDescriptor();
          m_pDevOutput        = result->GetGPUData(pTime);
          m_aOutputTensorDesc = result->GetDescriptor();

          checkCUDNN(cudnnPoolingForward(this->GetCudnnHandle(), m_aPoolingDesc,
                                        &m_alpha, m_aInputTensorDesc, m_pDevInput,
                                        &m_beta, m_aOutputTensorDesc, m_pDevOutput));


          checkCudaErrors(cudaDeviceSynchronize());
          // this->ForwardPropagate(pTime);
          return TRUE;
      }

      int BackPropagateOnGPU(int pTime = 0) {
          Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
          Tensor<DTYPE> *this_delta  = this->GetDelta();
          Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
          Tensor<DTYPE> *result      = this->GetResult();

          m_pDevInput         = input->GetGPUData(pTime);
          m_aInputTensorDesc  = input->GetDescriptor();
          m_pDevOutput        = result->GetGPUData(pTime);
          m_aOutputTensorDesc = result->GetDescriptor();
          m_pDevDelta         = this_delta->GetGPUData(pTime);
          m_aDeltaDesc        = this_delta->GetDescriptor();
          m_pDevInputDelta    = input_delta->GetGPUData(pTime);
          m_aInputDeltaDesc   = input_delta->GetDescriptor();

          checkCUDNN(cudnnPoolingBackward(this->GetCudnnHandle(), m_aPoolingDesc,
                                          &m_alpha, m_aOutputTensorDesc, m_pDevOutput,
                                          m_aDeltaDesc, m_pDevDelta, m_aInputTensorDesc, m_pDevInput,
                                          &m_alpha, m_aInputDeltaDesc, m_pDevInputDelta));


          checkCudaErrors(cudaDeviceSynchronize());
          // this->BackPropagate(pTime);
          return TRUE;
      }

#endif  // if __CUDNN__
};
//
#endif  // __AVGPOOLING__
