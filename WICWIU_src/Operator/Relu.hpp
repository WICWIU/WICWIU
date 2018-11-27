#ifndef RELU_H_
#define RELU_H_    value

#include "../Operator.hpp"

template<typename DTYPE> class Relu : public Operator<DTYPE>{
private:
#ifdef __CUDNN__
    cudnnTensorDescriptor_t m_aInputTensorDesc, m_aOutputTensorDesc, m_aDeltaDesc, m_aInputDeltaDesc;
    cudnnActivationDescriptor_t actDesc;
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;

    float m_alpha;
    float m_beta;
    double m_coef;

#endif  // __CUDNN__

public:
    Relu(Operator<DTYPE> *pInput) : Operator<DTYPE>(pInput) {
        #ifdef __DEBUG__
        std::cout << "Relu::Relu(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput);
    }

    Relu(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "Relu::Relu(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput);
    }

    ~Relu() {
        #ifdef __DEBUG__
        std::cout << "Relu::~Relu()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput) {
        #ifdef __DEBUG__
        std::cout << "Relu::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        Operator<DTYPE> *pInput = this->GetInput()[0];

        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        int inputCapacity  = pInput->GetResult()->GetCapacity();
        int outputCapacity = this->GetResult()->GetCapacity();

        m_alpha = 1.f;
        m_beta  = 0.f;
        m_coef  = 0.0;

        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aOutputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputDeltaDesc));
        checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));

        checkCUDNN(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, m_coef));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aOutputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));
    }

#endif  // if __CUDNN__

    void Delete() {
#ifdef __CUDNN__

        if (m_aInputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputTensorDesc));
        m_aInputTensorDesc = NULL;

        if (m_aOutputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aOutputTensorDesc));
        m_aOutputTensorDesc = NULL;

        if (m_aDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aDeltaDesc));
        m_aDeltaDesc = NULL;

        if (m_aInputDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputDeltaDesc));
        m_aInputDeltaDesc = NULL;

        if (actDesc) checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
        actDesc = NULL;

        // checkCudaErrors(cudaDeviceSynchronize());
#endif  // if __CUDNN__
    }

    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

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
                        (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                            = this->MAX((*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)], 0.f);
                    }
                }
            }
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

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
                        if ((*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)] > 0.0) {
                            (*input_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                                += (*this_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)];
                        } else {
                            (*input_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)] += 0;
                        }
                    }
                }
            }
        }


        return TRUE;
    }

    inline DTYPE MAX(DTYPE data1, DTYPE data2) {
        if (data1 >= data2) return data1;
        else return data2;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnActivationForward(this->GetCudnnHandle(), actDesc, &m_alpha,
                                          m_aInputTensorDesc, m_pDevInput, &m_beta,
                                          m_aOutputTensorDesc, m_pDevOutput));

        // checkCudaErrors(cudaDeviceSynchronize());
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        m_pDevInput      = input->GetGPUData(pTime);
        m_pDevOutput     = result->GetGPUData(pTime);
        m_pDevDelta      = this_delta->GetGPUData(pTime);
        m_pDevInputDelta = input_delta->GetGPUData(pTime);

        checkCUDNN(cudnnActivationBackward(this->GetCudnnHandle(), actDesc, &m_alpha,
                                           m_aOutputTensorDesc, m_pDevOutput,
                                           m_aDeltaDesc, m_pDevDelta,
                                           m_aInputTensorDesc, m_pDevInput, &m_alpha,
                                           m_aInputTensorDesc, m_pDevInputDelta));

        // checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // RELU_H_
