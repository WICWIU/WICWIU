#ifndef RELU_H_
#define RELU_H_    value

#include "..//Operator.h"

template<typename DTYPE> class Relu : public Operator<DTYPE>{
private:
#if __CUDNN__
    cudnnTensorDescriptor_t m_aInputTensorDesc, m_aOutputTensorDesc, m_aDeltaDesc, m_aInputDeltaDesc;
    cudnnActivationDescriptor_t actDesc;
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;
    DTYPE *m_aHostInput, *m_aHostOutput, *m_aHostInputDelta, *m_aHostDelta;

#endif  // __CUDNN__

public:
    Relu(Operator<DTYPE> *pInput) : Operator<DTYPE>(pInput) {
        std::cout << "Relu::Relu(Operator<DTYPE> *)" << '\n';
        this->Alloc(pInput);
    }

    Relu(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Relu::Relu(Operator<DTYPE> *)" << '\n';
        this->Alloc(pInput);
    }

    ~Relu() {
        std::cout << "Relu::~Relu()" << '\n';
#if __CUDNN__
        Delete();
#endif  // if __CUDNN__
    }

    int Alloc(Operator<DTYPE> *pInput) {
        std::cout << "Relu::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

#if __CUDNN__
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aOutputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputDeltaDesc));
        checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));

        int inputCapacity  = pInput->GetResult()->GetCapacity();
        int outputCapacity = this->GetResult()->GetCapacity();

        checkCudaErrors(cudaMalloc((void **)&m_pDevInput, (inputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevOutput, (outputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevInputDelta, (inputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevDelta, (outputCapacity * sizeof(DTYPE))));

        m_aHostInput      = new DTYPE[inputCapacity];
        m_aHostOutput     = new DTYPE[outputCapacity];
        m_aHostInputDelta = new DTYPE[inputCapacity];
        m_aHostDelta      = new DTYPE[outputCapacity];

#endif  // if __CUDNN__

        return TRUE;
    }

#if __CUDNN__
    void Delete() {
        checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(m_aOutputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(m_aDeltaDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputDeltaDesc));
        checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));

        checkCudaErrors(cudaFree(m_pDevInput));
        checkCudaErrors(cudaFree(m_pDevOutput));
        checkCudaErrors(cudaFree(m_pDevInputDelta));
        checkCudaErrors(cudaFree(m_pDevDelta));

        delete[] m_aHostInput;
        delete[] m_aHostOutput;
        delete[] m_aHostInputDelta;
        delete[] m_aHostDelta;
    }

#endif  // if __CUDNN__

    int ForwardPropagate() {
        if (this->GetDevice() == CPU) ComputeForwardPropagateOnCPU();
#ifdef __CUDNN__
        else if (this->GetDevice() == GPU) ComputeForwardPropagateOnGPU();
#endif  // if __CUDNN__
        else return FALSE;
        return TRUE;
    }

    int BackPropagate() {
        if (this->GetDevice() == CPU) ComputeBackPropagateOnCPU();
#ifdef __CUDNN__
        else if (this->GetDevice() == GPU) ComputeBackPropagateOnGPU();
#endif  // if __CUDNN__
        else return FALSE;
        return TRUE;
    }

    int ComputeForwardPropagateOnCPU() {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        for (int ti = 0; ti < timesize; ti++) {
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
        }

        return TRUE;
    }

    int ComputeBackPropagateOnCPU() {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        for (int ti = 0; ti < timesize; ti++) {
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
        }

        return TRUE;
    }

    inline DTYPE MAX(DTYPE data1, DTYPE data2) {
        if (data1 >= data2) return data1;
        else return data2;
    }

#if __CUDNN__
    int ComputeForwardPropagateOnGPU() {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfInput   = input->GetShape();
        int    inputCapacity  = input->GetCapacity();

        int batchsize   = (*shapeOfInput)[1];
        int channelsize = (*shapeOfInput)[2];
        int rowsize     = (*shapeOfInput)[3];
        int colsize     = (*shapeOfInput)[4];

        float alpha = 1.f; float beta = 0.f; double coef = 0.0;

        for (int i = 0; i < inputCapacity; i++) {
            m_aHostInput[i] = (*input)[i];
        }

        checkCUDNN(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, coef));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aOutputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCudaErrors(cudaMemcpy(m_pDevInput, m_aHostInput, (inputCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));

        // devInput and devOutput pointers may be equal but, identical Descriptor.
        checkCUDNN(cudnnActivationForward(this->GetCudnnHandle(), actDesc, &alpha,
                                          m_aInputTensorDesc, m_pDevInput, &beta,
                                          m_aOutputTensorDesc, m_pDevInput));

        checkCudaErrors(cudaMemcpy(m_aHostInput, m_pDevInput, (inputCapacity * sizeof(DTYPE)), cudaMemcpyDeviceToHost));

        for (int i = 0; i < inputCapacity; i++) {
            (*result)[i] = m_aHostInput[i];
        }

        return TRUE;
    }

    int ComputeBackPropagateOnGPU() {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Shape *shapeOfInput        = input->GetShape();

        int inputCapacity      = input->GetCapacity();
        int outputCapacity     = result->GetCapacity();
        int deltaCapacity      = this_delta->GetCapacity();
        int inputDeltaCapacity = input_delta->GetCapacity();

        int batchsize   = (*shapeOfInput)[1];
        int channelsize = (*shapeOfInput)[2];
        int rowsize     = (*shapeOfInput)[3];
        int colsize     = (*shapeOfInput)[4];

        float alpha = 1; float beta = 0; double coef = 0.0;

        for (int i = 0; i < inputCapacity; i++) {
            m_aHostInput[i] = (*input)[i];
        }

        for (int i = 0; i < outputCapacity; i++) {
            m_aHostOutput[i] = (*result)[i];
        }

        for (int i = 0; i < deltaCapacity; i++) {
            m_aHostDelta[i] = (*this_delta)[i];
        }

        checkCUDNN(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, coef));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aOutputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCudaErrors(cudaMemcpy(m_pDevInput, m_aHostInput, (inputCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(m_pDevOutput, m_aHostOutput, (outputCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(m_pDevDelta, m_aHostDelta, (deltaCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));


        checkCUDNN(cudnnActivationBackward(this->GetCudnnHandle(), actDesc, &alpha,
                                           m_aOutputTensorDesc, m_pDevOutput,
                                           m_aDeltaDesc, m_pDevDelta, m_aInputTensorDesc, m_pDevInput, &beta, m_aInputTensorDesc, m_pDevInputDelta));

        checkCudaErrors(cudaMemcpy(m_aHostInputDelta, m_pDevInputDelta, (inputDeltaCapacity * sizeof(DTYPE)), cudaMemcpyDeviceToHost));

        for (int i = 0; i < inputDeltaCapacity; i++) {
            (*input_delta)[i] += m_aHostInputDelta[i];
        }

        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // RELU_H_
