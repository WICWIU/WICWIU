#ifndef MAXPOOLING_H_
#define MAXPOOLING_H_    value

#include "..//Operator.h"

template<typename DTYPE> class Maxpooling2D : public Operator<DTYPE>{
private:
    int m_stride[2];
    int m_mask[2];
    int m_padding[2];

    Tensor<int> *indexOfMaxInput;

#if __CUDNN__
    cudnnTensorDescriptor_t m_aInputTensorDesc, m_aOutputTensorDesc, m_aDeltaDesc, m_aInputDeltaDesc;
    cudnnPoolingDescriptor_t m_aPoolingDesc;
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;
    DTYPE *m_aHostInput, *m_aHostOutput, *m_aHostInputDelta, *m_aHostDelta;

#endif  // __CUDNN__

public:
    Maxpooling2D(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Maxpooling2D::Maxpooling2D(Operator<DTYPE> *, int, int)" << '\n';
        this->Alloc(pInput, strideRow, strideCol, maskRow, maskCol);
    }

    Maxpooling2D(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol, int padding, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Maxpooling2D::Maxpooling2D(Operator<DTYPE> *, int, int, std::string)" << '\n';
        this->Alloc(pInput, strideRow, strideCol, maskRow, maskCol, padding, padding);
    }

    ~Maxpooling2D() {
        std::cout << "Maxpooling2D::~Maxpooling2D()" << '\n';
#if __CUDNN__
        Delete();
#endif  // if __CUDNN__
    }

    int Alloc(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol, int padding1 = 0, int padding2 = 0) {
        std::cout << "Maxpooling2D::Alloc(Operator<DTYPE> *, int, int)" << '\n';

        Shape *shapeOfInput = pInput->GetResult()->GetShape();

        m_stride[0] = strideRow;
        m_stride[1] = strideCol;

        m_mask[0] = maskRow;
        m_mask[1] = maskCol;

        m_padding[0] = padding1;
        m_padding[1] = padding2;

        int rowsize = 0;
        int colsize = 0;

        rowsize = ((*shapeOfInput)[3] - maskRow + (2 * m_padding[0])) / strideRow + 1;
        colsize = ((*shapeOfInput)[4] - maskCol + (2 * m_padding[1])) / strideCol + 1;

        this->SetResult(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], rowsize, colsize));
        this->SetDelta(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], rowsize, colsize));

        indexOfMaxInput = new Tensor<int>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], rowsize, colsize);

#if __CUDNN__
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aOutputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputDeltaDesc));
        checkCUDNN(cudnnCreatePoolingDescriptor(&m_aPoolingDesc));

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
        checkCUDNN(cudnnDestroyPoolingDescriptor(m_aPoolingDesc));

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
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();
        // result->Reset();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        int rowsizeOfMask = m_mask[0];
        int colsizeOfMask = m_mask[1];

        DTYPE max = 0.f;

        int indexOfResult = 0;
        int indexOfInput  = 0;

        int temprow = 0;
        int tempcol = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int mro = 0; mro < rowsizeOfMask; mro++) {
                            for (int mco = 0; mco < colsizeOfMask; mco++) {
                                temprow = m_stride[0] * ro + mro;
                                tempcol = m_stride[1] * co + mco;

                                indexOfResult = Index4D(shapeOfResult, ba, ch, ro, co);
                                indexOfInput  = Index4D(shapeOfInput, ba, ch, temprow, tempcol);

                                if ((mro == 0) && (mco == 0)) {
                                    max                               = (*input)[indexOfInput];
                                    (*result)[indexOfResult]          = max;
                                    (*indexOfMaxInput)[indexOfResult] = indexOfInput;
                                } else {
                                    if (max < (*input)[indexOfInput]) {
                                        max                               = (*input)[indexOfInput];
                                        (*result)[indexOfResult]          = max;
                                        (*indexOfMaxInput)[indexOfResult] = indexOfInput;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    int ComputeBackPropagateOnCPU() {
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        // input_delta->Reset();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfDelta       = this_delta->GetShape();

        int batchsize   = (*shapeOfDelta)[1];
        int channelsize = (*shapeOfDelta)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfDelta)[3];
        int colsize     = (*shapeOfDelta)[4];

        int indexOfDelta = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        indexOfDelta                                      = Index4D(shapeOfDelta, ba, ch, ro, co);
                        (*input_delta)[(*indexOfMaxInput)[indexOfDelta]] += (*this_delta)[indexOfDelta];
                    }
                }
            }
        }

        return TRUE;
    }

#if __CUDNN__
    int ComputeForwardPropagateOnGPU() {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        int rowsizeOfMask = m_mask[0];
        int colsizeOfMask = m_mask[1];

        int inputCapacity  = input->GetCapacity();
        int resultCapacity = result->GetCapacity();

        DTYPE alpha = 1; DTYPE beta = 0;

        for (int i = 0; i < inputCapacity; i++) {
            m_aHostInput[i] = (*input)[i];
        }

        checkCudaErrors(cudaMemcpy(m_pDevInput, m_aHostInput, (inputCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));

        checkCUDNN(cudnnSetPooling2dDescriptor(m_aPoolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                               rowsizeOfMask, colsizeOfMask, m_padding[0], m_padding[1], m_stride[0], m_stride[1]));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsizeOfInput, colsizeOfInput));

        checkCUDNN(cudnnGetPooling2dForwardOutputDim(m_aPoolingDesc, m_aInputTensorDesc,
                                                     &batchsize, &channelsize, &rowsize, &colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aOutputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnPoolingForward(this->GetCudnnHandle(), m_aPoolingDesc, &alpha, m_aInputTensorDesc, m_pDevInput,
                                       &beta, m_aOutputTensorDesc, m_pDevOutput));

        checkCudaErrors(cudaMemcpy(m_aHostOutput, m_pDevOutput, (resultCapacity * sizeof(DTYPE)), cudaMemcpyDeviceToHost));
        // checkCudaErrors(cudaDeviceSynchronize());

        for (int i = 0; i < resultCapacity; i++) {
            (*result)[i] = m_aHostOutput[i];
        }

        return TRUE;
    }

    int ComputeBackPropagateOnGPU() {
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *this_delta  = this->GetDelta();

        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        int indexOfDelta = 0;

        int n = (*shapeOfInput)[1]; int h = (*shapeOfInput)[3];
        int c = (*shapeOfInput)[2]; int w = (*shapeOfInput)[4];

        float alpha = 1; float beta = 0;
        int   inputCapacity      = input->GetCapacity();
        int   resultCapacity     = result->GetCapacity();
        int   deltaCapacity      = this_delta->GetCapacity();
        int   inputDeltaCapacity = inputCapacity;

        for (int i = 0; i < deltaCapacity; i++) {
            m_aHostDelta[i] = (*this_delta)[i];
        }

        for (int i = 0; i < inputCapacity; i++) {
            m_aHostInput[i] = (*input)[i];
        }

        for (int i = 0; i < resultCapacity; i++) {
            m_aHostOutput[i] = (*result)[i];
        }

        checkCudaErrors(cudaMemcpy(m_pDevInput, m_aHostInput, (inputCapacity * sizeof(float)), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_pDevOutput, m_aHostOutput, (resultCapacity * sizeof(float)), cudaMemcpyHostToDevice));

        checkCUDNN(cudnnSetPooling2dDescriptor(m_aPoolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                               m_mask[0], m_mask[1], m_padding[0], m_padding[1], m_stride[0], m_stride[1]));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsizeOfInput, colsizeOfInput));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsizeOfInput, colsizeOfInput));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aOutputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnPoolingBackward(this->GetCudnnHandle(), m_aPoolingDesc,
                                        &alpha, m_aOutputTensorDesc, m_pDevOutput,
                                        m_aDeltaDesc, m_pDevDelta, m_aInputTensorDesc, m_pDevInput,
                                        &beta, m_aInputDeltaDesc, m_pDevInputDelta));

        checkCudaErrors(cudaMemcpy(m_aHostInputDelta, m_pDevInputDelta, (inputDeltaCapacity * sizeof(float)), cudaMemcpyDeviceToHost));

        for (int i = 0; i < inputDeltaCapacity; i++) {
            (*input_delta)[i] = m_aHostInputDelta[i];
        }

        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // MAXPOOLING_H_
