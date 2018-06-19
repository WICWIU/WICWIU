#ifndef MAXPOOLING_H_
#define MAXPOOLING_H_    value

#include "../Operator.h"

template<typename DTYPE> class Maxpooling2D : public Operator<DTYPE>{
private:
    int m_stride[2];
    int m_mask[2];
    int m_padding[2];

    Tensor<int> *indexOfMaxInput;

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
    Maxpooling2D(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "Maxpooling2D::Maxpooling2D(Operator<DTYPE> *, int, int)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, strideRow, strideCol, maskRow, maskCol);
    }

    Maxpooling2D(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol, int padding, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "Maxpooling2D::Maxpooling2D(Operator<DTYPE> *, int, int, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, strideRow, strideCol, maskRow, maskCol, padding, padding);
    }

    ~Maxpooling2D() {
        #ifdef __DEBUG__
        std::cout << "Maxpooling2D::~Maxpooling2D()" << '\n';
        #endif  // __DEBUG__
#ifdef __CUDNN__
        Delete();
#endif  // if __CUDNN__
    }

    int Alloc(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol, int padding1 = 0, int padding2 = 0) {
        #ifdef __DEBUG__
        std::cout << "Maxpooling2D::Alloc(Operator<DTYPE> *, int, int)" << '\n';
        #endif  // __DEBUG__

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

        return TRUE;
    }

#ifdef __CUDNN__
    void InitializeAttributeForGPU() {
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

        m_alpha = 1.f;
        m_beta  = 0.f;

        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aOutputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputDeltaDesc));
        checkCUDNN(cudnnCreatePoolingDescriptor(&m_aPoolingDesc));

        checkCUDNN(cudnnSetPooling2dDescriptor(m_aPoolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                               m_mask[0], m_mask[1], m_padding[0], m_padding[1], m_stride[0], m_stride[1]));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsizeOfInput, colsizeOfInput));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aOutputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsizeOfInput, colsizeOfInput));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnGetPooling2dForwardOutputDim(m_aPoolingDesc, m_aInputTensorDesc,
                                                     &batchsize, &channelsize, &rowsize, &colsize));
    }

#endif  // if __CUDNN__

    void Delete() {
        delete indexOfMaxInput;
#ifdef __CUDNN__

        if (m_aInputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputTensorDesc));
        m_aInputTensorDesc = NULL;

        if (m_aOutputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aOutputTensorDesc));
        m_aOutputTensorDesc = NULL;

        if (m_aDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aDeltaDesc));
        m_aDeltaDesc = NULL;

        if (m_aInputDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputDeltaDesc));
        m_aInputDeltaDesc = NULL;

        if (m_aPoolingDesc) checkCUDNN(cudnnDestroyPoolingDescriptor(m_aPoolingDesc));
        m_aPoolingDesc = NULL;

        checkCudaErrors(cudaThreadSynchronize());

#endif  // if __CUDNN__
    }

    // 메모리 효율을 생각하면 time에 따라 취해야 할 액션이 다르다.
    int ForwardPropagate(int pTime = 0, int pThreadNum = 0) {
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

        int ti = pTime;
        int numOfThread = this->GetNumOfThread();

        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int mro = 0; mro < rowsizeOfMask; mro++) {
                            for (int mco = 0; mco < colsizeOfMask; mco++) {
                                temprow = m_stride[0] * ro + mro;
                                tempcol = m_stride[1] * co + mco;

                                indexOfResult = Index5D(shapeOfResult, ti, ba, ch, ro, co);
                                indexOfInput  = Index5D(shapeOfInput, ti, ba, ch, temprow, tempcol);

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

    int BackPropagate(int pTime = 0, int pThreadNum = 0) {
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfDelta       = this_delta->GetShape();

        int batchsize   = (*shapeOfDelta)[1];
        int channelsize = (*shapeOfDelta)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfDelta)[3];
        int colsize     = (*shapeOfDelta)[4];

        int indexOfDelta = 0;

        int ti = pTime;
        int numOfThread = this->GetNumOfThread();

        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        indexOfDelta                                      = Index5D(shapeOfDelta, ti, ba, ch, ro, co);
                        (*input_delta)[(*indexOfMaxInput)[indexOfDelta]] += (*this_delta)[indexOfDelta];
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

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnPoolingForward(this->GetCudnnHandle(), m_aPoolingDesc, &m_alpha, m_aInputTensorDesc, m_pDevInput,
                                       &m_beta, m_aOutputTensorDesc, m_pDevOutput));


        checkCudaErrors(cudaDeviceSynchronize());
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result      = this->GetResult();

        m_pDevInput      = input->GetGPUData(pTime);
        m_pDevOutput     = result->GetGPUData(pTime);
        m_pDevDelta      = this_delta->GetGPUData(pTime);
        m_pDevInputDelta = input_delta->GetGPUData(pTime);

        checkCUDNN(cudnnPoolingBackward(this->GetCudnnHandle(), m_aPoolingDesc,
                                        &m_alpha, m_aOutputTensorDesc, m_pDevOutput,
                                        m_aDeltaDesc, m_pDevDelta, m_aInputTensorDesc, m_pDevInput,
                                        &m_beta, m_aInputDeltaDesc, m_pDevInputDelta));


        checkCudaErrors(cudaDeviceSynchronize());
        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // MAXPOOLING_H_
