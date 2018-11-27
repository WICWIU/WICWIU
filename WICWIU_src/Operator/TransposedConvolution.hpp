#ifndef TRANSPOSEDCONVOLUTION_H_
#define TRANSPOSEDCONVOLUTION_H_    value

#include "../Operator.hpp"
#include <cstdio>

template<typename DTYPE> class TransposedConvolution2D : public Operator<DTYPE>{
private:
    int m_stride[2];
    int m_padding[2];

#ifdef __CUDNN__
    cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc, deltaDesc, inputDeltaDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc, filterDeltaDesc;
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevFilter, *m_pDevInputDelta, *m_pDevDelta, *m_pDevFilterDelta;

    cudnnConvolutionFwdAlgo_t m_algo;
    cudnnConvolutionBwdFilterAlgo_t m_filterAlgo;
    cudnnConvolutionBwdDataAlgo_t m_dataAlgo;

    size_t m_sizeInBytes;
    size_t m_dataSizeInBytes;
    size_t m_filterSizeInBytes;

    DTYPE m_alpha;
    DTYPE m_beta;

    void *m_devWorkSpace;
    void *m_dataDevWorkSpace;
    void *m_filterDevWorkSpace;
#endif  // __CUDNN__

public:
    TransposedConvolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, std::string pName = "NO NAME") : Operator<DTYPE>(pInput, pWeight, pName) {
        Alloc(pInput, pWeight, stride1, stride2, 0, 0);
    }

    TransposedConvolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding, std::string pName = "NO NAME") : Operator<DTYPE>(pInput, pWeight, pName) {
        Alloc(pInput, pWeight, stride1, stride2, padding, padding);
    }

    TransposedConvolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding1, int padding2, std::string pName = "NO NAME") : Operator<DTYPE>(pInput, pWeight, pName) {
        Alloc(pInput, pWeight, stride1, stride2, padding1, padding2);
    }

    virtual ~TransposedConvolution2D() {
        #ifdef __DEBUG__
        std::cout << "TransposedConvolution2D::~TransposedConvolution2D()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding1, int padding2) {
        int outputWidth  = 0;
        int outputHeight = 0;

        Shape *shapeOfInput  = pInput->GetResult()->GetShape();
        Shape *shapeOfWeight = pWeight->GetResult()->GetShape();

        if ((*shapeOfInput)[0] != 1) {
            printf("Receive invalid timesize value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        m_stride[0] = stride1;
        m_stride[1] = stride2;

        m_padding[0] = padding1;
        m_padding[1] = padding2;

        outputHeight = m_stride[0]*((*shapeOfInput)[3] - 1) + (*shapeOfWeight)[3] - (2 * m_padding[0]);
        outputWidth  = m_stride[1]*((*shapeOfInput)[4] - 1) + (*shapeOfWeight)[4] - (2 * m_padding[1]);

        this->SetResult(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[2], outputHeight, outputWidth));
        this->SetDelta(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[2], outputHeight, outputWidth));

        return TRUE;
    }


#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        Operator<DTYPE> *pInput  = this->GetInput()[0];
        Operator<DTYPE> *pWeight = this->GetInput()[1];

        Shape *shapeOfWeight = pWeight->GetResult()->GetShape();
        Shape *shapeOfInput  = pInput->GetResult()->GetShape();
        Shape *shapeOfResult = this->GetResult()->GetShape();

        int batchsize   = this->GetResult()->GetBatchSize();
        int channelsize = this->GetResult()->GetChannelSize();
        int rowsize     = this->GetResult()->GetRowSize();
        int colsize     = this->GetResult()->GetColSize();

        int batchsizeOfWeight   = (*shapeOfWeight)[1];
        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

        int batchsizeOfInput   = (*shapeOfInput)[1];
        int channelsizeOfInput = (*shapeOfInput)[2];
        int rowsizeOfInput     = (*shapeOfInput)[3];
        int colsizeOfInput     = (*shapeOfInput)[4];

        m_sizeInBytes       = 0;
        m_dataSizeInBytes   = 0;
        m_filterSizeInBytes = 0;

        m_alpha = 1;
        m_beta  = 0;

        m_devWorkSpace       = NULL;
        m_dataDevWorkSpace   = NULL;
        m_filterDevWorkSpace = NULL;

        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&deltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&inputDeltaDesc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDeltaDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfInput, channelsizeOfInput, rowsizeOfInput, colsizeOfInput));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              batchsizeOfWeight, channelsizeOfWeight, rowsizeOfWeight, colsizeOfWeight));

        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(deltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfInput, channelsizeOfInput, rowsizeOfInput, colsizeOfInput));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDeltaDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              batchsizeOfWeight, channelsizeOfWeight, rowsizeOfWeight, colsizeOfWeight));

        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, m_padding[0], m_padding[1], m_stride[0], m_stride[1],
                                                   1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));


        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(this->GetCudnnHandle(), outputTensorDesc, filterDesc, convDesc, inputTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &m_algo));

        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(this->GetCudnnHandle(), outputTensorDesc, filterDesc, convDesc,
                                                         inputTensorDesc, m_algo, &m_sizeInBytes));

        checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(this->GetCudnnHandle(), filterDesc, inputDeltaDesc, convDesc, deltaDesc,
                                                            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, 0, &m_dataAlgo));

        checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->GetCudnnHandle(), deltaDesc, inputTensorDesc, convDesc, filterDeltaDesc,
                                                              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, 0, &m_filterAlgo));

        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(this->GetCudnnHandle(), filterDesc, inputDeltaDesc, convDesc, deltaDesc, m_dataAlgo, &m_dataSizeInBytes));

        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(this->GetCudnnHandle(), deltaDesc, inputTensorDesc, convDesc, filterDeltaDesc, m_filterAlgo, &m_filterSizeInBytes));

        if (m_sizeInBytes != 0) {
            checkCudaErrors(cudaMalloc(&m_devWorkSpace, m_sizeInBytes));

            if (m_devWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                exit(-1);
            }
        }

        if (m_dataSizeInBytes != 0) {
            checkCudaErrors(cudaMalloc(&m_dataDevWorkSpace, m_dataSizeInBytes));

            if (m_dataDevWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                exit(-1);
            }
        }

        if (m_filterSizeInBytes != 0) {
            checkCudaErrors(cudaMalloc(&m_filterDevWorkSpace, m_filterSizeInBytes));

            if (m_filterDevWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                exit(-1);
            }
        }

        // checkCudaErrors(cudaDeviceSynchronize());
    }

#endif  // if __CUDNN__

    void Delete() {
#ifdef __CUDNN__

        if (inputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
        inputTensorDesc = NULL;

        if (outputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
        outputTensorDesc = NULL;

        if (deltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(deltaDesc));
        deltaDesc = NULL;

        if (inputDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(inputDeltaDesc));
        inputDeltaDesc = NULL;

        if (convDesc) checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        convDesc = NULL;

        if (filterDesc) checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
        filterDesc = NULL;

        if (filterDeltaDesc) checkCUDNN(cudnnDestroyFilterDescriptor(filterDeltaDesc));
        filterDeltaDesc = NULL;

        if (m_sizeInBytes != 0) {
            checkCudaErrors(cudaFree(m_devWorkSpace));
        }

        if (m_dataSizeInBytes != 0) {
            checkCudaErrors(cudaFree(m_dataDevWorkSpace));
        }

        if (m_filterSizeInBytes != 0) {
            checkCudaErrors(cudaFree(m_filterDevWorkSpace));
        }

        // checkCudaErrors(cudaDeviceSynchronize());
#endif  // if __CUDNN__
    }

    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Shape *shapeOfWeight  = weight->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

        int channelsizeOfInput = (*shapeOfInput)[2];
        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        int ti          = pTime;

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsizeOfInput; ch++) {
                for (int ro = 0; ro < rowsizeOfInput; ro++) {
                    for (int co = 0; co < colsizeOfInput; co++) {
                        for (int wch = 0; wch < channelsizeOfWeight; wch++) {
                            for (int wro = 0; wro < rowsizeOfWeight; wro++) {
                                for (int wco = 0; wco < colsizeOfWeight; wco++) {

                                    result_index  = Index5D(shapeOfResult, ti, ba, wch, (ro * m_stride[0]) + wro, (co * m_stride[1]) + wco);
                                    weight_index = Index5D(shapeOfWeight, 0, ch, wch, wro, wco);
                                    input_index = Index5D(shapeOfInput, ti, ba, ch, ro, co);

                                    (*result)[result_index] += ((*input)[input_index] * (*weight)[weight_index]);
                                }
                            }
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Shape *shapeOfInput        = input->GetShape();

        Tensor<DTYPE> *weight          = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[1]->GetGradient();
        Shape *shapeOfWeight           = weight->GetShape();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfResult      = this_delta->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

        int channelsizeOfInput = (*shapeOfInput)[2];
        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        int ti          = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
          for (int ch = 0; ch < channelsizeOfInput; ch++) {
            for (int ro = 0; ro < rowsizeOfInput; ro++) {
              for (int co = 0; co < colsizeOfInput; co++) {
                for (int wch = 0; wch < channelsizeOfWeight; wch++) {
                  for (int wro = 0; wro < rowsizeOfWeight; wro++) {
                    for (int wco = 0; wco < colsizeOfWeight; wco++) {

                      result_index  = Index5D(shapeOfResult, ti, ba, wch, (ro * m_stride[0]) + wro, (co * m_stride[1]) + wco);
                      weight_index = Index5D(shapeOfWeight, 0, ch, wch, wro, wco);
                      input_index = Index5D(shapeOfInput, ti, ba, ch, ro, co);

                      (*input_delta)[input_index]
                            += ((*weight)[weight_index] * (*this_delta)[result_index]);
                      (*weight_gradient)[weight_index]
                            += ((*input)[input_index] * (*this_delta)[result_index]);
                    }
                  }
                }
              }
            }
          }
        }

        return TRUE;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {
        //this->ForwardPropagate(pTime);

        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevFilter = weight->GetGPUData(0);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnConvolutionBackwardData(this->GetCudnnHandle(), &m_alpha, filterDesc, m_pDevFilter, inputTensorDesc, m_pDevInput, convDesc,
                                               m_dataAlgo, m_dataDevWorkSpace, m_dataSizeInBytes, &m_beta, outputTensorDesc, m_pDevOutput));


        // checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        //this->BackPropagate(pTime);

        Tensor<DTYPE> *input           = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta     = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *weight          = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[1]->GetGradient();
        Tensor<DTYPE> *this_delta      = this->GetDelta();

        m_pDevInput       = input->GetGPUData(pTime);
        m_pDevFilter      = weight->GetGPUData(0);
        m_pDevDelta       = this_delta->GetGPUData(pTime);
        m_pDevInputDelta  = input_delta->GetGPUData(pTime);
        m_pDevFilterDelta = weight_gradient->GetGPUData(0);



        checkCUDNN(cudnnConvolutionForward(this->GetCudnnHandle(), &m_alpha, deltaDesc, m_pDevDelta, filterDesc, m_pDevFilter, convDesc,
                                           m_algo, m_devWorkSpace, m_sizeInBytes, &m_beta, inputDeltaDesc, m_pDevInputDelta));

        checkCUDNN(cudnnConvolutionBackwardFilter(this->GetCudnnHandle(), &m_alpha, deltaDesc, m_pDevDelta, inputTensorDesc, m_pDevInput, convDesc,
                                                  m_filterAlgo, m_filterDevWorkSpace, m_filterSizeInBytes, &m_beta, filterDesc, m_pDevFilterDelta));

        // checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

#endif  // if __CUDNN__


};

#endif  // TRANSPOSEDCONVOLUTION_H_
