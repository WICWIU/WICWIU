#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_    value

#include "..//Operator.h"
#include <cstdio>

template<typename DTYPE> class Convolution2D : public Operator<DTYPE>{
private:
    int m_stride[2];
    int m_padding[2];

#if __CUDNN__
    cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc, deltaDesc, inputDeltaDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc, filterDeltaDesc;
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevFilter, *m_pDevInputDelta, *m_pDevDelta, *m_pDevFilterDelta;
    DTYPE *m_aHostInput, *m_aHostOutput, *m_aHostFilter, *m_aHostInputDelta, *m_aHostDelta, *m_aHostFilterDelta;
#endif  // __CUDNN__

public:
    Convolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, std::string pName = "NO NAME") : Operator<DTYPE>(pInput, pWeight, pName) {
        Alloc(pInput, pWeight, stride1, stride2, 0, 0);
    }

    Convolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding, std::string pName = "NO NAME") : Operator<DTYPE>(pInput, pWeight, pName) {
        Alloc(pInput, pWeight, stride1, stride2, padding, padding);
    }

    Convolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding1, int padding2, std::string pName = "NO NAME") : Operator<DTYPE>(pInput, pWeight, pName) {
        Alloc(pInput, pWeight, stride1, stride2, padding1, padding2);
    }

    virtual ~Convolution2D() {
        std::cout << "Convolution2D::~Convolution2D()" << '\n';
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

        outputHeight = ((*shapeOfInput)[3] - (*shapeOfWeight)[3] + (2 * m_padding[0])) / m_stride[0] + 1;
        outputWidth  = ((*shapeOfInput)[4] - (*shapeOfWeight)[4] + (2 * m_padding[1])) / m_stride[1] + 1;

        this->SetResult(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], outputHeight, outputWidth));
        this->SetDelta(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], outputHeight, outputWidth));

#if __CUDNN__
        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&deltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&inputDeltaDesc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDeltaDesc));

        int inputCapacity  = pInput->GetResult()->GetCapacity();
        int outputCapacity = this->GetResult()->GetCapacity();
        int filterCapacity = pWeight->GetResult()->GetCapacity();

        checkCudaErrors(cudaMalloc((void **)&m_pDevInput, (inputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevOutput, (outputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevFilter, (filterCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevInputDelta, (inputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevDelta, (outputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevFilterDelta, (filterCapacity * sizeof(DTYPE))));

        m_aHostInput       = new DTYPE[inputCapacity];
        m_aHostOutput      = new DTYPE[outputCapacity];
        m_aHostFilter      = new DTYPE[filterCapacity];
        m_aHostInputDelta  = new DTYPE[inputCapacity];
        m_aHostDelta       = new DTYPE[outputCapacity];
        m_aHostFilterDelta = new DTYPE[filterCapacity];

#endif  // if __CUDNN__

        return TRUE;
    }

    void Delete() {
#if __CUDNN__
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(deltaDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(inputDeltaDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDeltaDesc));

        checkCudaErrors(cudaFree(m_pDevInput));
        checkCudaErrors(cudaFree(m_pDevOutput));
        checkCudaErrors(cudaFree(m_pDevFilter));
        checkCudaErrors(cudaFree(m_pDevInputDelta));
        checkCudaErrors(cudaFree(m_pDevDelta));
        checkCudaErrors(cudaFree(m_pDevFilterDelta));

        delete[] m_aHostInput;
        delete[] m_aHostOutput;
        delete[] m_aHostFilter;
        delete[] m_aHostInputDelta;
        delete[] m_aHostDelta;
        delete[] m_aHostFilterDelta;
#endif  // if __CUDNN__
    }

    int ForwardPropagate() {
        if (this->GetDevice() == CPU) ComputeForwardPropagateOnCPU();
        // if (this->GetDevice() == CPU) ComputeForwardPropagateOnCPU_MT();
#ifdef __CUDNN__
        else if (this->GetDevice() == GPU) ComputeForwardPropagateOnGPU();
#endif  // if __CUDNN__
        else return FALSE;
        return TRUE;
    }

    int BackPropagate() {
        if (this->GetDevice() == CPU) ComputeBackPropagateOnCPU();
        // if (this->GetDevice() == CPU) ComputeBackPropagateOnCPU_MT();
#ifdef __CUDNN__
        else if (this->GetDevice() == GPU) ComputeBackPropagateOnGPU();
#endif  // if __CUDNN__
        else return FALSE;
        return TRUE;
    }

    // int ComputeForwardPropagateOnCPU_MT() {
    // int numOfThread = this->GetInput()[0]->GetResult()->GetBatchSize();
    //
    // std::thread **pthread = new std::thread *[numOfThread];
    //
    // for (int i = 0; i < numOfThread; i++) {
    // pthread[i] = new std::thread(ComputeForwardPropagateOnCPU_T, this, i, numOfThread);
    // }
    //
    // for (int i = 0; i < numOfThread; i++) {
    // pthread[i]->join();
    // }
    //
    // return TRUE;
    // }
    //
    // static int ComputeForwardPropagateOnCPU_T(Convolution2D<DTYPE> *obj, int threadNum, int numOfThread) {
    // Tensor<DTYPE> *input = obj->GetInput()[0]->GetResult();
    // Shape *shapeOfInput  = input->GetShape();
    //
    // Tensor<DTYPE> *weight = obj->GetInput()[1]->GetResult();
    // Shape *shapeOfWeight  = weight->GetShape();
    //
    // Tensor<DTYPE> *result = obj->GetResult();
    // Shape *shapeOfResult  = result->GetShape();
    //
    // int batchsize   = (*shapeOfResult)[1];
    // int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
    // int rowsize     = (*shapeOfResult)[3];
    // int colsize     = (*shapeOfResult)[4];
    //
    // int channelsizeOfWeight = (*shapeOfWeight)[2];
    // int rowsizeOfWeight     = (*shapeOfWeight)[3];
    // int colsizeOfWeight     = (*shapeOfWeight)[4];
    //
    // int rowsizeOfInput = (*shapeOfInput)[3];
    // int colsizeOfInput = (*shapeOfInput)[4];
    //
    // for (int ba = threadNum; ba < batchsize; ba += numOfThread) {
    // for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
    // for (int ro = 0; ro < rowsize; ro++) {
    // for (int co = 0; co < colsize; co++) {
    // for (int wch = 0; wch < channelsizeOfWeight; wch++) {  // == (*shapeOfInput)[2];
    // for (int wro = 0; wro < rowsizeOfWeight; wro++) {
    // for (int wco = 0; wco < colsizeOfWeight; wco++) {
    // (*result)[Index4D(shapeOfResult, ba, ch, ro, co)]
    // += ((*input)[Index4D(shapeOfInput, ba, wch, (obj->GetStrideList())[0] * ro + wro, (obj->GetStrideList())[1] * co + wco)]
    // * (*weight)[Index4D(shapeOfWeight, ch, wch, wro, wco)]);
    // }
    // }
    // }
    // }
    // }
    // }
    // }
    //
    // return TRUE;
    // }

    int ComputeForwardPropagateOnCPU() {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Shape *shapeOfWeight  = weight->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int wch = 0; wch < channelsizeOfWeight; wch++) {  // == (*shapeOfInput)[2];
                            for (int wro = 0; wro < rowsizeOfWeight; wro++) {
                                for (int wco = 0; wco < colsizeOfWeight; wco++) {
                                    (*result)[Index4D(shapeOfResult, ba, ch, ro, co)]
                                        += ((*input)[Index4D(shapeOfInput, ba, wch, m_stride[0] * ro + wro, m_stride[1] * co + wco)]
                                            * (*weight)[Index4D(shapeOfWeight, ch, wch, wro, wco)]);
                                }
                            }
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    // int ComputeBackPropagateOnCPU_MT() {
    // int numOfThread = this->GetInput()[0]->GetResult()->GetBatchSize();
    //
    // std::thread **pthread = new std::thread *[numOfThread];
    //
    // for (int i = 0; i < numOfThread; i++) {
    // pthread[i] = new std::thread(ComputeBackPropagateOnCPU_T, this, i, numOfThread);
    // }
    //
    // for (int i = 0; i < numOfThread; i++) {
    // pthread[i]->join();
    // }
    //
    // return TRUE;
    // }
    //
    // static int ComputeBackPropagateOnCPU_T(Convolution2D<DTYPE> *obj, int threadNum, int numOfThread) {
    // Tensor<DTYPE> *input       = obj->GetInput()[0]->GetResult();
    // Tensor<DTYPE> *input_delta = obj->GetInput()[0]->GetDelta();
    // Shape *shapeOfInput        = input->GetShape();
    //
    // Tensor<DTYPE> *weight          = obj->GetInput()[1]->GetResult();
    // Tensor<DTYPE> *weight_gradient = obj->GetInput()[1]->GetGradient();
    // Shape *shapeOfWeight           = weight->GetShape();
    //
    // Tensor<DTYPE> *this_delta = obj->GetDelta();
    // Shape *shapeOfResult      = this_delta->GetShape();
    //
    // int batchsize   = (*shapeOfResult)[1];
    // int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
    // int rowsize     = (*shapeOfResult)[3];
    // int colsize     = (*shapeOfResult)[4];
    //
    // int channelsizeOfWeight = (*shapeOfWeight)[2];
    // int rowsizeOfWeight     = (*shapeOfWeight)[3];
    // int colsizeOfWeight     = (*shapeOfWeight)[4];
    //
    // int rowsizeOfInput = (*shapeOfInput)[3];
    // int colsizeOfInput = (*shapeOfInput)[4];
    //
    // int input_index  = 0;
    // int weight_index = 0;
    // int result_index = 0;
    //
    // for (int ba = threadNum; ba < batchsize; ba += numOfThread) {
    // for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
    // for (int ro = 0; ro < rowsize; ro++) {
    // for (int co = 0; co < colsize; co++) {
    // for (int wch = 0; wch < channelsizeOfWeight; wch++) {  // == (*shapeOfInput)[2];
    // for (int wro = 0; wro < rowsizeOfWeight; wro++) {
    // for (int wco = 0; wco < colsizeOfWeight; wco++) {
    // input_index  = Index4D(shapeOfInput, ba, wch, (obj->GetStrideList())[0] * ro + wro, (obj->GetStrideList())[1] * co + wco);
    // weight_index = Index4D(shapeOfWeight, ch, wch, wro, wco);
    // result_index = Index4D(shapeOfResult, ba, ch, ro, co);
    //
    // (*input_delta)[input_index]
    // += ((*weight)[weight_index] * (*this_delta)[result_index]);
    //
    // (*weight_gradient)[weight_index]
    // += ((*input)[input_index] * (*this_delta)[result_index]);
    // }
    // }
    // }
    // }
    // }
    // }
    // }
    //
    // return TRUE;
    // }

    int ComputeBackPropagateOnCPU() {
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Shape *shapeOfInput        = input->GetShape();

        Tensor<DTYPE> *weight          = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[1]->GetGradient();
        Shape *shapeOfWeight           = weight->GetShape();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfResult      = this_delta->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int wch = 0; wch < channelsizeOfWeight; wch++) {  // == (*shapeOfInput)[2];
                            for (int wro = 0; wro < rowsizeOfWeight; wro++) {
                                for (int wco = 0; wco < colsizeOfWeight; wco++) {
                                    input_index  = Index4D(shapeOfInput, ba, wch, m_stride[0] * ro + wro, m_stride[1] * co + wco);
                                    weight_index = Index4D(shapeOfWeight, ch, wch, wro, wco);
                                    result_index = Index4D(shapeOfResult, ba, ch, ro, co);

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

#if __CUDNN__
    int ComputeForwardPropagateOnGPU() {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Shape *shapeOfWeight  = weight->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int batchsizeOfWeight   = (*shapeOfWeight)[1];
        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

        int batchsizeOfInput   = (*shapeOfInput)[1];
        int channelsizeOfInput = (*shapeOfInput)[2];
        int rowsizeOfInput     = (*shapeOfInput)[3];
        int colsizeOfInput     = (*shapeOfInput)[4];

        cudnnConvolutionFwdAlgo_t algo;
        DTYPE alpha = 1;
        DTYPE beta  = 0;

        int inputCapacity  = input->GetCapacity();
        int outputCapacity = result->GetCapacity();
        int filterCapacity = weight->GetCapacity();

        for (int i = 0; i < inputCapacity; i++) {
            m_aHostInput[i] = (*input)[i];
        }

        for (int i = 0; i < filterCapacity; i++) {
            m_aHostFilter[i] = (*weight)[i];
        }

        checkCudaErrors(cudaMemcpy(m_pDevInput, m_aHostInput, (inputCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_pDevFilter, m_aHostFilter, (filterCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfInput, channelsizeOfInput, rowsizeOfInput, colsizeOfInput));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              batchsizeOfWeight, channelsizeOfWeight, rowsizeOfWeight, colsizeOfWeight));

        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, m_padding[0], m_padding[1], m_stride[0], m_stride[1],
                                                   1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        /* WE CAN OBTAIN THE OUTPUT DIMENSION FROM cudnnGetConvolutionNdForwardOutputDim() FUNCTION
         * BUT, THESE ALREADY EXIST IN OUR MODEL*/
        // cudnnGetConvolutionNdForwardOutputDim( ... )
        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        /* FIND THE BEST ALGORITHM ACCORDING TO PREFERENCE */
        // CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(this->GetCudnnHandle(), inputTensorDesc, filterDesc, convDesc, outputTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &algo));

        size_t sizeInBytes  = 0;
        void  *devWorkSpace = NULL;

        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(this->GetCudnnHandle(), inputTensorDesc, filterDesc, convDesc,
                                                           outputTensorDesc, algo, &sizeInBytes));

        if (sizeInBytes != 0) {
            checkCudaErrors(cudaMalloc(&devWorkSpace, sizeInBytes));

            if (devWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                return FALSE;
            }
        }

        checkCUDNN(cudnnConvolutionForward(this->GetCudnnHandle(), &alpha, inputTensorDesc, m_pDevInput, filterDesc, m_pDevFilter, convDesc,
                                           algo, devWorkSpace, sizeInBytes, &beta, outputTensorDesc, m_pDevOutput));

        checkCudaErrors(cudaMemcpy(m_aHostOutput, m_pDevOutput, (outputCapacity * sizeof(DTYPE)), cudaMemcpyDeviceToHost));

        for (int i = 0; i < outputCapacity; i++) {
            (*result)[i] = m_aHostOutput[i];
        }

        if (sizeInBytes != 0) {
            checkCudaErrors(cudaFree(devWorkSpace));
        }

        checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

    int ComputeBackPropagateOnGPU() {
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Shape *shapeOfInput        = input->GetShape();

        Tensor<DTYPE> *weight          = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[1]->GetGradient();
        Shape *shapeOfWeight           = weight->GetShape();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfResult      = this_delta->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int batchsizeOfWeight   = (*shapeOfWeight)[1];
        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

        int batchsizeOfInput   = (*shapeOfInput)[1];
        int channelsizeOfInput = (*shapeOfInput)[2];
        int rowsizeOfInput     = (*shapeOfInput)[3];
        int colsizeOfInput     = (*shapeOfInput)[4];

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        cudnnConvolutionBwdFilterAlgo_t filterAlgo;
        cudnnConvolutionBwdDataAlgo_t   dataAlgo;
        DTYPE alpha = 1;
        DTYPE beta  = 0;

        int inputCapacity       = input->GetCapacity();
        int filterCapacity      = weight->GetCapacity();
        int inputDeltaCapacity  = input->GetCapacity();
        int deltaCapacity       = this_delta->GetCapacity();
        int filterDeltaCapacity = weight_gradient->GetCapacity();

        for (int i = 0; i < inputCapacity; i++) {
            m_aHostInput[i] = (*input)[i];
        }

        for (int i = 0; i < filterCapacity; i++) {
            m_aHostFilter[i] = (*weight)[i];
        }

        for (int i = 0; i < deltaCapacity; i++) {
            m_aHostDelta[i] = (*this_delta)[i];
        }

        checkCudaErrors(cudaMemcpy(m_pDevInput, m_aHostInput, (inputCapacity) * sizeof(DTYPE), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_pDevFilter, m_aHostFilter, (filterCapacity) * sizeof(DTYPE), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_pDevDelta, m_aHostDelta, (deltaCapacity) * sizeof(DTYPE), cudaMemcpyHostToDevice));

        // printf("\n***** DEVICE VARIABLES ARE COPIED FROM HOST TO DEVICE *****\n");

        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfInput, channelsizeOfInput, rowsizeOfInput, colsizeOfInput));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              batchsizeOfWeight, channelsizeOfWeight, rowsizeOfWeight, colsizeOfWeight));

        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, m_padding[0], m_padding[1], m_stride[0], m_stride[1],
                                                   1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        /* WE CAN OBTAIN THE OUTPUT DIMENSION FROM cudnnGetConvolutionNdForwardOutputDim() FUNCTION
         * BUT, THESE ALREADY EXIST IN OUR MODEL*/
        checkCUDNN(cudnnSetTensor4dDescriptor(deltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfInput, channelsizeOfInput, rowsizeOfInput, colsizeOfInput));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDeltaDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              batchsizeOfWeight, channelsizeOfWeight, rowsizeOfWeight, colsizeOfWeight));

        /* FIND THE BEST ALGORITHM ACCORDING TO PREFERENCE */
        // CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
        // CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
        checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(this->GetCudnnHandle(), filterDesc, deltaDesc, convDesc, inputDeltaDesc,
                                                            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, 0, &dataAlgo));

        checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->GetCudnnHandle(), inputTensorDesc, deltaDesc, convDesc, filterDeltaDesc,
                                                              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, 0, &filterAlgo));

        // printf("\n***** CUDA INITIALIZATION SUCCESS *****\n");
        size_t dataSizeInBytes = 0; size_t filterSizeInBytes = 0;
        void  *dataDevWorkSpace = NULL; void *filterDevWorkSpace = NULL;

        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(this->GetCudnnHandle(), filterDesc, deltaDesc, convDesc, inputDeltaDesc, dataAlgo, &dataSizeInBytes));
        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(this->GetCudnnHandle(), inputTensorDesc, deltaDesc, convDesc, filterDesc, filterAlgo, &filterSizeInBytes));

        if (dataSizeInBytes != 0) {
            checkCudaErrors(cudaMalloc(&dataDevWorkSpace, dataSizeInBytes));

            if (dataDevWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                return FALSE;
            }
        }

        if (filterSizeInBytes != 0) {
            checkCudaErrors(cudaMalloc(&filterDevWorkSpace, filterSizeInBytes));

            if (filterDevWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                return FALSE;
            }
        }

        checkCUDNN(cudnnConvolutionBackwardData(this->GetCudnnHandle(), &alpha, filterDesc, m_pDevFilter, deltaDesc, m_pDevDelta, convDesc,
                                                dataAlgo, dataDevWorkSpace, dataSizeInBytes, &beta, inputDeltaDesc, m_pDevInputDelta));

        checkCUDNN(cudnnConvolutionBackwardFilter(this->GetCudnnHandle(), &alpha, inputTensorDesc, m_pDevInput, deltaDesc, m_pDevDelta, convDesc,
                                                  filterAlgo, filterDevWorkSpace, filterSizeInBytes, &beta, filterDesc, m_pDevFilterDelta));

        checkCudaErrors(cudaMemcpy(m_aHostInputDelta, m_pDevInputDelta, (inputDeltaCapacity) * sizeof(DTYPE), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(m_aHostFilterDelta, m_pDevFilterDelta, (filterDeltaCapacity) * sizeof(DTYPE), cudaMemcpyDeviceToHost));

        // Device to mem
        for (int i = 0; i < inputDeltaCapacity; i++) {
            (*input_delta)[i] += m_aHostInputDelta[i];
        }

        for (int i = 0; i < filterCapacity; i++) {
            (*weight_gradient)[i] += m_aHostFilterDelta[i];
        }

        if (dataSizeInBytes != 0) {
            checkCudaErrors(cudaFree(dataDevWorkSpace));
        }

        if (filterSizeInBytes != 0) {
            checkCudaErrors(cudaFree(filterDevWorkSpace));
        }

        return TRUE;
    }

#endif  // if __CUDNN__

    int* GetStrideList() {
        return m_stride;
    }

    int* GetPaddingList() {
        return m_padding;
    }
};


#endif  // CONVOLUTION_H_
