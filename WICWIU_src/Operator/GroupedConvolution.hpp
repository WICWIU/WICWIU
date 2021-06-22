#ifndef GROUPEDCONVOLUTION_H_
#define GROUPEDCONVOLUTION_H_    value

#include "../Operator.hpp"
#include <cstdio>

/*!
@class GroupedConvolution2D      GroupedConvolution2D class
*/
template<typename DTYPE> class GroupedConvolution2D : public Operator<DTYPE>{
private:
    int m_stride[2];
    ///< stride값. [0]은 row, [1]은 colunm을 각각 의미한다.
    int m_padding[2];
    ///< padding값 [0]은 height, [1]은 width를 각각 의미한다.
    int m_group;
    ///< Group 값을 지정하는 변수

#ifdef __CUDNN__
    cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc, deltaDesc, inputDeltaDesc;
    ///< GPU내의 Tensor값들을 가르키기 위한 descriptor.
    cudnnConvolutionDescriptor_t convDesc;
    ///< Convolution에 대한 description을 포함하는 구조체 포인터.
    cudnnFilterDescriptor_t filterDesc, filterDeltaDesc;
    ///<  필터 데이터셋을 가리키는 구조체 포인터.
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevFilter, *m_pDevInputDelta, *m_pDevDelta, *m_pDevFilterDelta;
    ///< cudnn 연산에서 사용 할 데이터를 가리키는 맴버 변수.

    cudnnConvolutionFwdAlgo_t m_algo;
    ///<  ForwardPropagate Convolution연산을 하기 위한 다양한 알고리즘을 제공하는 변수.
    cudnnConvolutionBwdFilterAlgo_t m_filterAlgo;
    ///< BackwardPropagate Convolution연산을 하기 위한 다양한 알고리즘을 제공하는 변수.
    cudnnConvolutionBwdDataAlgo_t m_dataAlgo;
    ///< BackwardPropagate Convolution연산을 하기 위한 다양한 알고리즘을 제공하는 변수.

    size_t m_sizeInBytes;
    ///< Convolution 연산에 필요한 메모리를 계산하여 저장하기 위한 맴버변수.
    size_t m_dataSizeInBytes;
    ///< Convolution 연산에 필요한 메모리를 계산하여 저장하기 위한 맴버변수.
    size_t m_filterSizeInBytes;
    ///< Convolution 연산에 필요한 메모리를 계산하여 저장하기 위한 맴버변수.

    DTYPE m_alpha;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    DTYPE m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y

    void *m_devWorkSpace;
    ///< Convolution 연산을 위해 할당받은 메모리 공간을 가리키는 포인터
    void *m_dataDevWorkSpace;
    ///< Convolution 연산을 위해 할당받은 메모리 공간을 가리키는 포인터
    void *m_filterDevWorkSpace;
    ///< Convolution 연산을 위해 할당받은 메모리 공간을 가리키는 포인터

#endif  // __CUDNN__

public:
    /*!
    @brief GroupedConvolution2D의 생성자.
    @details 파라미터로 받은 pInput, pWeight, stride1, stride2로 Alloc한다.
    @param pInput Convolution할 Operator
    @param pWeight Convolution할 weight.
    @param stride1 stride row값
    @param stride2 stride colunm값
    @param pName 사용자가 부여한 Operator이름.
    @ref int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding1, int padding2)
    */
    GroupedConvolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int group, std::string pName = "NO NAME", int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pWeight, pName, pLoadflag) {
        Alloc(pInput, pWeight, stride1, stride2, 0, 0, group);
    }

    /*!
    @brief GroupedConvolution2D의 생성자.
    @details 파라미터로 받은 pInput, pWeight, stride1, stride2, padding로 Alloc한다.
    @param pInput Convolution할 Operator
    @param pWeight Convolution할 weight.
    @param stride1 stride row값
    @param stride2 stride colunm값
    @param padding padding 할 값. height, width 모두 이 값으로 한다.
    @param pName 사용자가 부여한 Operator이름.
    @ref int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding1, int padding2)
    */
    GroupedConvolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding, int group, std::string pName = "NO NAME", int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pWeight, pName, pLoadflag) {
        Alloc(pInput, pWeight, stride1, stride2, padding, padding, group);
    }

    /*!
    @brief GroupedConvolution2D의 생성자.
    @details 파라미터로 받은 pInput, pWeight, stride1, stride2, padding1, padding2로 Alloc한다.
    @param pInput Convolution할 Operator
    @param pWeight Convolution할 weight.
    @param stride1 stride row값
    @param stride2 stride colunm값
    @param padding1 height padding값
    @param padding2 width padding값
    @param pName 사용자가 부여한 Operator이름.
    @ref int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding1, int padding2)
    */
    GroupedConvolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding1, int padding2, int group, std::string pName = "NO NAME", int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pWeight, pName, pLoadflag) {
        Alloc(pInput, pWeight, stride1, stride2, padding1, padding2, group);
    }

    /*!
    @brief GroupedConvolution2D의 소멸자.
    @details Delete매소드를 사용해 GPU에 할당했던 값들을 해제한다.
    @ref void Delete()
    */
    virtual ~GroupedConvolution2D() {
        #ifdef __DEBUG__
        std::cout << "GroupedConvolution2D::~GroupedConvolution2D()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    /*!
    @brief 파라미터로 받은 pInput, pWeight, stride1, stride2, padding1, padding2으로 맴버 변수들을 초기화 한다.
    @details pInput과 pWeight의 Shape과 stride, padding값으로 output으로 Result와 Delta로 사용 할 Tensor의 Shape을 정의한다.
    @param pInput Convolution할 Operator
    @param pWeight Convolution할 weight.
    @param stride1 stride row값
    @param stride2 stride colunm값
    @param padding1 height padding값
    @param padding2 width padding값
    */
    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding1, int padding2, int group) {
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
        
        this->m_group = group;

        outputHeight = ((*shapeOfInput)[3] - (*shapeOfWeight)[3] + (2 * m_padding[0])) / m_stride[0] + 1;
        outputWidth  = ((*shapeOfInput)[4] - (*shapeOfWeight)[4] + (2 * m_padding[1])) / m_stride[1] + 1;

        this->SetResult(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], outputHeight, outputWidth));
        
        this->SetDelta(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], outputHeight, outputWidth));
        
        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief cudnn을 사용하기 전 관련 맴버변수들을 초기화 한다.
    @details TensorDesriptor들을 생성하고, TensorDesriptor들의 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
    @details Convolution연산에 필요한 알고리즘을 정의하고, 연산에 필요한 메모리공간을 할당 받는다.
    @param idOfDevice 사용할 GPU의 id
    */
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
                                                   1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

        checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, this->m_group));

        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(this->GetCudnnHandle(), inputTensorDesc, filterDesc, convDesc, outputTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &m_algo));

        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(this->GetCudnnHandle(), inputTensorDesc, filterDesc, convDesc,
                                                           outputTensorDesc, m_algo, &m_sizeInBytes));

        checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(this->GetCudnnHandle(), filterDesc, deltaDesc, convDesc, inputDeltaDesc,
                                                            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &m_dataAlgo));

        checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->GetCudnnHandle(), inputTensorDesc, deltaDesc, convDesc, filterDeltaDesc,
                                                              CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &m_filterAlgo));

        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(this->GetCudnnHandle(), filterDesc, deltaDesc, convDesc, inputDeltaDesc, m_dataAlgo, &m_dataSizeInBytes));

        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(this->GetCudnnHandle(), inputTensorDesc, deltaDesc, convDesc, filterDeltaDesc, m_filterAlgo, &m_filterSizeInBytes));


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

    /*!
    @brief GPU에 할당했던 메모리를 해제하고 각 포인터들을 NULL로 초기화한다.
    @details inputTensorDesc, outputTensorDesc,deltaDesc, inputDeltaDesc, convDesc, filterDesc,filterDeltaDesc들을 삭제하고 NULL로 초기화한다.
    @details cudnn연산을 위해 할당 했던 메모리들을 해제시킨다.
    */
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

        return TRUE;
    }

        int BackPropagate(int pTime = 0) {
        
        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief GPU에서 동작하는 ForwardPropagate 메소드.
    @details cudnn이 제공하는 Convolution ForwardPropagate 메소드를 실행한다.
    @details Convolution의 ForwardPropagate결과는 m_pDevOutput에 저장된다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int ForwardPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevFilter = weight->GetGPUData(0);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnConvolutionForward(this->GetCudnnHandle(), &m_alpha, inputTensorDesc, m_pDevInput, filterDesc, m_pDevFilter, convDesc,
                                           m_algo, m_devWorkSpace, m_sizeInBytes, &m_beta, outputTensorDesc, m_pDevOutput));


        // checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

    /*!
    @brief GPU에서 동작하는 BackwardPropagate 메소드.
    @details cudnn이 제공하는 Convolution BackwardPropagate 메소드를 실행한다.
    @details Convolution의 BackwardPropagate결과는 m_pDevInputDelta와 m_pDevFilterDelta에 저장된다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int BackPropagateOnGPU(int pTime = 0) {
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

        checkCUDNN(cudnnConvolutionBackwardData(this->GetCudnnHandle(), &m_alpha, filterDesc, m_pDevFilter, deltaDesc, m_pDevDelta, convDesc,
                                                m_dataAlgo, m_dataDevWorkSpace, m_dataSizeInBytes, &m_alpha, inputDeltaDesc, m_pDevInputDelta));

        checkCUDNN(cudnnConvolutionBackwardFilter(this->GetCudnnHandle(), &m_alpha, inputTensorDesc, m_pDevInput, deltaDesc, m_pDevDelta, convDesc,
                                                  m_filterAlgo, m_filterDevWorkSpace, m_filterSizeInBytes, &m_alpha, filterDesc, m_pDevFilterDelta));

        // checkCudaErrors(cudaDeviceSynchronize());

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


#endif  // GROUPEDCONVOLUTION_H_
