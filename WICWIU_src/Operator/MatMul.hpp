#ifndef MATMUL_H_
#define MATMUL_H_    value

#include "../Operator.hpp"
#include <cstdio>

template<typename DTYPE> class MatMul : public Operator<DTYPE>{
private:
#ifdef __CUDNN__
    cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc, deltaDesc, inputDeltaDesc;
    ///< GPU내의 Tensor값들을 가르키기 위한 descriptor.
    cudnnConvolutionDescriptor_t convDesc;
    ///< Convolution에 대한 description을 포함하는 구조체 포인터.
    cudnnFilterDescriptor_t filterDesc, filterDeltaDesc;
    ///< 필터 데이터셋을 가리키는 구조체 포인터.
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevFilter, *m_pDevInputDelta, *m_pDevDelta, *m_pDevFilterDelta;
    ///<  cudnn 연산에서 사용 할 데이터를 가리키는 맴버 변수.
    // DTYPE *m_pHostInput, *m_pHostOutput, *m_pHostFilter, *m_pHostInputDelta, *m_pHostDelta, *m_pHostFilterDelta;

    cudnnConvolutionFwdAlgo_t m_algo;
    ///< ForwardPropagate Convolution연산을 하기 위한 다양한 알고리즘을 제공하는 변수.
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
    ///<  Convolution 연산을 위해 할당받은 메모리 공간을 가리키는 포인터
    void *m_filterDevWorkSpace;
    ///<  Convolution 연산을 위해 할당받은 메모리 공간을 가리키는 포인터
    int m_Loadflag;

#endif  // __CUDNN__

public:
    /*!
    @brief MatMul의 생성자.
    @details 파라미터로 받은 pWeight와 pInput으로 Alloc한다.
    @param pWeight MatMul할 weight.
    @param pInput Matmul할 input Operator.
    @param pName 사용자가 부여한 Operator이름.
    @ref int Alloc(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput)
    */
    MatMul(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput, std::string pName, int pLoadflag) : Operator<DTYPE>(pWeight, pInput, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "MatMul::MatMul(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pWeight, pInput, pLoadflag);
    }

    /*!
    @brief MatMul의 소멸자
    @details Delete매소드를 사용해 GPU에 할당했던 값들을 해제한다.
    @ref void Delete()
    */
    virtual ~MatMul() {
        #ifdef __DEBUG__
        std::cout << "Convolution2D::~Convolution2D()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    /*!
    @brief 파라미터로 받은 pWeight, pInput으로 맴버 변수들을 초기화 한다.
    @details timesize, batchsize, channelsize, row_size는 pInput의 Shape과 같게,  colsize는 pWeight와 같게 초기화한다.
    @details input x weight을 하기 때문에 rowsize는 pInput의 Shape을, colsize는 pWeight의 Shape을 받는다.
    @details Result와 Delta를 저장하기 위해 input의 rowsize, weight의 colsize를 갖는 Tensor를 생성한다.
    @param pWeight MatMul할 weight.
    @param pInput Matmul할 input Operator.
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput, int pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "MatMul::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pWeight->GetResult()->GetRowSize();

        m_Loadflag = pLoadflag;

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief cudnn을 사용하기 전 관련 맴버변수들을 초기화 한다.
    @details TensorDesriptor들을 생성하고, TensorDesriptor들의 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
    @details Convolution연산에 필요한 알고리즘을 정의하고, 연산에 필요한 메모리공간을 할당 받는다. MatMul은 Convolution연산을 이용한다.
    @param idOfDevice 사용할 GPU의 id
    */
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        Operator<DTYPE> *pWeight = this->GetInput()[0];
        Operator<DTYPE> *pInput  = this->GetInput()[1];

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pWeight->GetResult()->GetRowSize();

        int inputCapacity  = pInput->GetResult()->GetCapacity();
        int outputCapacity = this->GetResult()->GetCapacity();
        int filterCapacity = pWeight->GetResult()->GetCapacity();

        Shape *shapeOfWeight = pWeight->GetResult()->GetShape();
        Shape *shapeOfInput  = pInput->GetResult()->GetShape();
        Shape *shapeOfResult = this->GetResult()->GetShape();

        int rowsizeOfWeight = (*shapeOfWeight)[3];
        int colsizeOfWeight = (*shapeOfWeight)[4];

        int batchsizeOfInput = (*shapeOfInput)[1];
        int colsizeOfInput   = (*shapeOfInput)[4];

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
                                              batchsizeOfInput, 1, 1, colsizeOfInput));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              rowsizeOfWeight, 1, 1, colsizeOfWeight));

        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, colsize, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(deltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, colsize, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfInput, 1, 1, colsizeOfInput));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDeltaDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              rowsizeOfWeight, 1, 1, colsizeOfWeight));

        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1,
                                                   1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(this->GetCudnnHandle(), inputTensorDesc, filterDesc, convDesc, outputTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &m_algo));

        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(this->GetCudnnHandle(), inputTensorDesc, filterDesc, convDesc,
                                                           outputTensorDesc, m_algo, &m_sizeInBytes));

        checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(this->GetCudnnHandle(), filterDesc, deltaDesc, convDesc, inputDeltaDesc,
                                                            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &m_dataAlgo));

        checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->GetCudnnHandle(), inputTensorDesc, deltaDesc, convDesc, filterDeltaDesc,
                                                              CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &m_filterAlgo));

        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(this->GetCudnnHandle(), filterDesc, deltaDesc, convDesc, inputDeltaDesc, m_dataAlgo, &m_dataSizeInBytes));

        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(this->GetCudnnHandle(), inputTensorDesc, deltaDesc, convDesc, filterDesc, m_filterAlgo, &m_filterSizeInBytes));

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
    @details m_devWorkSpace, m_dataDevWorkSpace, m_filterDevWorkSpace들이 가리키는 메모리를 해제한다.
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

    /*!
    @brief MatMul의 ForwardPropagate매소드.
    @details weight의 각 row의 값들과 input의 Colunm의 각 값들을 곱하여 result에 더한다.
    @details [2 x 3] x [3 x 1]일때  3이 hiddensize
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *weight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int hiddensize = input->GetColSize();

        int weight_index = 0;
        int input_index  = 0;
        int result_index = 0;

        Shape *weightTenShape = weight->GetShape();
        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int hid = 0; hid < hiddensize; hid++) {
                            (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                                += (*weight)[Index5D(weightTenShape, 0, 0, 0, co, hid)]
                                   * (*input)[Index5D(inputTenShape, ti, ba, ch, ro, hid)];
                        }
                    }
                }
            }
        }


        return TRUE;
    }

    /*!
    @brief MatMul의 BackPropagate 매소드.
    @details input_delta에 weight * this_delta값을 더해주고,
    @details weight_gradient에는 input * this_delta값을 더해준다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *weight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();

        Tensor<DTYPE> *this_delta      = this->GetDelta();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *input_delta     = this->GetInput()[1]->GetDelta();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();
        int hiddensize  = input_delta->GetColSize();

        Shape *weightTenShape = weight->GetShape();
        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = this_delta->GetShape();

        int weight_index = 0;
        int input_index  = 0;
        int result_index = 0;

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int hid = 0; hid < hiddensize; hid++) {
                            weight_index = Index5D(weightTenShape, 0, 0, 0, co, hid);
                            input_index  = Index5D(inputTenShape, ti, ba, ch, ro, hid);
                            result_index = Index5D(resultTenShape, ti, ba, ch, ro, co);

                            (*input_delta)[input_index]      += (*weight)[weight_index] * (*this_delta)[result_index];
                            (*weight_gradient)[weight_index] += (*input)[input_index] * (*this_delta)[result_index];
                        }
                    }
                }
            }
        }


        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief GPU에서 동작하는 ForwardPropagate 메소드.
    @details cudnn이 제공하는 Convolution ForwardPropagate 메소드를 이용하여 MatMul의 ForwardPropagate연산을 한다.
    @details MatMul의 ForwardPropagate결과는 m_pDevOutput에 저장된다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int ForwardPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *weight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevFilter = weight->GetGPUData(0);
        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnConvolutionForward(this->GetCudnnHandle(), &m_alpha, inputTensorDesc, m_pDevInput, filterDesc, m_pDevFilter, convDesc,
                                           m_algo, m_devWorkSpace, m_sizeInBytes, &m_beta, outputTensorDesc, m_pDevOutput));

        // checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

    /*!
    @brief GPU에서 동작하는 BackwardPropagate 메소드.
    @details cudnn이 제공하는 Convolution BackwardPropagate 메소드를 아용하여 MatMul의 BackwardPropagate연산을 한다.
    @details Convolution의 BackwardPropagate결과는 m_pDevInputDelta와 m_pDevFilterDelta에 저장된다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int BackPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *weight          = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *input           = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *input_delta     = this->GetInput()[1]->GetDelta();
        Tensor<DTYPE> *this_delta      = this->GetDelta();

        m_pDevFilter      = weight->GetGPUData(0);
        m_pDevInput       = input->GetGPUData(pTime);
        m_pDevDelta       = this_delta->GetGPUData(pTime);
        m_pDevFilterDelta = weight_gradient->GetGPUData(0);
        m_pDevInputDelta  = input_delta->GetGPUData(pTime);

        checkCUDNN(cudnnConvolutionBackwardData(this->GetCudnnHandle(), &m_alpha, filterDesc, m_pDevFilter, deltaDesc, m_pDevDelta, convDesc,
                                                m_dataAlgo, m_dataDevWorkSpace, m_dataSizeInBytes, &m_alpha, inputDeltaDesc, m_pDevInputDelta));

        checkCUDNN(cudnnConvolutionBackwardFilter(this->GetCudnnHandle(), &m_alpha, inputTensorDesc, m_pDevInput, deltaDesc, m_pDevDelta, convDesc,
                                                  m_filterAlgo, m_filterDevWorkSpace, m_filterSizeInBytes, &m_alpha, filterDesc, m_pDevFilterDelta));

        // checkCudaErrors(cudaDeviceSynchronize());
        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // MATMUL_H_
