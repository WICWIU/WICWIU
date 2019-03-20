#ifndef MAXPOOLING_H_
#define MAXPOOLING_H_    value

#include "../Operator.hpp"

template<typename DTYPE> class Maxpooling2D : public Operator<DTYPE>{
private:
    int m_stride[2];
    ///< stride 값
    int m_mask[2];
    ///< mask size
    int m_padding[2];
    ///< padding size

    Tensor<int> *indexOfMaxInput;
    ///< Backproagate할 떄 delta값을 흘려보낼 index를 기억하기 위한 맴버변수.

#ifdef __CUDNN__
    cudnnTensorDescriptor_t m_aInputTensorDesc, m_aOutputTensorDesc, m_aDeltaDesc, m_aInputDeltaDesc;
    ///< GPU내의 Tensor값들을 가르키기 위한 descriptor.
    cudnnPoolingDescriptor_t m_aPoolingDesc;
    ///< pooling연산 descriptor 구조체를 가라키는 포인터.
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;
    ///<  cudnn 연산에서 사용 할 데이터를 가리키는 맴버 변수.
    // DTYPE *m_aHostInput, *m_aHostOutput, *m_aHostInputDelta, *m_aHostDelta;

    float m_alpha;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    float m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    double m_coef;
    ///< unUsed Variable


#endif  // __CUDNN__

public:
    /*!
    @brief Maxpooling2D의 생성자
    @details 파라미터로 받은 pInput, strideRow, strideCol, maskRow, maskCol으로 Alloc한다.
    @param pInput Maxpooling2D할 대상 Operator.
    @param maskRow Filter의 Row size
    @param maskCol Filter의 Colunm size
    @param strideRow Row stirde값
    @param strideCol Colunm stride값
    @param pName 사용자가 부여한 Operator이름
    */
    Maxpooling2D(Operator<DTYPE> *pInput, int maskRow, int maskCol, int strideRow, int strideCol, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "Maxpooling2D::Maxpooling2D(Operator<DTYPE> *, int, int)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, strideRow, strideCol, maskRow, maskCol);
    }

    /*!
    @brief Maxpooling2D의 생성자
    @details 파라미터로 받은 pInput, strideRow, strideCol, maskRow, maskCol으로 Alloc한다.
    @param pInput Maxpooling2D할 대상 Operator.
    @param maskRow Filter의 Row size
    @param maskCol Filter의 Colunm size
    @param strideRow Row stirde값
    @param strideCol Colunm stride값
    @param padding padding size
    @param pName 사용자가 부여한 Operator이름
    */
    Maxpooling2D(Operator<DTYPE> *pInput, int maskRow, int maskCol, int strideRow, int strideCol, int padding, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "Maxpooling2D::Maxpooling2D(Operator<DTYPE> *, int, int, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, strideRow, strideCol, maskRow, maskCol, padding, padding);
    }

    /*!
    @brief Maxpooling2D의 소멸자
    */
    ~Maxpooling2D() {
        #ifdef __DEBUG__
        std::cout << "Maxpooling2D::~Maxpooling2D()" << '\n';
        #endif  // __DEBUG__
#ifdef __CUDNN__
        Delete();
#endif  // if __CUDNN__
    }

    /*!
    @brief 파라미터로 받은 변수로부터 맴버 변수들을 초기화 한다.
    @details Result와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
    @details m_stride, m_mask, m_padding값들을 초기화 하고 rowsize, colsize를 결정한다.
    @param pInput 생성 할 Tensor의 Shape정보를 가진 Operator.
    @param strideRow Row stirde값.
    @param strideCol Colunm stride값.
    @param maskRow Filter의 Row size.
    @param maskCol Filter의 Colunm size.
    @param padding1 row_padding 값.
    @param padding2 col_padding 값.
    @return 성공 시 TRUE.
    */
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
    /*!
    @brief cudnn을 사용하기 전 관련 맴버변수들을 초기화 한다.
    @details Activation 함수를 Maxpooling2D로 지정한다.
    @details TensorDesriptor들을 생성하고, TensorDesriptor들으 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
    @param idOfDevice 사용할 GPU의 id
    */
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
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

    /*!
    @brief Delete 메소드
    @details cudnnDescriptor들을 GPU메모리에서 해제하고 포인터를 null로 초기화한다.
    */
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

        // checkCudaErrors(cudaDeviceSynchronize());

#endif  // if __CUDNN__
    }

    /*!
    @brief Maxpooling2D의 ForwardPropagate 매소드.
    @details Filter(temprow * tempcol)의 범위 중 가장 큰 값을 result에 저장하고, index는 indexOfMaxInput에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    // 메모리 효율을 생각하면 time에 따라 취해야 할 액션이 다르다.
    int ForwardPropagate(int pTime = 0) {
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

        for (int ba = 0; ba < batchsize; ba++) {
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

    /*!
    @brief Maxpooling2D의 BackPropagate 매소드.
    @details 계산한 delta값을 input_delta에 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfDelta       = this_delta->GetShape();

        int batchsize   = (*shapeOfDelta)[1];
        int channelsize = (*shapeOfDelta)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfDelta)[3];
        int colsize     = (*shapeOfDelta)[4];

        int indexOfDelta = 0;

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
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
    /*!
    @brief GPU에서 작동하는 ForwardPropagate 메소드.
    @details 지정한 Activation function(Maxpooling2D) functiondml ForwardPropagate연산을 실행한다.
    @details m_pDevOutput에 결과 값을 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int ForwardPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnPoolingForward(this->GetCudnnHandle(), m_aPoolingDesc, &m_alpha, m_aInputTensorDesc, m_pDevInput,
                                       &m_beta, m_aOutputTensorDesc, m_pDevOutput));


        // checkCudaErrors(cudaDeviceSynchronize());
        return TRUE;
    }

    /*!
    @brief GPU에서 작동하는 BackPropagate메소드.
    @details 지정한 Activation function(Maxpooling2D)의 BackPropagate연산을 실행한다.
    @details m_pDevDelta에 결과 값을 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
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
                                        &m_alpha, m_aInputDeltaDesc, m_pDevInputDelta));


        // checkCudaErrors(cudaDeviceSynchronize());
        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // MAXPOOLING_H_
