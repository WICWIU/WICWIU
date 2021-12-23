#ifndef ADD_H_
#define ADD_H_    value

#include "../Operator.hpp"

/*!
@class Addall Tensor의 모든 값들을 서로 더하는 class
*/
template<typename DTYPE> class Addall : public Operator<DTYPE>{
private:
    Shape *m_pLeftTenShape;
    ///< 연산 할 Tensor의 Shape
    Shape *m_pRightTenShape;
    ///< 연산 할 Tensor의 Shape

    int m_timesize;
    ///< time의 dimension 크기
    int m_batchsize;
    ///< batch의 dimension 크기
    int m_channelsize;
    ///< channel의 dimension 크기
    int m_rowsize;
    ///< row의 dimension 크기
    int m_colsize;
    ///< col의 dimension 크기


#ifdef __CUDNN__
    cudnnTensorDescriptor_t leftTensorDesc, rightTensorDesc, outputTensorDesc, leftDeltaDesc, rightDeltaDesc, deltaDesc;
    ///< GPU내의 Tensor값들을 가르키기 위한 descriptor.
    DTYPE *m_pDevLeft, *m_pDevRight, *m_pDevOutput, *m_pDevLeftDelta, *m_pDevRightDelta, *m_pDevDelta;
    ///< cudnn 연산에서 사용 할 데이터를 가리키는 맴버 변수.

    DTYPE m_alpha;
    ///<  연산 간 두 Operand의 가중치를 표현하기 한 변수. ex) z = α*x + β*y
    DTYPE m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 귀한 변수. ex) z = α*x + β*y

#endif  // __CUDNN__

public:
    /*!
    @brief Addall의 생성자
    @details pLeftInput, pRightInput을 Alloc시킨다.
    @param pLeftInput Alloc할 대상 Operator.
    @param pRightInput Alloc할 대상 Operator.
    @param pName Operator에 사용자가 부여한 이름.
    @ref int Alloc(Operator<DTYPE> *pLeftInput, Operator<DTYPE> *pRightInput)
    */
    Addall(Operator<DTYPE> *pLeftInput, Operator<DTYPE> *pRightInput, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pLeftInput, pRightInput, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "Addall::Addall(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pLeftInput, pRightInput);
    }

    /*!
    @brief Addall의 소멸자.
    */
    ~Addall() {
        #ifdef __DEBUG__
        std::cout << "Addall::~Addall()" << '\n';
        #endif  // __DEBUG__
    }

    /*!
    @brief 파라미터로 들어온 pLeftInput을 이용해 맴버 변수들을 초기화한다.
    @details 파라미터로 들어온 pLeftInput과 m_pRightInput의 Shape정보를 맴버변수에 저장하고 다른 맴버 변수들은 pLeftInput의 Shape값으로 초기화한다.
    @details Result값과 gradient값을 저장 할 Tensor를 새로 만든다.
    @param 생성 할 Tensor의 Shape정보를 가진 Operator
    @param pRightInput 연산에 사용 할 inputTensor.
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE> *pLeftInput, Operator<DTYPE> *pRightInput) {
        #ifdef __DEBUG__
        std::cout << "Addall::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        m_pLeftTenShape  = pLeftInput->GetResult()->GetShape();
        m_pRightTenShape = pRightInput->GetResult()->GetShape();

        m_timesize    = (*m_pLeftTenShape)[0];
        m_batchsize   = (*m_pLeftTenShape)[1];
        m_channelsize = (*m_pLeftTenShape)[2];
        m_rowsize     = (*m_pLeftTenShape)[3];
        m_colsize     = (*m_pLeftTenShape)[4];

        this->SetResult(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        this->SetGradient(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief cudnn을 사용하기 전 관련 맴버변수들을 초기화 한다.
    @details TensorDesriptor들을 생성하고, TensorDesriptor들의 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
    @param idOfDevice 사용할 GPU의 id
    */
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        m_alpha = 1;
        m_beta  = 0;

        checkCUDNN(cudnnCreateTensorDescriptor(&leftTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&rightTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&leftDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&rightDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&deltaDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(leftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(rightTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(leftDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(rightDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(deltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        // checkCudaErrors(cudaDeviceSynchronize());
    }

#endif  // if __CUDNN__

    /*!
    @brief 메모리를 헤제하는 Delete 메소드.
    @details cudnnDescriptor들을 GPU메모리에서 해제하고 포인터를 null로 초기화한다.
    */
    void Delete() {
#ifdef __CUDNN__

        if (leftTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(leftTensorDesc));
        leftTensorDesc = NULL;

        if (rightTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(rightTensorDesc));
        rightTensorDesc = NULL;

        if (outputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
        outputTensorDesc = NULL;

        if (leftDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(leftDeltaDesc));
        leftDeltaDesc = NULL;

        if (rightDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(rightDeltaDesc));
        rightDeltaDesc = NULL;

        if (deltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(deltaDesc));
        deltaDesc = NULL;

        // checkCudaErrors(cudaDeviceSynchronize());
#endif  // if __CUDNN__
    }

    /*!
    @brief Addall의 forwardPropagate 매소드.
    @details Container에 저장한 left, right의 Result값을 서로 더해 result에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0으로 사용한다.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left   = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *right  = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int m_ti = pTime;

        for (int m_ba = 0; m_ba < m_batchsize; m_ba++) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*result)[Index5D(m_pLeftTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                            = (*left)[Index5D(m_pLeftTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                              + (*right)[Index5D(m_pRightTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];
                    }
                }
            }
        }

        return TRUE;
    }

    /*!
    @brief Addall의 BackPropagate 매소드.
    @details Container에 저장한 pLeftInput, pRightInput의 Gradient값에 계산한 Gradient값을 각각 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0으로 사용한다.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left_grad  = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *right_grad = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        int m_ti = pTime;

        for (int m_ba = 0; m_ba < m_batchsize; m_ba++) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*left_grad)[Index5D(m_pLeftTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                            += (*this_grad)[Index5D(m_pLeftTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];

                        (*right_grad)[Index5D(m_pRightTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                            += (*this_grad)[Index5D(m_pLeftTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];
                    }
                }
            }
        }

        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief GPU에서 동작하는 Addall ForwardPropagate 메소드.
    @details cudnnAddTensor를 이용히여  m_pDevLeft, m_pDevRight의 값을 m_pDevOutput에 더해 넣는다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int ForwardPropagateOnGPU(int pTime) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left   = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *right  = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevLeft   = left->GetGPUData(pTime);
        m_pDevRight  = right->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &m_alpha, leftTensorDesc, m_pDevLeft,
                                  &m_alpha, outputTensorDesc, m_pDevOutput));

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &m_alpha, rightTensorDesc, m_pDevRight,
                                  &m_alpha, outputTensorDesc, m_pDevOutput));

        return TRUE;
    }

    /*!
    @brief GPU에서 동작하는 Addall BackwardPropagate 메소드.
    @details cudnnAddTensor를 이용히여 m_pDevDelta값을 m_pDevLeftDelta, m_pDevRightDelta에 각각 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int BackPropagateOnGPU(int pTime) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left_grad  = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *right_grad = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        m_pDevLeftDelta  = left_grad->GetGPUData(pTime);
        m_pDevRightDelta = right_grad->GetGPUData(pTime);
        m_pDevDelta      = this_grad->GetGPUData(pTime);

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &m_alpha, deltaDesc, m_pDevDelta,
                                  &m_alpha, leftDeltaDesc, m_pDevLeftDelta));

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &m_alpha, deltaDesc, m_pDevDelta,
                                  &m_alpha, rightDeltaDesc, m_pDevRightDelta));


        return TRUE;
    }

#endif  // __CUDNN__
};

/*!
@class AddColWise Tensor의 값 중 Colunm에만 값을 더하는 class
*/
template<typename DTYPE> class AddColWise : public Operator<DTYPE>{
private:
    Shape *m_pInputTenShape;
    ///< 더해질 Tensor의 Shape.
    Shape *m_pBiasTenShape;
    ///< 더할 Bias의 Shape.

    int m_timesize;
    ///< timetime
    int m_batchsize;
    ///< batchbatch
    int m_channelsize;
    ///< channelchannel
    int m_rowsize;
    ///< rowrow
    int m_colsize;
    ///< colcol


#ifdef __CUDNN__
    cudnnTensorDescriptor_t inputTensorDesc, biasTensorDesc, outputTensorDesc, inputDeltaDesc, biasDeltaDesc, deltaDesc;
    ///<  GPU내의 Tensor값들을 가르키기 위한 descriptor.
    DTYPE *m_pDevInput, *m_pDevBias, *m_pDevOutput, *m_pDevInputDelta, *m_pDevBiasDelta, *m_pDevDelta;
    ///<  cudnn 연산에서 사용 할 데이터를 가리키는 맴버 변수.

    cudnnReduceTensorDescriptor_t reduceTensorDesc;
    DTYPE *m_pDevIndices, *m_pDevWorkspace;
    size_t m_pDevIndicesSizeInBytes, m_pDevWorkspaceSizeInBytes;

    DTYPE m_alpha;
    ///<  연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    DTYPE m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y

#endif  // __CUDNN__

public:
    /*!
    @brief AddColWise의 생성자
    @details pInput, pBias을 Alloc시킨다.
    @param pInput Alloc할 대상 Operator.
    @param pBais Alloc할 대상 Operator.
    @param pName Operator에 사용자가 부여한 이름.
    @ref int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias)
    */
    AddColWise(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pBias, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "AddColWise::AddColWise(Operator<DTYPE> *, Operator<DTYPE> *, std::string, int)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pBias);
    }

    /*!
    @brief AddColWise의 소멸자
    */
    ~AddColWise() {
        #ifdef __DEBUG__
        std::cout << "AddColWise::~AddColWise()" << '\n';
        #endif  // __DEBUG__
    }

    /*!
    @brief 파라미터로 들어온 pInput, pBias를 이용해 맴버 변수들을 초기화 한다
    @details 파라미터로 들어온 pInput, pBias의 Shape정보를 맴버 변수에 저장하고 다른 맴버 변수들은 pInput의 Shape값으로 초기화 한다.
    @details Result값과 gradient값을 저장 할 Tensor를 새로 만든다.
    @param 생성 할 Tensor의 Shape정보를 가진 Operator.
    @param pBias 더할 Operator.
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias) {
        #ifdef __DEBUG__
        std::cout << "AddColWise::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        m_pInputTenShape = pInput->GetResult()->GetShape();
        m_pBiasTenShape  = pBias->GetResult()->GetShape();

        m_timesize    = (*m_pInputTenShape)[0];
        m_batchsize   = (*m_pInputTenShape)[1];
        m_channelsize = (*m_pInputTenShape)[2];
        m_rowsize     = (*m_pInputTenShape)[3];
        m_colsize     = (*m_pInputTenShape)[4];

        #ifdef __DEBUG__

        if ((*m_pBiasTenShape)[0] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[1] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[2] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[3] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);
        #endif  // __DEBUG__

        this->SetResult(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        this->SetGradient(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief cudnn을 사용하기 전 관련 맴버변수들을 초기화 한다.
    @details TensorDesriptor들을 생성하고, TensorDesriptor들의 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
    @param idOfDevice 사용할 GPU의 id
    */
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        m_alpha = 1;
        m_beta  = 0;

        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&inputDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&deltaDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              1, 1, 1, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(biasDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              1, 1, 1, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(deltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        if (m_batchsize == 1 && m_channelsize == 1 && m_rowsize == 1) {
            reduceTensorDesc = NULL;
            m_pDevIndices = NULL;
            m_pDevWorkspace = NULL;
            m_pDevIndicesSizeInBytes = 0;
            m_pDevWorkspaceSizeInBytes = 0;
        }
        else {
            checkCUDNN(cudnnCreateReduceTensorDescriptor(&reduceTensorDesc));
            checkCUDNN(cudnnSetReduceTensorDescriptor(reduceTensorDesc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

            checkCUDNN(cudnnGetReductionIndicesSize(this->GetCudnnHandle(), reduceTensorDesc, deltaDesc, biasDeltaDesc, &m_pDevIndicesSizeInBytes));
            checkCUDNN(cudnnGetReductionWorkspaceSize(this->GetCudnnHandle(), reduceTensorDesc, deltaDesc, biasDeltaDesc, &m_pDevWorkspaceSizeInBytes));

            checkCudaErrors(cudaMalloc(&m_pDevIndices, m_pDevIndicesSizeInBytes));
            checkCudaErrors(cudaMalloc(&m_pDevWorkspace, m_pDevWorkspaceSizeInBytes));
        }



        // checkCudaErrors(cudaDeviceSynchronize());
    }

#endif  // if __CUDNN__

    /*!
    @brief 메모리를 헤제하는 Delete 메소드.
    @details cudnnDescriptor들을 GPU메모리에서 해제하고 포인터를 null로 초기화한다.
    */
    void Delete() {
#ifdef __CUDNN__

        if (inputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
        inputTensorDesc = NULL;

        if (biasTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
        biasTensorDesc = NULL;

        if (outputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
        outputTensorDesc = NULL;

        if (inputDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(inputDeltaDesc));
        inputDeltaDesc = NULL;

        if (biasDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(biasDeltaDesc));
        biasDeltaDesc = NULL;

        if (deltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(deltaDesc));
        deltaDesc = NULL;

        // checkCudaErrors(cudaDeviceSynchronize());
#endif  // if __CUDNN__
    }

    /*!
    @brief AddColWise의 forwardPropagate 매소드.
    @details Container에 저장한 Input과 bias의 Colunm값을 서로 더해 result에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0으로 사용한다.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *bias   = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int m_ti = pTime;

        for (int m_ba = 0; m_ba < m_batchsize; m_ba++) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*result)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                            = (*input)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                              + (*bias)[m_co];
                    }
                }
            }
        }


        return TRUE;
    }

    /*!
    @brief AddColWise의 BackPropagate 매소드.
    @details Container에 저장한 pInput, pBias의 Gradient값에 계산을 통해 구한 gradient값을 각각 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0으로 사용한다.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input_grad = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *bias_grad  = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        int m_ti = pTime;

        for (int m_ba = 0; m_ba < m_batchsize; m_ba++) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*input_grad)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                            += (*this_grad)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];

                        (*bias_grad)[m_co]
                            += (*this_grad)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];
                    }
                }
            }
        }


        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief GPU에서 동작하는 AddColWise ForwardPropagate 메소드.
    @details cudnnAddTensor를 이용히여  m_pDevInput, m_pDevBias의 값을 더하여 m_pDevOutput에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int ForwardPropagateOnGPU(int pTime) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *bias   = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevBias   = bias->GetGPUData(0);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &m_alpha, inputTensorDesc, m_pDevInput,
                                  &m_alpha, outputTensorDesc, m_pDevOutput));

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &m_alpha, biasTensorDesc, m_pDevBias,
                                  &m_alpha, outputTensorDesc, m_pDevOutput));

        return TRUE;
    }

    /*!
    @brief GPU에서 동작하는 AddColWise BackwardPropagate 메소드.
    @details cudnnAddTensor를 이용히여 m_pDevDelta값을 m_pDevInputDelta, m_pDevBiasDelta에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int BackPropagateOnGPU(int pTime) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input_grad = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *bias_grad  = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        m_pDevInputDelta = input_grad->GetGPUData(pTime);
        m_pDevBiasDelta  = bias_grad->GetGPUData(0);
        m_pDevDelta      = this_grad->GetGPUData(pTime);

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &m_alpha, deltaDesc, m_pDevDelta,
                                  &m_alpha, inputDeltaDesc, m_pDevInputDelta));

        if (m_batchsize == 1 && m_channelsize == 1 && m_rowsize == 1) {
            checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &m_alpha, deltaDesc, m_pDevDelta,
                                  &m_beta, biasDeltaDesc, m_pDevBiasDelta));
        }
        else {
            checkCUDNN(cudnnReduceTensor(this->GetCudnnHandle(), reduceTensorDesc,
                                m_pDevIndices, m_pDevIndicesSizeInBytes,
                                m_pDevWorkspace, m_pDevWorkspaceSizeInBytes,
                                &m_alpha, deltaDesc, m_pDevDelta,
                                &m_beta, biasDeltaDesc, m_pDevBiasDelta));
        }

        return TRUE;
    }

#endif  // __CUDNN__
};

/*!
@class  AddChannelWise Tensor의 channel값만 서로 더하는 class
*/
template<typename DTYPE> class AddChannelWise : public Operator<DTYPE>{
private:
    Shape *m_pInputTenShape;
    ///< 더해질 Tensor의 Shape
    Shape *m_pBiasTenShape;
    ///< 더 할 Tensor의 Shape

    int m_timesize;
    ///< time의 dimension 크기
    int m_batchsize;
    ///<  batch의 dimension 크기
    int m_channelsize;
    ///< channe의 dimension 크기
    int m_rowsize;
    ///<  row의 dimension 크기
    int m_colsize;
    ///< col의 dimension 크기

#ifdef __CUDNN__
    cudnnTensorDescriptor_t inputTensorDesc, biasTensorDesc, outputTensorDesc, inputDeltaDesc, biasDeltaDesc, deltaDesc;
    ///<  GPU내의 Tensor값들을 가르키기 위한 descriptor.
    DTYPE *m_pDevInput, *m_pDevBias, *m_pDevOutput, *m_pDevInputDelta, *m_pDevBiasDelta, *m_pDevDelta;
    ///<  cudnn 연산에서 사용 할 데이터를 가리키는 맴버 변수.

    DTYPE m_alpha;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    DTYPE m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y

#endif  // __CUDNN__

public:
    /*!
    @brief AddChannelWise의 생성자
    @details pInput, pBias을 Alloc시킨다.
    @param pInput Alloc할 대상 Operator.
    @param pBias Alloc할 대상 Operator.
    @param pName Operator에 사용자가 부여한 이름.
    @ref int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias)
    */
    AddChannelWise(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pBias, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "AddChannelWise::AddChannelWise(Operator<DTYPE> *, Operator<DTYPE> *, std::string, int)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pBias);
    }

    /*!
    @brief AddChannelWise의 소멸자.
    */
    ~AddChannelWise() {
        #ifdef __DEBUG__
        std::cout << "AddChannelWise::~AddChannelWise()" << '\n';
        #endif  // __DEBUG__
    }

    /*!
    @brief 파라미터로 들어온 pInput, pBias를 이용해 맴버 변수들을 초기화 한다
    @details 파라미터로 들어온 pInput, pBias의 Shape정보를 맴버 변수에 저장하고 다른 맴버 변수들은 pInput의 Shape값으로 초기화 한다.
    @details Result값과 gradient값을 저장 할 Tensor를 새로 만든다.
    @param 생성 할 Tensor의 Shape정보를 가진 Operator.
    @param pBias 더할 Operator.
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias) {
        #ifdef __DEBUG__
        std::cout << "AddColWise::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        m_pInputTenShape = pInput->GetResult()->GetShape();
        m_pBiasTenShape  = pBias->GetResult()->GetShape();

        m_timesize    = (*m_pInputTenShape)[0];
        m_batchsize   = (*m_pInputTenShape)[1];
        m_channelsize = (*m_pInputTenShape)[2];
        m_rowsize     = (*m_pInputTenShape)[3];
        m_colsize     = (*m_pInputTenShape)[4];

        #ifdef __DEBUG__

        if ((*m_pBiasTenShape)[0] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[1] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[3] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[4] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);
        #endif  // __DEBUG__

        this->SetResult(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        this->SetGradient(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        return TRUE;
    }

    #ifdef __CUDNN__
    /*!
    @brief cudnn을 사용하기 전 관련 맴버변수들을 초기화 한다.
    @details TensorDesriptor들을 생성하고, TensorDesriptor들의 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
    @param idOfDevice 사용할 GPU의 id
    */
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        m_alpha = 1;
        m_beta  = 0;

        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&inputDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&deltaDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              1, m_channelsize, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(biasDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              1, m_channelsize, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(deltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_batchsize, m_channelsize, m_rowsize, m_colsize));

        // checkCudaErrors(cudaDeviceSynchronize());
    }

    #endif  // if __CUDNN__

    /*!
    @brief 메모리를 헤제하는 Delete 메소드.
    @details cudnnDescriptor들을 GPU메모리에서 해제하고 포인터를 null로 초기화한다.
    */
    void Delete() {
    #ifdef __CUDNN__

        if (inputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
        inputTensorDesc = NULL;

        if (biasTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
        biasTensorDesc = NULL;

        if (outputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
        outputTensorDesc = NULL;

        if (inputDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(inputDeltaDesc));
        inputDeltaDesc = NULL;

        if (biasDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(biasDeltaDesc));
        biasDeltaDesc = NULL;

        if (deltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(deltaDesc));
        deltaDesc = NULL;

        // checkCudaErrors(cudaDeviceSynchronize());
    #endif  // if __CUDNN__
    }

    /*!
    @brief AddChannelWise의 forwardPropagate 매소드.
    @details Container에 저장한 Input, bias의 Channel값을 Result에 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0으로 사용한다.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *bias   = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int m_ti = pTime;

        for (int m_ba = 0; m_ba < m_batchsize; m_ba++) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*result)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                            = (*input)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                              + (*bias)[m_ch];
                    }
                }
            }
        }


        return TRUE;
    }

    /*!
    @brief AddColWise의 BackPropagate 매소드.
    @details Container에 저장한 pInput, pBias의 Gradient값애 계산을 통해 구한 gradient값을 각각 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0으로 사용한다.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input_grad = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *bias_grad  = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        int m_ti = pTime;

        for (int m_ba = 0; m_ba < m_batchsize; m_ba++) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*input_grad)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                            += (*this_grad)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];

                        (*bias_grad)[m_ch]
                            += (*this_grad)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];
                    }
                }
            }
        }

        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief GPU에서 동작하는 AddChannelWise ForwardPropagate 메소드.
    @details cudnnAddTensor를 이용히여  m_pDevInput, m_pDevBias값을 더하여 m_pDevOutput에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int ForwardPropagateOnGPU(int pTime) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *bias   = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevBias   = bias->GetGPUData(0);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &m_alpha, inputTensorDesc, m_pDevInput,
                                  &m_beta, outputTensorDesc, m_pDevOutput));

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &m_alpha, biasTensorDesc, m_pDevBias,
                                  &m_alpha, outputTensorDesc, m_pDevOutput));

        // this->ForwardPropagate();
        return TRUE;
    }

    /*!
    @brief GPU에서 동작하는 AddChannelWise BackwardPropagate 메소드.
    @details cudnnAddTensor를 이용히여  m_pDevDelta값을 m_pDevOutput, m_pDevBiasDelta에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int BackPropagateOnGPU(int pTime) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input_grad = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *bias_grad  = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        m_pDevInputDelta = input_grad->GetGPUData(pTime);
        m_pDevBiasDelta  = bias_grad->GetGPUData(0);
        m_pDevDelta      = this_grad->GetGPUData(pTime);

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &m_alpha, deltaDesc, m_pDevDelta,
                                  &m_alpha, inputDeltaDesc, m_pDevInputDelta));

        checkCUDNN(cudnnConvolutionBackwardBias(this->GetCudnnHandle(),
                                                &m_alpha, deltaDesc, m_pDevDelta,
                                                &m_alpha, biasDeltaDesc, m_pDevBiasDelta))

        // this->BackPropagate();

        return TRUE;
    }

#endif  // __CUDNN__
};


#endif  // ADD_H_