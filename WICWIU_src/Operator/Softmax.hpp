#ifndef SOFTMAX_H_
#define SOFTMAX_H_ value

#include "../Operator.hpp"

template <typename DTYPE>
class Softmax : public Operator<DTYPE> {
    DTYPE m_epsilon;
    ///< Softmax연산 중 더해지는 epsilon값.

    int m_timesize;
    ///< 연산 할 Tensor가 위치한 Time값.

    DTYPE **sum;
    ///< Softmax연산 중 Tensor값들의 합을 저장하기 위한 포인터.
    DTYPE **max;
    ///< Softmax연산 중 Tensor값들 중 가장 큰 값을 저장하기 위한 포인터.
#ifdef __CUDNN__
    cudnnTensorDescriptor_t m_aInDesc, m_aOutDesc, m_aInputDeltaDesc, m_aDeltaDesc;

    cudnnSoftmaxAlgorithm_t m_algo;
    cudnnSoftmaxMode_t      m_mode;

    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;

    DTYPE m_alpha;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    DTYPE m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
#endif

public:
    /*!
    @brief Softmax의 생성자.
    @details 파라미터로 받은 pOperator, epsilon을 Alloc시킨다.
    @param pOperator Softmax할 대상 Operator, 이 매소드에서 Alloc시킨다.
    @param epsilon ForwardPropagate에 사용힐 값. 0으로 나누어지는 것을 방지하는 역할을 한다.
    @ref virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f
    */
    Softmax(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f, int pLoadflag = TRUE) : Operator<DTYPE>(pOperator, pLoadflag) {
#ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
#endif // __DEBUG__
        Alloc(pOperator, epsilon);
    }

    /*!
    @brief Softmax의 생성자.
    @details 파라미터로 받은 pOperator을 Alloc한다.
    @param pOperator Softmax할 대상 Operator, 이 매소드에서 Alloc시킨다.
    @param pName 사용자가 Operator에 부여한 이름.
    @ref virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f
    */
    Softmax(Operator<DTYPE> *pOperator, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pOperator, pName, pLoadflag) {
#ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
#endif // __DEBUG__
        Alloc(pOperator);
    }

    /*!
    @brief Softmax의 생성자.
    @details 파라미터로 받은 pOperator, epsilon을 Alloc시킨다.
    @param pOperator Softmax할 대상 Operator, 이 매소드에서 Alloc시킨다.
    @prram epsilon ForwardPropagate에 사용힐 값. 0으로 나누어지는 것을 방지하는 역할을 한다.
    @param pName 사용자가 Operator에 부여한 이름.
    @ref virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f
    */
    Softmax(Operator<DTYPE> *pOperator, DTYPE epsilon, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pOperator, pName, pLoadflag) {
#ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
#endif // __DEBUG__
        Alloc(pOperator, epsilon);
    }

    /*!
    @brief Softmax의 소멸자.
    */
    ~Softmax() {
#ifdef __DEBUG__
        std::cout << "Softmax::~Softmax()" << '\n';
#endif // __DEBUG__
    }

    /*!
    @brief 파라미터로 받은 pOperator로 맴버변수들을 초기화 하고 Result, Gradient를 설정한다.
    @details input으로 받은 Operator의 Shape정보들로 맴버 변수드을 초기화 하고, 같은 Shape을 갖는 Tensor를 만들어 Result와 Gradient로 설정한다.
    @param pOperator Softmax할 Operator들
    @param epsilon 0으로 나누어지는 것을 방지하기위해 softmax식의 분모에 더하는 값.
    @return 성공 시 TRUE.
    */
    virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f) {
        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_timesize = timesize;

        sum = new DTYPE *[timesize];
        max = new DTYPE *[timesize];

        for (int i = 0; i < timesize; i++) {
            sum[i] = new DTYPE[batchsize];
            max[i] = new DTYPE[batchsize];
        }

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetGradient(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

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
        Operator<DTYPE> *pInput = this->GetInput()[0];

        Shape *shapeOfInput  = pInput->GetResult()->GetShape();
        Shape *shapeOfResult = this->GetResult()->GetShape();

        int batchsize   = this->GetResult()->GetBatchSize();
        int channelsize = this->GetResult()->GetChannelSize();
        int rowsize     = this->GetResult()->GetRowSize();
        int colsize     = this->GetResult()->GetColSize();

        int batchsizeOfInput   = (*shapeOfInput)[1];
        int channelsizeOfInput = (*shapeOfInput)[2];
        int rowsizeOfInput     = (*shapeOfInput)[3];
        int colsizeOfInput     = (*shapeOfInput)[4];

        int batchsizeOfResult   = (*shapeOfResult)[1];
        int channelsizeOfResult = (*shapeOfResult)[2];
        int rowsizeOfResult     = (*shapeOfResult)[3];
        int colsizeOfResult     = (*shapeOfResult)[4];

        m_alpha = 1;
        m_beta  = 0;

        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aOutDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputDeltaDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfInput, channelsizeOfInput, rowsizeOfInput, colsizeOfInput));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aOutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfResult, channelsizeOfResult, rowsizeOfResult, colsizeOfResult));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfInput, channelsizeOfInput, rowsizeOfInput, colsizeOfInput));

        //cudnnSoftmaxAlgorithm_t - 62pg
        m_algo = CUDNN_SOFTMAX_ACCURATE;    //CUDNN_SOFTMAX_FAST

        //cudnnSoftmaxMode_t  - 62pg
        // m_mode = CUDNN_SOFTMAX_MODE_CHANNEL;
        m_mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    }

#endif // if __CUDNN__

    /*!
    @brief Alloc매소드에서 할당했던 sum, max를 삭제하고 포인터를 NULL로 초기화 한다.
    */
    virtual void Delete() {
        if (sum) {
            for (int i = 0; i < m_timesize; i++) {
                delete[] sum[i];
                sum[i] = NULL;
            }
            delete[] sum;
        }

        if (max) {
            for (int i = 0; i < m_timesize; i++) {
                delete[] max[i];
                max[i] = NULL;
            }
            delete[] max;
        }
#ifdef __CUDNN__
        if (m_aInDesc)
            checkCUDNN(cudnnDestroyTensorDescriptor(m_aInDesc));
        m_aInDesc = NULL;

        if (m_aOutDesc)
            checkCUDNN(cudnnDestroyTensorDescriptor(m_aOutDesc));
        m_aOutDesc = NULL;

        if (m_aDeltaDesc)
            checkCUDNN(cudnnDestroyTensorDescriptor(m_aDeltaDesc));
        m_aDeltaDesc = NULL;

        if (m_aInputDeltaDesc)
            checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputDeltaDesc));
        m_aInputDeltaDesc = NULL;
#endif // if __CUDNN__
    }

    /*!
    @brief Softmax의 ForwardPropagate 매소드
    @details max값을 계산하고, exp()한 모든 값들을 더해 sum을 구한 뒤, 각각의 exp(input)한 값을 sum으로 나누어주어 확률값을 구하고 result에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            sum[ti][ba] = 0.f;
            max[ti][ba] = 0.f;
        }

        int capacity = colsize;

        int start = 0;
        int end   = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            max[ti][ba] = Max(input, start, end);
        }

        DTYPE temp = 0.f;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                temp += (exp((*input)[i] - max[ti][ba]) + m_epsilon);
            }
            sum[ti][ba] = temp;
            temp        = 0.f;
        }

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*result)[i] = (exp((*input)[i] - max[ti][ba]) + m_epsilon) / sum[ti][ba];
            }
        }
        return TRUE;
    }

    /*!
    @brief softmax의 BackPropagate 매소드.
    @details softmax의 미분 값을 구하여 input_delta에 넣어준다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int ti = pTime;

        int capacity = colsize;

        int start = 0;
        int end   = 0;

        float temp = 0.f;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            temp = 0.f;

            for (int i = start; i < end; i++) {
                temp += (*this_delta)[i] * (*result)[i];
            }

            for (int i = start; i < end; i++) {
                (*input_delta)[i] = (*result)[i] * ((*this_delta)[i] - temp);
            }
        }

        return TRUE;
    }

    /*!
    @brief 파라미터로 받은 Tensor에서 가장 큰 값을 반환하는 함수.
    @param input 가장 큰 값을 찾을 대상 Tensor.
    @param start 값을 찾을 Tensor안에서의 시작위치.
    @param end 값을 찾을 Tensor안에서의 종료위치.
    @return input Tensor의 값들 중 가장 큰 값..
    */
    DTYPE Max(Tensor<DTYPE> *input, int start, int end) {
        DTYPE max = (*input)[start];

        for (int i = start + 1; i < end; i++) {
            if ((*input)[i] > max)
                max = (*input)[i];
        }

        return max;
    }

#ifdef __CUDNN__

    int ForwardPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnSoftmaxForward(this->GetCudnnHandle(), m_algo, m_mode,
                                       &m_alpha, m_aInDesc, m_pDevInput,
                                       &m_beta, m_aOutDesc, m_pDevOutput));

        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *result      = this->GetResult();

        m_pDevOutput     = result->GetGPUData(pTime);
        m_pDevDelta      = this_delta->GetGPUData(pTime);
        m_pDevInputDelta = input_delta->GetGPUData(pTime);

        checkCUDNN(cudnnSoftmaxBackward(this->GetCudnnHandle(), m_algo, m_mode,
                                        &m_alpha, m_aOutDesc, m_pDevOutput, m_aDeltaDesc, m_pDevDelta,
                                        &m_beta, m_aInputDeltaDesc, m_pDevInputDelta));

        return TRUE;
    }

#endif // if __CUDNN__

};

#endif // SOFTMAX_H_
