#ifndef LIMITRELU_H_
#define LIMITRELU_H_    value

#include "../Operator.hpp"

template<typename DTYPE> class LimitRelu : public Operator<DTYPE>{
private:
#ifdef __CUDNN__
    cudnnTensorDescriptor_t m_aInputTensorDesc, m_aOutputTensorDesc, m_aDeltaDesc, m_aInputDeltaDesc;
    ///< GPU내의 Tensor값들을 가르키기 위한 descriptor.
    cudnnActivationDescriptor_t actDesc;
    ///< Activation 연산의 description을 가리키는 구조체 포인터.
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;
    ///< cudnn 연산에서 사용 할 데이터를 가리키는 맴버 변수.

    float m_alpha;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    float m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    float m_threshold;
    ///< Limit Threshold를 지정하기 위한 값

#endif  // __CUDNN__

public:
    /*!
    @brief Relu의 생성자.
    @details 파라미터로 받은 pInput으로 Alloc한다.
    @param pInput Alloc할 대상 Operator
    @ref int Alloc(Operator<DTYPE> *pInput)
    */
    LimitRelu(Operator<DTYPE> *pInput, float threshold, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "Relu::Relu(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, threshold);
    }

    /*!
    @brief Relu의 생성자.
    @details 파라미터로 받은 pInput으로 Alloc한다.
    @param pInput Alloc할 대상 Operator
    @param pName Operator에 사용자가 부여한 이름.
    @ref int Alloc(Operator<DTYPE> *pInput)
    */
    LimitRelu(Operator<DTYPE> *pInput, float threshold, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "Relu::Relu(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, threshold);
    }

    /*!
    @brief Relu의 소멸자.
    @see void Delete()
    */
    ~LimitRelu() {
        #ifdef __DEBUG__
        std::cout << "Relu::~Relu()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    /*!
    @brief 파라미터로 받은 pinput으로부터 맴버 변수들을 초기화 한다.
    @details Result와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
    @param pInput 생성 할 Tensor의 Shape정보를 가진 Operator
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE> *pInput, float threshold) {
        #ifdef __DEBUG__
        std::cout << "Relu::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        m_threshold = threshold;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));


        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief cudnn을 사용하기 전 관련 맴버변수들을 초기화 한다.
    @details Activation 함수를 Relu로 지정한다.
    @details TensorDesriptor들을 생성하고, TensorDesriptor들의 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
    @param idOfDevice 사용할 GPU의 id
    */
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        Operator<DTYPE> *pInput = this->GetInput()[0];

        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        int inputCapacity  = pInput->GetResult()->GetCapacity();
        int outputCapacity = this->GetResult()->GetCapacity();

        m_alpha = 1.f;
        m_beta  = 0.f;

        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aOutputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputDeltaDesc));
        checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));

        checkCUDNN(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, m_threshold));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aOutputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));
    }

#endif  // if __CUDNN__

    void Delete() {
#ifdef __CUDNN__

        if (m_aInputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputTensorDesc));
        m_aInputTensorDesc = NULL;

        if (m_aOutputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aOutputTensorDesc));
        m_aOutputTensorDesc = NULL;

        if (m_aDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aDeltaDesc));
        m_aDeltaDesc = NULL;

        if (m_aInputDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputDeltaDesc));
        m_aInputDeltaDesc = NULL;

        if (actDesc) checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
        actDesc = NULL;

#endif  // if __CUDNN__
    }

        int ForwardPropagate(int pTime = 0) {
        
    }

        int BackPropagate(int pTime = 0) {
                
    }

    /*!
    @brief MAX함수.
    @details input 2개 중 더 큰 값을 반환한다.
    @param data1 비교할 값.
    @param data2 비교할 값
    @return data1, data2중 더 큰 값.
    */
    
    // inline DTYPE MAX(DTYPE data1, DTYPE data2) {
    //     if (data1 >= data2) return data1;
    //     else return data2;
    // }

#ifdef __CUDNN__
    /*!
    @brief GPU에서 동작하는 ForwardPropagate 메소드.
    @details 지정한 Activation function(Relu)의 ForwardPropagate연산을 실행한다.
    @details m_pDevOutput에 결과 값을 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int ForwardPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnActivationForward(this->GetCudnnHandle(), actDesc, &m_alpha,
                                          m_aInputTensorDesc, m_pDevInput, &m_beta,
                                          m_aOutputTensorDesc, m_pDevOutput));

        // checkCudaErrors(cudaDeviceSynchronize());
        return TRUE;
    }

    /*!
    @brief GPU에서 동작하는 BackPropagate메소드.
    @details 지정한 Activation function(Relu)의 BackPropagate연산을 실행한다.
    @details m_pDevDelta에 결과 값을 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int BackPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        m_pDevInput      = input->GetGPUData(pTime);
        m_pDevOutput     = result->GetGPUData(pTime);
        m_pDevDelta      = this_delta->GetGPUData(pTime);
        m_pDevInputDelta = input_delta->GetGPUData(pTime);

        checkCUDNN(cudnnActivationBackward(this->GetCudnnHandle(), actDesc, &m_alpha,
                                           m_aOutputTensorDesc, m_pDevOutput,
                                           m_aDeltaDesc, m_pDevDelta,
                                           m_aInputTensorDesc, m_pDevInput, &m_alpha,
                                           m_aInputTensorDesc, m_pDevInputDelta));

        // checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // LIMITRELU_H_
