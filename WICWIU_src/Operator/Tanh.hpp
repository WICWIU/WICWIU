#ifndef TANH_H_
#define TANH_H_    value

#include "../Operator.hpp"

template<typename DTYPE>
class Tanh : public Operator<DTYPE>{
private:

#ifdef __CUDNN__

    cudnnActivationDescriptor_t activationDesc;

    //deltaDesc = 본인의 deltaDesc
    //m_aInputDeltaDesc = 계산해서 아래에 넘겨줄 dalta
    cudnnTensorDescriptor_t m_aInputTensorDesc, m_aOutputTensorDesc, m_aDeltaDesc, m_aInputDeltaDesc;
    ///< GPU내의 Tensor값들을 가르키기 위한 descriptor.

    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;
    ///<  cudnn 연산에서 사용 할 데이터를 가리키는 맴버 변수.

    float m_alpha;
    ///<  연산 간 두 Operand의 가중치를 표현하기 한 변수. ex) z = α*x + β*y
    float m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 귀한 변수. ex) z = α*x + β*y
    double m_coef;

#endif  // __CUDNN__

public:
    /*!
    @brief Tanh의 생성자
    @details 파라미터로 받은 pInput으로 Alloc한다.
    @param pInput Alloc할 대상 Operator
    @param pName Operator에 사용자가 부여한 이름.
    */
    Tanh(Operator<DTYPE> *pInput, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "Tanh::Tanh(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput);
    }

    /*!
    @brief Tanh의 소멸자.
    */
    ~Tanh() {
        std::cout << "Tanh::~Tanh()" << '\n';
    }

    /*!
    @brief 파라미터로 받은 pInput으로부터 맴버 변수들을 초기화 한다.
    @details Result와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
    @param pInput 생성 할 Tensor의 Shape정보를 가진 Operator
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE> *pInput) {
        #ifdef __DEBUG__
        std::cout << "Tanh::Alloc(Operator *, Operator *)" << '\n';
        #endif  // __DEBUG__

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
    @details TensorDesriptor들을 생성하고, TensorDesriptor들의 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
    @param idOfDevice 사용할 GPU의 id
    */
    void InitializeAttributeForGPU(unsigned int idOfDevice) {

        Operator<DTYPE> *pInput  = this->GetInput()[0];

        //int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_alpha = 1.f;
        m_beta  = 0.f;
        m_coef = 0.0;       //means nothing in tanh activation

        checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));

        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputTensorDesc));            //입력값
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aOutputTensorDesc));           //출력값
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputDeltaDesc));             //입력의 delta 즉 계산해서 넘겨 줄 곳
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaDesc));                  //위에서 전해주는 datlta값


        checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, m_coef));



        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aOutputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));
    }

#endif  // if __CUDNN__

    /*!
    @brief Tanh의 ForwardPropagate 매소드
    @details input의 Tensor값들을 Tanh을 취한 뒤 result에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        if(isnan((*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)]) != 0)
                          std::cout<<"입력값이 이미 nan임";
                        (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                            = tanh((*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)]);

                        if(isnan((*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]) != 0)
                          std::cout<<"출력에서 nan발생"<<(*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)]<<'\n';//<<ti<<ba<<ch<<ro<<co<<'\n';
                    }
                }
            }
        }

        return TRUE;
    }

    /*!
    @brief Tanh의 BackPropagate 매소드.
    @details result값으로 tanh의 미분 값을 계산하여 input_delta에 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        //std::cout<<"tanh back"<<'\n';

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        (*input_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                            += (1 - (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)])
                               * (1 + (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)])
                               * (*this_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)];
                        //std::cout<<(*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]<<'\n';
                        //std::cout<<(*input_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)]<<'\n'<<(*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]<<'\n'<<(*this_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)]<<'\n';
                    }
                }
            }
        }

        return TRUE;
    }


#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime) {

        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnActivationForward(this->GetCudnnHandle(), activationDesc, &m_alpha, m_aInputTensorDesc, m_pDevInput, &m_beta, m_aOutputTensorDesc, m_pDevOutput));

        return TRUE;
    }

    int BackPropagateOnGPU(int pTime) {

      Tensor<DTYPE> *input           = this->GetInput()[0]->GetResult();
      Tensor<DTYPE> *result          = this->GetResult();
      Tensor<DTYPE> *input_delta     = this->GetInput()[0]->GetDelta();
      Tensor<DTYPE> *this_delta      = this->GetDelta();

      m_pDevInput = input->GetGPUData(pTime);
      m_pDevOutput = result->GetGPUData(pTime);
      m_pDevDelta = this_delta->GetGPUData(pTime);
      m_pDevInputDelta = input_delta->GetGPUData(pTime);

      checkCUDNN(cudnnActivationBackward(this->GetCudnnHandle(), activationDesc, &m_alpha, m_aOutputTensorDesc, m_pDevOutput, m_aDeltaDesc, m_pDevDelta,
                                         m_aInputTensorDesc, m_pDevInput, &m_beta, m_aInputDeltaDesc, m_pDevInputDelta));

      return TRUE;
    }

#endif  // __CUDNN__
};

#endif  // TANH_H_
