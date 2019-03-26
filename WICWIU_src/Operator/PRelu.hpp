#ifndef PRELU_H_
#define PRELU_H_    value

#include "../Operator.hpp"
#include <math.h>

template<typename DTYPE> class PRelu : public Operator<DTYPE>{
private:

#ifdef __CUDNN__
    cudnnTensorDescriptor_t m_aInputTensorDesc, m_aOutputTensorDesc, m_aDeltaDesc, m_aInputDeltaDesc;
    cudnnActivationDescriptor_t actDesc;
    //cudnnTensorDescriptor_t m_aWeightTensorDesc, m_sWeightDeltaDesc;

    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;
    //DTYPE *m_pDevWeight, *m_pDevWeightDelta;

    float m_alpha;
    float m_beta;
    double m_coef;
#endif  // __CUDNN__

public:
    /*!
    @brief PRelu의 생성자.
    @details 파라미터로 받은 pInput, pWeight으로 Alloc한다.
    @param pInput Alloc할 대상 Operator
    @param pWeight 입력값이 음수일 경우 사용하는 기울기
    @ref int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight)
    */
    PRelu(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pWeight, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "PRelu::PRelu(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeight);
    }

    /*!
    @brief PRelu의 생성자.
    @details 파라미터로 받은 pInput, pWeight으로 Alloc한다.
    @param pInput Alloc할 대상 Operator
    @param pWeight 입력값이 음수일 경우 사용하는 기울기
    @param pName Operator에 사용자가 부여한 이름.
    @ref int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight)
    */
    PRelu(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pWeight, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "PRelu::PRelu(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeight);
    }

    /*!
    @brief LRelu의 소멸자.
    @see void Delete()
    */
    ~PRelu() {
        #ifdef __DEBUG__
        std::cout << "PRelu::~PRelu()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    /*!
    @brief 파라미터로 받은 pinput으로부터 맴버 변수들을 초기화 한다.
    @details Result와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
    @param pInput 생성 할 Tensor의 Shape정보를 가진 Operator
    @param pWeight 입력값이 음수일 경우 사용하는 Tensor의 정보를 가진 Operator
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight) {
        #ifdef __DEBUG__
        std::cout << "PRelu::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
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
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        Operator<DTYPE> *pInput = this->GetInput()[0];
        //Operator<DTYPE> *pWeight = this->GetInput()[1];

        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        int inputCapacity  = pInput->GetResult()->GetCapacity();
        int outputCapacity = this->GetResult()->GetCapacity();

        m_alpha = 1.f;
        m_beta  = 0.f;
        m_coef  = 0.0;

        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aOutputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputDeltaDesc));
        checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));

        checkCUDNN(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, m_coef));

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

#ifdef __CUDNN__
    void Delete() {
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

        // checkCudaErrors(cudaDeviceSynchronize());
   }
#endif  // if __CUDNN__


    /*!
    @brief PRelu의 ForwardPropagate 매소드.
    @details input의 Tensor값들 중 0.f이상의 값은 그대로 result에 저장하고,
    @details 0.f미만의 값은 weight Tensor의 값과 곱한 후 저장한다.
    @param pTime pInput의 m_timesize값, default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        int index = 0;
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {

                        index = Index5D(resultTenShape, ti, ba, ch, ro, co);
                        if((*input)[index] >= 0.f)
                            (*result)[index] = (*input)[index];
                        else
                            (*result)[index] = (*weight)[index] * (*input)[index];
                    }
                }
            }
        }

        /*int index = Index5D(resultTenShape, 0, 0, 0, 0, 0); //initial index
        DTYPE *input_ptr = &(*input)[index];
        DTYPE *weight_ptr = &(*weight)[index];
        DTYPE *result_ptr = &(*result)[index];
        DTYPE *input_limit = input_ptr + batchsize * channelsize * rowsize * colsize;
        for(; input_ptr < input_limit; input_ptr++, weight_ptr++, result_ptr++){

            if(*input_ptr >= 0.f)
              *result_ptr = *input_ptr;
            else
              *result_ptr = (*weight_ptr) * (*input_ptr);
        }
        */
        return TRUE;
    }

    /*!
    @brief PRelu의 BackPropagate 매소드.
    @details input_delta는 result값이 0보다 클 경우 input_delta에 더하고,
    @details 0보다 작을 경우 input_delta에 weight을 곱한 후 더한다.
    @details weight_delta는 result값이 0보다 클 경우 0에 더하고,
    @details 0보다 작을 경우 input_delta에 input을 곱한 후 더한다.
    @param pTime pInput의 m_timesize값, default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result      = this->GetResult();

        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *weight_delta = this->GetInput()[1]->GetDelta();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        int ti = pTime;
        int x;

        int index = 0;
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {

                        index = Index5D(resultTenShape, ti, ba, ch, ro, co);

                        if ((*result)[index] > 0.f) {
                            (*input_delta)[index] += (*this_delta)[index];
                            (*weight_delta)[index] += 0;
                        } else {
                            (*input_delta)[index] += (*weight)[index]*(*this_delta)[index];
                            (*weight_delta)[index] += (*input)[index]*(*this_delta)[index];
                        }
                        /*
                        if(std::isnan((*weight)[index]) || std::isnan((*this_delta)[index]) || std::isnan((*input_delta)[index]) || std::isnan((*weight_delta)[index])){
                          std::cin >> x;
                        }*/
                    }
                }
            }
        }
        /*
        int index = Index5D(resultTenShape, 0, 0, 0, 0, 0); //initial index
        DTYPE *input_delta_ptr = &(*input_delta)[index];
        DTYPE *weight_delta_ptr = &(*weight_delta)[index];
        DTYPE *this_delta_ptr = &(*this_delta)[index];
        DTYPE *input_ptr = &(*input)[index];
        DTYPE *weight_ptr = &(*weight)[index];
        DTYPE *result_ptr = &(*result)[index];
        DTYPE *delta_limit = input_delta_ptr + batchsize * channelsize * rowsize * colsize;
        for(; input_delta_ptr < delta_limit; input_delta_ptr++, this_delta_ptr++, input_ptr++, weight_ptr++, result_ptr++){
            if(*result_ptr > 0.f)
              *input_delta_ptr += *this_delta_ptr;
            else
              *input_delta_ptr += (*weight_ptr) * (*this_delta_ptr);
              *weight_delta_ptr += (*input_ptr) * (*this_delta_ptr);
        }
        */
        return TRUE;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {
        //this->ForwardPropagate(pTime);
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnActivationForward(this->GetCudnnHandle(), actDesc,
                                          &m_alpha, m_aInputTensorDesc, m_pDevInput,
                                          &m_beta, m_aOutputTensorDesc, m_pDevOutput));

        // checkCudaErrors(cudaDeviceSynchronize());
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        //this->BackPropagate(pTime);
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
                                           m_aInputTensorDesc, m_pDevInput, &m_beta,
                                           m_aInputTensorDesc, m_pDevInputDelta));

        // checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

#endif  // if __CUDNN__

};


#endif  // PRELU_H_
