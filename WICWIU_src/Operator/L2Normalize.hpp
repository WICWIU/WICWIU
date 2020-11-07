#ifndef L2NORMALIZE_H_
#define L2NORMALIZE_H_    value

#include "../Operator.hpp"
#include <math.h>

template<typename DTYPE> class L2Normalize : public Operator<DTYPE>{
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
    float* norm2List;
    Tensor<DTYPE>* norm2ListGPU_;
#endif  // __CUDNN__

public:
    L2Normalize(Operator<DTYPE> *pInput, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, "NO NAME", pLoadflag) {
        this->Alloc(pInput);
    }

    
    L2Normalize(Operator<DTYPE> *pInput, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pName, pLoadflag){
        std::cout <<"initt" << std::endl;
        std::cout << "INIT!!!!" << std::endl;
        this->Alloc(pInput);
    }

    /*!
    @brief LRelu의 소멸자.
    @see void Delete()
    */
    ~L2Normalize() {
        #ifdef __DEBUG__
        std::cout << "L2Normalize::~L2Normalize()" << '\n';
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
    int Alloc(Operator<DTYPE> *pInput) {

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        norm2List = new float[batchsize];
        // this->norm2ListGPU_ = Tensor<DTYPE>::Zeros(1, 1, 1, 1, batchsize);
        this->norm2ListGPU_ = NULL;

        
        
        std::cout << "Alloc!!!" << std::endl;

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }


#ifdef __CUDNN__
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

        // checkCudaErrors(cudaDeviceSynchronize());
#endif  // if __CUDNN__
   }


    /*!
    @brief PRelu의 ForwardPropagate 매소드.
    @details input의 Tensor값들 중 0.f이상의 값은 그대로 result에 저장하고,
    @details 0.f미만의 값은 weight Tensor의 값과 곱한 후 저장한다.
    @param pTime pInput의 m_timesize값, default는 0을 사용.
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


        float norm2 = 0.f;
        int ti = pTime;

        int index = 0;
        for (int ba = 0; ba < batchsize; ba++) {
            norm2 = 0.f;
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {

                        index = Index5D(resultTenShape, ti, ba, ch, ro, co);
                        if((*input)[index] >= 0.f)
                            norm2 += (*input)[index] * (*input)[index];
                    }
                }
            }

            norm2 = sqrt(norm2);
            norm2List[ba] = norm2;

            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {

                        index = Index5D(resultTenShape, ti, ba, ch, ro, co);
                        if((*input)[index] >= 0.f)
                            (*result)[index] = (*input)[index] / norm2;
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
        Tensor<DTYPE> *result      = this->GetResult();

        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        float sumOfDelta = 0.f;

        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        int index = 0;
        for (int ba = 0; ba < batchsize; ba++) {
            sumOfDelta = 0.f;
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {

                            sumOfDelta += (*this_delta)[index] * (*result)[index];
                        }
                    }
                }
            

            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                            (*input_delta)[index] += (((*this_delta)[index] - (*result)[index] * sumOfDelta) / norm2List[ba]);
                        }
                    }
                }
            
        }

        return TRUE;
    }

#ifdef __CUDNN__
    
    int ForwardPropagateOnGPU(int pTime = 0);

    

    int BackPropagateOnGPU(int pTime = 0);

#endif  // if __CUDNN__

};


#endif  // L2NORMALIZE_H_
