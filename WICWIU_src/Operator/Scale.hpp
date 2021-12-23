#ifndef __SCALE_HPP__
#define __SCALE_HPP__

#include "../Operator.hpp"

/**
 * @brief 입력 Tensor의 값들을 일정 비율로 Scaling하는 Operator Class.
 */
template <typename DTYPE>
class Scale : public Operator<DTYPE>
{
private:
    float m_scaleFactor;

#ifdef __CUDNN__
    Tensor<DTYPE> *m_aScaleFactorTensor;

    cudnnOpTensorDescriptor_t m_aOpTensorDesc;

    cudnnTensorDescriptor_t m_aInOutDesc, m_aDeltaDesc, m_aInputDeltaDesc;

    DTYPE *m_pDevInput, *m_pDevScale, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;

    DTYPE m_alpha;
    DTYPE m_beta;

#endif

public:

    /*!
    @brief Scale의 생성자.
    @details 파라미터로 받은 pInput, scaleFactor으로 Alloc한다.
    @param pInput Alloc할 대상 Operator
    @param scaleFactor pInput tensor 에 scale 적용 할 값
    @ref int Alloc(pInput, scaleFactor)
    */
    Scale(Operator<DTYPE> *pInput, float scaleFactor, std::string pName = "NO NAME", int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pName, pLoadflag)
    {
#ifdef __DEBUG__
        std::cout << "Scale<DTYPE>::Scale(Operator<DTYPE> *, int d_k, std::string , int )" << '\n';
#endif // __DEBUG__

        m_scaleFactor = 0;

        Alloc(pInput, scaleFactor);
    }

    /*!
    @brief 파라미터로 받은 pinput으로부터 맴버 변수들을 초기화 한다.
    @details Result와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
    @details scaleFactor는 m_scaleFactor에 저장한다.
    @param pInput Alloc할 대상 Operator
    @param scaleFactor pInput tensor 에 scale 적용 할 값
    @return 성공 시 TRUE
    */
    int Alloc(Operator<DTYPE> *pInput, float scaleFactor)
    {
#ifdef __DEBUG__
        std::cout << "Scale<DTYPE>::Alloc(Operator<DTYPE> *, int )" << '\n';
#endif // __DEBUG__

        m_scaleFactor = scaleFactor;

        Shape *inputShape = pInput->GetResult()->GetShape();

        int timesize = (*inputShape)[0];
        int batchsize = (*inputShape)[1];
        int channelsize = (*inputShape)[2];
        int rowsize = (*inputShape)[3];
        int colsize = (*inputShape)[4];

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetGradient(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
#ifdef __CUDNN__
        m_aScaleFactorTensor = Tensor<DTYPE>::Constants(timesize, batchsize, channelsize, rowsize, colsize, m_scaleFactor);
#endif
        return TRUE;
    }

#ifdef __CUDNN__
/*!
    @brief cudnn을 사용하기 전 관련 맴버변수들을 초기화 한다.
    @details TensorDesriptor들을 생성하고, TensorDesriptor들의 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
    @details Scale 연산에 필요한 알고리즘을 정의하고, 연산에 필요한 메모리공간을 할당 받는다. Scale은 cudnnOpTensor 연산을 이용한다.
    @param idOfDevice 사용할 GPU의 id
    */
    void InitializeAttributeForGPU(unsigned int idOfDevice)
    {
        Operator<DTYPE> *pInput = this->GetInput()[0];
        Shape *shape = pInput->GetResult()->GetShape();

        int timesize = (*shape)[0];
        int batchsize = (*shape)[1];
        int channelsize = (*shape)[2];
        int rowsize = (*shape)[3];
        int colsize = (*shape)[4];

        m_alpha = 1;
        m_beta = 0;

        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInOutDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputDeltaDesc));

        checkCUDNN(cudnnCreateOpTensorDescriptor(&m_aOpTensorDesc));
        checkCUDNN(cudnnSetOpTensorDescriptor(m_aOpTensorDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInOutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        m_aScaleFactorTensor->SetDeviceGPU(this->GetDeviceID());
    }
#endif

    /*!
    @brief GPU에 할당했던 메모리를 해제하고 각 포인터들을 NULL로 초기화한다.
    @details m_aOpTensorDesc, m_aInOutDesc,m_aDeltaDesc, m_aInputDeltaDesc들을 삭제하고 NULL로 초기화한다.
    */
    void Delete()
    {
#ifdef __CUDNN__

        if (m_aOpTensorDesc)
            checkCUDNN(cudnnDestroyOpTensorDescriptor(m_aOpTensorDesc));
        m_aOpTensorDesc = NULL;

        if (m_aInOutDesc)
            checkCUDNN(cudnnDestroyTensorDescriptor(m_aInOutDesc));
        m_aInOutDesc = NULL;

        if (m_aDeltaDesc)
            checkCUDNN(cudnnDestroyTensorDescriptor(m_aDeltaDesc));
        m_aDeltaDesc = NULL;

        if (m_aInputDeltaDesc)
            checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputDeltaDesc));
        m_aInputDeltaDesc = NULL;

#endif
    }

    /*!
    @brief Scale의 ForwardPropagate매소드.
    @details inputTensor의 각 element마다 m_scaleFactor을 곱해준다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0)
    {

        Tensor<DTYPE> *inputTensor = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        Shape *resultShape = result->GetShape();

        int timesize = (*resultShape)[0];
        int batchsize = (*resultShape)[1];
        int channelsize = (*resultShape)[2];
        int rowsize = (*resultShape)[3];
        int colsize = (*resultShape)[4];

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++)
        {
            for (int ch = 0; ch < channelsize; ch++)
            {
                for (int ro = 0; ro < rowsize; ro++)
                {
                    for (int co = 0; co < colsize; co++)
                    {
                        int index = Index5D(resultShape, ti, ba, ch, ro, co);
                        (*result)[index] = m_scaleFactor * (*inputTensor)[index];
                    }
                }
            }
        }

        return TRUE;
    }

    /*!
    @brief Scale의 BackPropagate 매소드.
    @details input_delta에 m_scaleFactor * this_delta값을 더해준다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0)
    {
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *this_delta = this->GetDelta();

        Shape *pThisDeltaTenShape = this_delta->GetShape();
        Shape *pInputDeltaTenShape = input_delta->GetShape();

        int batchsize = input_delta->GetBatchSize();
        int channelsize = input_delta->GetChannelSize();
        int rowsize = input_delta->GetRowSize();
        int colsize = input_delta->GetColSize();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++)
        {
            for (int ch = 0; ch < channelsize; ch++)
            {
                for (int ro = 0; ro < rowsize; ro++)
                {
                    for (int co = 0; co < colsize; co++)
                    {
                        int index = Index5D(pThisDeltaTenShape, ti, ba, ch, ro, co);
                        (*input_delta)[index] = m_scaleFactor * (*this_delta)[index];
                    }
                }
            }
        }

        return TRUE;
    }

#ifdef __CUDNN__
    /*!
    @brief GPU에서 동작하는 ForwardPropagate 메소드.
    @details cudnn이 제공하는 cudnnOpTensor 연산을 이용하여 Scale의 ForwardPropagate연산을 한다.
    @details Scale의 ForwardPropagate결과는 m_pDevOutput에 저장된다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int ForwardPropagateOnGPU(int pTime = 0)
    {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);
        m_pDevScale = m_aScaleFactorTensor->GetGPUData(pTime);

        checkCUDNN(cudnnOpTensor(this->GetCudnnHandle(), m_aOpTensorDesc,
                                 &m_alpha, m_aInOutDesc, m_pDevInput,
                                 &m_alpha, m_aInOutDesc, m_pDevScale,
                                 &m_beta, m_aInOutDesc, m_pDevOutput));

        return TRUE;
    }

    /*!
    @brief GPU에서 동작하는 BackwardPropagate 메소드.
    @details cudnn이 제공하는 cudnnOpTensor 연산을 아용하여 Scale의 BackwardPropagate 연산을 한다.
    @details Scale BackwardPropagate결과는 m_pDevInputDelta에 저장된다.
    @param pTime 연산 할 Tensor가 위치한 Time값.
    @return 성공 시 TRUE.
    */
    int BackPropagateOnGPU(int pTime = 0)
    {
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *this_delta = this->GetDelta();

        m_pDevDelta = this_delta->GetGPUData(pTime);
        m_pDevInputDelta = input_delta->GetGPUData(pTime);
        m_pDevScale = m_aScaleFactorTensor->GetGPUData(pTime);

        checkCUDNN(cudnnOpTensor(this->GetCudnnHandle(), m_aOpTensorDesc,
                                 &m_alpha, m_aDeltaDesc, m_pDevDelta,
                                 &m_alpha, m_aDeltaDesc, m_pDevScale,
                                 &m_beta, m_aInputDeltaDesc, m_pDevInputDelta));
        return TRUE;
    }
#endif
};

#endif