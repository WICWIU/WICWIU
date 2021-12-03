#ifndef __MASKED_FILL_HPP__
#define __MASKED_FILL_HPP__

#include "../Operator.hpp"
/**
 * @brief Mask Tensor와 비교하여 Masking 할 위치의 Input Tensor의 값을 특정 값으로 Set하는 Operator Class
 */
template <typename DTYPE> class MaskedFill : public Operator<DTYPE> {
private:
    DTYPE m_maskingValue;

#ifdef __CUDNN__
    DTYPE *m_pDevInput, *m_pDevMask, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;
#endif

public:
    /**
     * @brief MaskedFill의 생성자.
     * @details 파라미터로 받은 pInput, pMask, masking value로 Alloc한다.
     * @param pInput Alloc 할 대상 Operator
     * @param pMask Alloc 할 대상 Operator, Super Class의 생성자에 의해 Alloc 된다.
     * @param maskingValue Masking 할 위치에 저장할 값 
     */
    MaskedFill(Operator<DTYPE> *pInput, Operator<DTYPE> *pMask, DTYPE maskingValue, std::string pName = "NO NAME", int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pMask, pName, pLoadflag) {
#ifdef __DEBUG__
        std::cout << "MaksedFill<DTYPE>::MaskedFill(Operator<DTYPE> *, Operator<DTYPE> *, std::string , int )" << '\n';
#endif // __DEBUG__
        m_maskingValue = maskingValue;
        Alloc(pInput);
    }
    /**
     * @brief 파라미터로 받은 pInput, scaleFactor로부터 멤버 변수들을 초기화한다.
     * @details Result와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
     * @param pInput 생성할 Tensor의 Shape정보를 가진 Operator
     * @return 성공시 TRUE 
     */
    int Alloc(Operator<DTYPE> *pInput) {
        #ifdef __DEBUG__
            std::cout << "MaksedFill<DTYPE>::MaskedFill(Operator<DTYPE> * , Operator<DTYPE> * )" << '\n';
        #endif // __DEBUG__

        #ifdef __CUDNN__
            m_pDevInput = NULL;
            m_pDevMask = NULL;
            m_pDevOutput = NULL;
            m_pDevInputDelta = NULL;
            m_pDevDelta = NULL;
        #endif  // __CUDNN__


        Tensor<DTYPE> *pTensor = pInput->GetResult();

        int timesize    = pTensor->GetTimeSize();
        int batchsize   = pTensor->GetBatchSize();
        int channelsize = pTensor->GetChannelSize();
        int rowsize     = pTensor->GetRowSize();
        int colsize     = pTensor->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    void Delete() {

        #ifdef __CUDNN__
            if(m_pDevInput) {
                checkCudaErrors(cudaFree(m_pDevInput));
                m_pDevInput = NULL;
            }
            if(m_pDevMask) {
                checkCudaErrors(cudaFree(m_pDevMask));
                m_pDevMask = NULL;
            }
            if(m_pDevOutput) {
                checkCudaErrors(cudaFree(m_pDevOutput));
                m_pDevOutput = NULL;
            }
            if(m_pDevInputDelta) {
                checkCudaErrors(cudaFree(m_pDevInputDelta));
                m_pDevInputDelta = NULL;
            }
            if(m_pDevDelta) {
                checkCudaErrors(cudaFree(m_pDevDelta));
                m_pDevDelta = NULL;
            }
        #endif // __CUDNN__
    }
    /**
     * @brief MaskedFill의 ForwardPropagate 메소드.
     * @details Mask Tensor와 비교하여 Masking 할 위치의 Input Tensor의 값을 Masking Value로 Set.
     * @param pTime pInput의 m_timesize값, default는 0을 사용.
     * @return 성공 시 TRUE.
     */
    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *mask   = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int maskBatch = mask->GetBatchSize();

        Shape *resultTenShape = result->GetShape();
        Shape *maskTenShape   = mask->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        int index = Index5D(resultTenShape, ti, ba, ch, ro, co);
                        if ((*mask)[Index5D(maskTenShape, ti, ba, 0, ro, co)])
                            (*result)[index] = m_maskingValue;
                        else
                            (*result)[index] = (*input)[index];
                    }
                }
            }
        }

        return TRUE;
    }
    /**
     * @brief MaskedFill의 BackPropagate 메소드.
     * @details Mask Tensor와 비교하여 Masking 할 위치의 Input Gradient의 값을 0으로 설정.
     * @param pTime pInput의 m_timesize값, default는 0을 사용.
     * @return 성공 시 TRUE.
     */
    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *mask        = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *this_delta  = this->GetDelta();

        Shape *pThisDeltaTenShape  = this_delta->GetShape();
        Shape *pInputDeltaTenShape = input_delta->GetShape();

        Shape *maskTenShape = mask->GetShape();

        int timesize    = input_delta->GetTimeSize();
        int batchsize   = input_delta->GetBatchSize();
        int channelsize = input_delta->GetChannelSize();
        int rowsize     = input_delta->GetRowSize();
        int colsize     = input_delta->GetColSize();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        int index = Index5D(pThisDeltaTenShape, ti, ba, ch, ro, co);
                        if ((*mask)[Index5D(maskTenShape, ti, ba, 0, ro, co)]) {
                            (*input_delta)[index] = 0;
                        }
                        else {
                            (*input_delta)[index] = (*this_delta)[index];
                        }
                    }
                }
            }
        }
        return TRUE;
    }
#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime);

    int BackPropagateOnGPU(int pTime);
#endif
};

#endif
