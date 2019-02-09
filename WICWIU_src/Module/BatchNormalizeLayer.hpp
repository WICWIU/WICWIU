#ifndef __BATCH_NORMALIZE_LAYER__
#define __BATCH_NORMALIZE_LAYER__    value

#include "../Module.hpp"

/*!
@class BatchNormalizeLayer Operator들을 그래프로 구성해 Batch Normalization Layer의 기능을 수행하는 모듈을 생성하는 클래스
@details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 Batch Normalization Layer의 기능을 수행한다
*/
template<typename DTYPE> class BatchNormalizeLayer : public Module<DTYPE>{
private:
public:
    /*!
    @brief BatchNormalizeLayer 클래스 생성자
    @details BatchNormalizeLayer 클래스의 Alloc 메소드를 호출한다.
    @see BatchNormalizeLayer<DTYPE>::Alloc(Operator<DTYPE> *pInput, int pIsChannelwise, std::string pName)
    */
    BatchNormalizeLayer(Operator<DTYPE> *pInput, int pIsChannelwise = FALSE, std::string pName = "NO NAME") : Module<DTYPE>(pName) {
        Alloc(pInput, pIsChannelwise, pName);
    }

    /*!
    @brief BatchNormalizeLayer 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.
    */
    virtual ~BatchNormalizeLayer() {}

    /*!
    @brief Batch Normalize Layer 그래프를 동적으로 할당 및 구성하는 메소드
    @details Input Operator의 Element에 대해 배치 정규화(Batch Normailzation)을 수행한다.
    @param pInput 해당 Layer의 Input에 해당하는 Operator
    @param pIsChannelWise Column-Wise Normalization 유무, 0일 시 Column-Wise롤 연산, 0이 아닐 시 Channel-Wise로 연산
    @param pName Module의 이름
    @return TRUE
    @see BatchNormalize<DTYPE>::BatchNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise = TRUE, std::string pName = NULL) Module<DTYPE>::AnalyzeGraph(Operator<DTYPE> *pResultOperator)
    */
    int Alloc(Operator<DTYPE> *pInput, int pIsChannelwise, std::string pName) {
        this->SetInput(pInput);
        Operator<DTYPE> *out = pInput;
        Shape *pInputShape   = out->GetResult()->GetShape();

        Tensorholder<DTYPE> *pGamma = NULL;
        Tensorholder<DTYPE> *pBeta  = NULL;
        Tensorholder<DTYPE> *pTotalGamma  = NULL;
        Tensorholder<DTYPE> *pTotalBeta  = NULL;

        if (pIsChannelwise) {
            int pNumInputChannel = (*pInputShape)[2];
            // for He initialization
            pGamma = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumInputChannel, 1, 1, 1), "BatchNormalize_Gamma_" + pName);
            pBeta  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, pNumInputChannel, 1, 1), "BatchNormalize_Beta_" + pName);
            // pTotalGamma = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumInputChannel, 1, 1, 1), "BatchNormalize_TotalGamma_" + pName);
            // pTotalBeta  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, pNumInputChannel, 1, 1), "BatchNormalize_TotalBeta_" + pName);
        } else {
            int pNumInputCol = (*pInputShape)[4];
            // for He initialization
            pGamma = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumInputCol, 1), "BatchNormalize_Gamma_" + pName);
            pBeta  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, 1, 1, pNumInputCol), "BatchNormalize_Beta_" + pName);
            // pTotalGamma = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumInputCol, 1), "BatchNormalize_TotalGamma_" + pName);
            // pTotalBeta  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, 1, 1, pNumInputCol), "BatchNormalize_TotalBeta_" + pName);
        }
        // std::cout << pGamma->GetResult()->GetShape() << '\n';
        // std::cout << pBeta->GetResult()->GetShape() << '\n';

        out = new BatchNormalize<DTYPE>(out, pGamma, pBeta, pIsChannelwise, "BatchNormalize_BatchNormalize_" + pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __BATCH_NORMALIZE_LAYER__
