#ifndef __BATCH_NORMALIZE_LAYER__
#define __BATCH_NORMALIZE_LAYER__    value

#include "../Module.h"

template<typename DTYPE> class BatchNormalizeLayer : public Module<DTYPE>{
private:
public:
    BatchNormalizeLayer(Operator<DTYPE> *pInput, int pIsChannelwise = FALSE, std::string pName = "NO NAME") : Module<DTYPE>(pName) {
        Alloc(pInput, pIsChannelwise, pName);
    }

    virtual ~BatchNormalizeLayer() {}

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
