#ifndef __BATCH_NORMALIZE_LAYER__
#define __BATCH_NORMALIZE_LAYER__ value

#include "../Layer.h"

template<typename DTYPE> class BatchNormalizeLayer2D : public Layer<DTYPE>{
private:
public:
    BatchNormalizeLayer2D(Operator<DTYPE> *pInput, int pNumOfChannel, std::string pName = "NO NAME") : Layer<DTYPE>(pName) {
        Alloc(pInput, pNumOfChannel, pName);
    }

    virtual ~BatchNormalizeLayer2D() {}

    int Alloc(Operator<DTYPE> *pInput, int pNumOfChannel, std::string pName) {
        Operator<DTYPE> *out = pInput;

        Tensorholder<DTYPE> *pGamma = (Tensorholder<DTYPE> *) this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumOfChannel, 1, 1, 1.0), "BatchNormalize_Gamma_" + pName));
        Tensorholder<DTYPE> *pBeta  = (Tensorholder<DTYPE> *) this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, pNumOfChannel, 1, 1), "BatchNormalize_Beta_" + pName));
            std::cout << pGamma->GetResult()->GetShape() << '\n';
            std::cout << pBeta->GetResult()->GetShape() << '\n';

        out = this->AddOperator(new BatchNormalize<DTYPE>(out, pGamma, pBeta, TRUE, "BatchNormalize_BatchNormalize_" + pName));

        return TRUE;
    }
};

template<typename DTYPE> class BatchNormalizeLayer : public Layer<DTYPE>{
private:
public:
    BatchNormalizeLayer(Operator<DTYPE> *pInput, int pIsChannelwise = FALSE, std::string pName = "NO NAME") {
        Alloc(pInput, pIsChannelwise, pName);
    }

    virtual ~BatchNormalizeLayer() {}

    int Alloc(Operator<DTYPE> *pInput, int pIsChannelwise, std::string pName) {
        Operator<DTYPE> *out = pInput;
        Shape *pInputShape   = out->GetResult()->GetShape();

        Tensorholder<DTYPE> *pGamma = NULL;
        Tensorholder<DTYPE> *pBeta  = NULL;

        if (pIsChannelwise) {
            int pNumInputChannel = (*pInputShape)[2];
            pGamma = (Tensorholder<DTYPE> *) this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumInputChannel, 1, 1, 1.0), "BatchNormalize_Gamma_" + pName));
            pBeta = (Tensorholder<DTYPE> *) this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, pNumInputChannel, 1, 1), "BatchNormalize_Beta_" + pName));
        } else {
            int pNumInputCol     = (*pInputShape)[4];
            pGamma = (Tensorholder<DTYPE> *) this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumInputCol, 1.0), "BatchNormalize_Gamma_" + pName));
            pBeta  = (Tensorholder<DTYPE> *) this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, 1, 1, pNumInputCol), "BatchNormalize_Beta_" + pName));
        }
            std::cout << pGamma->GetResult()->GetShape() << '\n';
            std::cout << pBeta->GetResult()->GetShape() << '\n';

        out = this->AddOperator(new BatchNormalize<DTYPE>(out, pGamma, pBeta, pIsChannelwise, "BatchNormalize_BatchNormalize_" + pName));

        return TRUE;
    }
};


#endif  // __BATCH_NORMALIZE_LAYER__
