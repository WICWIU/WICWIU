#ifdef __CUDNN__

#ifndef __CUDNN_BATCH_NORMALIZE_LAYER__
#define __CUDNN_BATCH_NORMALIZE_LAYER__ value

#include "../Module.h"

// template<typename DTYPE> class CUDNNBatchNormalizeLayer2D : public Module<DTYPE>{
// private:
// public:
//     CUDNNBatchNormalizeLayer2D(Operator<DTYPE> *pInput, int pNumOfChannel, std::string pName = "NO NAME") : Module<DTYPE>(pName) {
//         Alloc(pInput, pNumOfChannel, pName);
//     }
//
//     virtual ~CUDNNBatchNormalizeLayer2D() {}
//
//     int Alloc(Operator<DTYPE> *pInput, int pNumOfChannel, std::string pName) {
//         Operator<DTYPE> *out = pInput;
//
//         Tensorholder<DTYPE> *pGamma = (Tensorholder<DTYPE> *) this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumOfChannel, 1, 1, 1.0), "CUDNNBatchNormalize_Gamma_" + pName));
//         Tensorholder<DTYPE> *pBeta  = (Tensorholder<DTYPE> *) this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, pNumOfChannel, 1, 1), "CUDNNBatchNormalize_Beta_" + pName));
//             std::cout << pGamma->GetResult()->GetShape() << '\n';
//             std::cout << pBeta->GetResult()->GetShape() << '\n';
//
//         out = this->AddOperator(new CUDNNBatchNormalize<DTYPE>(out, pGamma, pBeta, TRUE, "CUDNNBatchNormalize_CUDNNBatchNormalize_" + pName));
//
//         return TRUE;
//     }
// };

template<typename DTYPE> class CUDNNBatchNormalizeLayer : public Module<DTYPE>{
private:
public:
    CUDNNBatchNormalizeLayer(Operator<DTYPE> *pInput, int pIsChannelwise = FALSE, std::string pName = "NO NAME") {
        Alloc(pInput, pIsChannelwise, pName);
    }

    virtual ~CUDNNBatchNormalizeLayer() {}

    int Alloc(Operator<DTYPE> *pInput, int pIsChannelwise, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;
        Shape *pInputShape   = out->GetResult()->GetShape();

        Tensorholder<DTYPE> *pGamma = NULL;
        Tensorholder<DTYPE> *pBeta  = NULL;

        if (pIsChannelwise) {
            int pNumInputChannel = (*pInputShape)[2];
            pGamma = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumInputChannel, 1, 1, 1.0), "CUDNNBatchNormalize_Gamma_" + pName);
            pBeta = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, pNumInputChannel, 1, 1), "CUDNNBatchNormalize_Beta_" + pName);
        } else {
            int pNumInputCol     = (*pInputShape)[4];
            pGamma = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumInputCol, 1.0), "CUDNNBatchNormalize_Gamma_" + pName);
            pBeta  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, 1, 1, pNumInputCol), "CUDNNBatchNormalize_Beta_" + pName);
        }
            std::cout << pGamma->GetResult()->GetShape() << '\n';
            std::cout << pBeta->GetResult()->GetShape() << '\n';

        out = new CUDNNBatchNormalize<DTYPE>(out, pGamma, pBeta, pIsChannelwise, "CUDNNBatchNormalize_CUDNNBatchNormalize_" + pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __CUDNN_BATCH_NORMALIZE_LAYER__

#endif // __CUDNN__
