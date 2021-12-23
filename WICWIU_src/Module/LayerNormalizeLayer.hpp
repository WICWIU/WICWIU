#ifndef __LAYER_NORMALIZE_LAYER_HPP__
#define __LAYER_NORMALIZE_LAYER_HPP__

#include "../Module.hpp"

template<typename DTYPE> class LayerNormalizeLayer : public Module<DTYPE> {
private:
public:
    LayerNormalizeLayer(Operator<DTYPE> *pInput, int unbiased=TRUE, float epsilon = 0.00001, int batchIndex=1, std::string pName = "NO NAME") : Module<DTYPE>(pName) {
#ifdef __DEBUG__
        std::cout << "LayerNormalizeLayer<DTYPE>::LayerNormalizeLayer(Operator<DTYPE> *pInput, int, float, int, string)" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, unbiased, epsilon, batchIndex, pName);    
    }

    int Alloc(Operator<DTYPE> *pInput, int unbiased, float epsilon, int batchIndex, std::string pName) {
#ifdef __DEBUG__
        std::cout << "LayerNormalizeLayer<DTYPE>::Alloc(Operator<DTYPE> *pInput, int, float, int, std::string pName)" << '\n';
#endif  // __DEBUG__

        this->SetInput(1, pInput);

        Operator<DTYPE> *out;

        Tensorholder<DTYPE> *pWeight = NULL;
        Tensorholder<DTYPE> *pBias = NULL;

        Shape* pShape   = pInput->GetResult()->GetShape();
        int timesize    = pShape->GetDim(0);
        int batchsize   = pShape->GetDim(1);
        int channelsize = pShape->GetDim(2);
        int rowsize     = pShape->GetDim(3);
        int colsize     = pShape->GetDim(4);

        int dimsize[5]  = {timesize, batchsize, channelsize, rowsize, colsize};
        for(int i = 0; i <= batchIndex; i++){
            dimsize[i] = 1;
        }

        pWeight = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(dimsize[0], dimsize[1], dimsize[2], dimsize[3], dimsize[4], 1), pName + "_Weight");
        pBias   = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(dimsize[0], dimsize[1], dimsize[2], dimsize[3], dimsize[4]), pName + "_Bias");

        out     = new LayerNormalize<DTYPE>(pInput, pWeight, pBias, batchIndex, unbiased, epsilon, pName);


        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif
