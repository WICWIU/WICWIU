#ifndef __GRU_LAYER__
#define __GRU_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class GRULayer : public Module<DTYPE>{
private:
public:

    GRULayer(Operator<DTYPE> *pInput, int inputSize, int hiddenSize, Operator<DTYPE> *initHidden, int useBias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputSize, hiddenSize, initHidden, useBias, pName);
    }
    virtual ~GRULayer() {}

    int Alloc(Operator<DTYPE> *pInput, int inputSize, int hiddenSize, Operator<DTYPE> *initHidden, int useBias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        //weight
        Tensorholder<DTYPE> *pWeightIG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddenSize, inputSize, 0.0, 0.01), "GRULayer_pWeight_IG_" + pName);
        Tensorholder<DTYPE> *pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddenSize, hiddenSize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);
        Tensorholder<DTYPE> *pWeightICH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, inputSize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);
        Tensorholder<DTYPE> *pWeightHCH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, hiddenSize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);

        //bias
        Tensorholder<DTYPE> *gBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 2*hiddenSize, 0.f), "RNN_Bias_f" + pName);
        Tensorholder<DTYPE> *chBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddenSize, 0.f), "RNN_Bias_f" + pName);

#ifdef __CUDNN__
        pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, 3*hiddenSize+3*inputSize+3, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);    //For 1 bias option
#endif  // __CUDNN__

        out = new GRU<DTYPE>(out, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias, initHidden);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif
