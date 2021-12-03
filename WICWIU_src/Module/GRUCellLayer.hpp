#ifndef __GRUCELL_LAYER__
#define __GRUCELL_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class GRUCellLayer : public Module<DTYPE>{
private:
public:

    GRUCellLayer(Operator<DTYPE> *pInput, int inputsize, int hiddensize, Operator<DTYPE> *initHidden, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputsize, hiddensize, initHidden, use_bias, pName);
    }
    virtual ~GRUCellLayer() {}

    int Alloc(Operator<DTYPE> *pInput, int inputsize, int hiddensize, Operator<DTYPE> *initHidden, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        //weight
        Tensorholder<DTYPE> *pWeightIG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, inputsize, 0.0, 0.01), "GRUCellLayer_pWeight_IG_" + pName);
        Tensorholder<DTYPE> *pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, hiddensize, 0.0, 0.01), "GRUCellLayer_pWeight_HG_" + pName);
        Tensorholder<DTYPE> *pWeightICH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "GRUCellLayer_pWeight_ICH_" + pName);
        Tensorholder<DTYPE> *pWeightHCH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "GRUCellLayer_pWeight_HCH_" + pName);

        //bias
        Tensorholder<DTYPE> *gBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 2*hiddensize, 0.f), "GRUCellLayer_Bias_g" + pName);
        Tensorholder<DTYPE> *chBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "GRUCellLayer_Bias_ch" + pName);

#ifdef __CUDNN__
        pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, 3*hiddensize+3*inputsize+3, 0.0, 0.01), "__GRUCellLayer_pWeight_h2h_" + pName);  
#endif  // __CUDNN__

        out = new GRUCell<DTYPE>(out, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias, initHidden);


        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif
