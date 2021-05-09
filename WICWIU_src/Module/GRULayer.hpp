#ifndef __GRU_LAYER__
#define __GRU_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class GRULayer : public Module<DTYPE>{
private:
public:

    GRULayer(Operator<DTYPE> *pInput, int inputsize, int hiddensize, Operator<DTYPE> *initHidden, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputsize, hiddensize, initHidden, use_bias, pName);
    }
    virtual ~GRULayer() {}

    int Alloc(Operator<DTYPE> *pInput, int inputsize, int hiddensize, Operator<DTYPE> *initHidden, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        //weight
        Tensorholder<DTYPE> *pWeightIG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, inputsize, 0.0, 0.01), "GRULayer_pWeight_IG_" + pName);
        Tensorholder<DTYPE> *pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, hiddensize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);
        Tensorholder<DTYPE> *pWeightICH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);
        Tensorholder<DTYPE> *pWeightHCH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);

        //bias
        Tensorholder<DTYPE> *gBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 2*hiddensize, 0.f), "RNN_Bias_f" + pName);
        Tensorholder<DTYPE> *chBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_f" + pName);

#ifdef __CUDNN__
        pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, 3*hiddensize+3*inputsize+3, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);    //bias 1개 일때!!!        //이거 output으로 바꿔서 해보기!!!
#endif  // __CUDNN__


        //out = new GRU<DTYPE>(out, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias);
        out = new SeqGRU<DTYPE>(out, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias, initHidden);

        //이제 h2o 부분 밖으로 뺴기!!!
        // out = new MatMul<DTYPE>(pWeight_h2o, out, "rnn_matmul_ho");
        //
        // if (use_bias) {
        //     Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, outputsize, 0.f), "Add_Bias_" + pName);
        //     out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        // }


        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif
