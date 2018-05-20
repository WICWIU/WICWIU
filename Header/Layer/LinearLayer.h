#ifndef __LINEAR_LAYER__
#define __LINEAR_LAYER__    value

#include "../Layer.h"

template<typename DTYPE> class Linear : public Layer<DTYPE>{
private:
public:
    Linear(Operator<float> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias = FALSE, std::string pName = NULL) : Layer<DTYPE>(pName){
        Alloc(pInput, pNumInputCol, pNumOutputCol, use_bias, pName);
    }

    virtual ~Linear() {}

    int Alloc(Operator<float> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias, std::string pName) {
        Operator<float> *out = pInput;

        Tensorholder<DTYPE> *pWeight = (Tensorholder<DTYPE> *)this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Truncated_normal(1, 1, 1, pNumOutputCol, pNumInputCol, 0.0, 0.1), "Layer_Weight_" + pName));
        out = this->AddOperator(new MatMul<DTYPE>(pWeight, out, "Layer_MatMul_" + pName));

        if (use_bias) {
            Tensorholder<DTYPE> *pBias = (Tensorholder<DTYPE> *)this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumOutputCol, 0.f), "Add_Bias_" + pName));
            out = this->AddOperator(new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName));
        }

        return TRUE;
    }
};

#endif  // __LINEAR_LAYER__
