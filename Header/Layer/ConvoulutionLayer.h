#ifndef __CONVOLUTION_LAYER__
#define __CONVOLUTION_LAYER__    value

#include "../Layer.h"

template<typename DTYPE> class ConvolutionLayer2D : public Layer<DTYPE>{
private:
public:
    ConvolutionLayer2D(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, int pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int pPadding, int use_bias = FALSE, std::string pName = "NO NAME") : Layer<DTYPE>(pName){
        Alloc(pInput, pNumInputChannel, pNumOutputChannel, pNumKernelRow, pNumKernelCol, pStrideRow, pStrideCol, pPadding, pPadding, use_bias, pName);
    }

    virtual ~ConvolutionLayer2D() {}

    int Alloc(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, int pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int pPaddingRow, int pPaddingCol, int use_bias, std::string pName) {
        Operator<DTYPE> *out = pInput;

        Tensorholder<DTYPE> *pWeight = (Tensorholder<DTYPE> *)this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Truncated_normal(1, pNumOutputChannel, pNumInputChannel, pNumKernelRow, pNumKernelCol, 0.0, 0.1), "Convolution2D_Weight_" + pName));
        out = this->AddOperator(new Convolution2D<DTYPE>(out, pWeight, pStrideRow, pStrideCol, pPaddingRow, pPaddingCol, "Convolution2D_Convolution2D_" + pName));

        if(use_bias){
            Tensorholder<DTYPE> *pBias = (Tensorholder<DTYPE> *)this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumOutputChannel, 1, 1, 0), "Convolution2D_Bias_" + pName));
            out = this->AddOperator(new Add<DTYPE>(out, pBias, "Convolution2D_Add_" + pName));
        }

        return TRUE;
    }
};


#endif  // __CONVOLUTION_LAYER__
