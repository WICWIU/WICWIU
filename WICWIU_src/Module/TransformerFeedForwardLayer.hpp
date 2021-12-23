#ifndef __TRANSFORMER_FEEDFORWARD_LAYER_HPP__
#define __TRANSFORMER_FEEDFORWARD_LAYER_HPP__

#include "../Module.hpp"

#define D_FF 2048

template<typename DTYPE> class TransformerFeedForwardLayer : public Module<DTYPE> {
private:
public:
    TransformerFeedForwardLayer(Operator<DTYPE> *pInput, int d_model, float droprate = 0.1, std::string pName = "NO NAME") : Module<DTYPE>(pName) {
#ifdef __DEBUG__
        std::cout << "TransformerFeedForwardLayer<DTPYE>::TransformerFeedForwardLayer(Operator<DTYPE> *pInput, int d_model, float droprate, std::string pName)" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, d_model, droprate, pName);
    }

    int Alloc(Operator<DTYPE> *pInput, int d_model, float droprate, std::string pName) {
#ifdef __DEBUG__
        std::cout << "TransformerFeedForwardLayer<DTPYE>::Alloc(Operator<DTYPE> *pInput, int d_model, float droprate, std::string pName)" << '\n';
#endif  // __DEBUG__

        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        out = new Linear<DTYPE>(out, d_model, D_FF, FALSE, pName+"_Linear1");
        out = new Relu<DTYPE>(out, pName+"_Relu");
        out = new Dropout<DTYPE>(out, droprate, pName+"_Dropout");
        out = new Linear<DTYPE>(out, D_FF, d_model, FALSE, pName+"_Linear2");

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif
