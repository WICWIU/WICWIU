#ifndef __TANFORMER_SUB_LAYER_CONNECTION_HPP__
#define __TANFORMER_SUB_LAYER_CONNECTION_HPP__

#include "../Module.hpp"

template<typename DTYPE> class TransformerSubLayerConnection : public Module<DTYPE> {
private:
public:
    TransformerSubLayerConnection(Operator<DTYPE>* pInput, Operator<DTYPE>* pRemember, float epsilon, float Droprate, std::string pName = "NO NAME") : Module<DTYPE>(pName) {
#ifdef __DEBUG__
        std::cout << "TransformerSubLayerConnection::TransformerSubLayerConnection(Operator<DTYPE>* pInput, float epsilon, std::string pName)" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, pRemember, epsilon, Droprate, pName);
    }

    int Alloc(Operator<DTYPE>* pInput, Operator<DTYPE>* pRemember, float epsilon, float Droprate, std::string pName) {
#ifdef __DEBUG__
        std::cout << "TransformerSubLayerConnection<DTYPE>::Alloc(Operator<DTYPE>* pInput, float epsilon, float Droprate, std::string pName)" << '\n';
#endif  // __DEBUG__

        this->SetInput(2, pInput, pRemember);

        Operator<DTYPE>* out = pInput;

        out = new Dropout<DTYPE>(out, Droprate, pName + "_Dropout");
        out = new Addall<DTYPE>(out, pRemember, pName + "_Add");
        out = new LayerNormalizeLayer<DTYPE>(out, TRUE, epsilon, 3, pName + "_LayerNormalizeLayer");

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif
