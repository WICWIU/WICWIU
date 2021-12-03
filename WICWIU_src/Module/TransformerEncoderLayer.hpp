#ifndef __TRANSFORMER_ENCODER_LAYER_HPP__
#define __TRANSFORMER_ENCODER_LAYER_HPP__

#include "../Module.hpp"

#define EPSILON 1e-6
#define DROPRATE 0.1

template<typename DTYPE> class TransformerEncoderLayer : public Module<DTYPE> {
private:
    Operator<DTYPE> *m_pMultiHeadAttention;
public:
    TransformerEncoderLayer(Operator<DTYPE> *pInput, Operator<DTYPE> *psrcMask, int embedding_dim, int n_head, std::string pName = "No Name") : Module<DTYPE>(pName) {
#ifdef __DEBUG__
        std::cout << "TransformerEncoderLayer<DTYPE>::TransformerEncoderLayer(Operator<DTYPE> *, Operator<DTYPE> *,int, int, std::string)" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, psrcMask, embedding_dim, n_head, pName);
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *psrcMask, int embedding_dim, int n_head, std::string pName) {
#ifdef __DEBUG__
        std::cout << "TransformerEncoderLayer<DTYPE>::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int embedding_dim, int n_head, std::string Name)" << '\n';
#endif  // __DEBUG__

        this->SetInput(2, pInput, psrcMask);
        Operator<DTYPE> *out = pInput;
        Operator<DTYPE> *remember = pInput;

        Operator<DTYPE> *outQuery = out;
        Operator<DTYPE> *outKey   = out;
        Operator<DTYPE> *outValue = out;


        out = new MultiHeadAttention<DTYPE>(outKey, outQuery, outValue, psrcMask, DROPRATE, embedding_dim, n_head, pName + "_EncoderSelfMultiHeadAttention");
        m_pMultiHeadAttention = out;
        out = new TransformerSubLayerConnection<DTYPE>(out, remember, EPSILON, DROPRATE, pName + "_SubLayerConnection1");

        remember = out;

        out = new TransformerFeedForwardLayer<DTYPE>(out, embedding_dim, DROPRATE, pName + "_FeedForward");
        out = new TransformerSubLayerConnection<DTYPE>(out, remember, EPSILON, DROPRATE, pName + "_SubLayerConnection2");

        this->AnalyzeGraph(out);

        return TRUE;
    }

    Operator<DTYPE> *GetMultiHeadAttention() {
        return m_pMultiHeadAttention;
    }
};

#endif