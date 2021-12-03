#ifndef __TRANSFORMER_DECODER_LAYER_HPP__
#define __TRANSFORMER_DECODER_LAYER_HPP__

#include "../Module.hpp"

#define EPSILON 1e-6
#define DROPRATE 0.1

template<typename DTYPE> class TransformerDecoderLayer : public Module<DTYPE> {
private:
    Operator<DTYPE> *m_pSelfMultiHeadAttention;
    Operator<DTYPE> *m_pSrcMultiHeadAttention;
public:
    TransformerDecoderLayer(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, Operator<DTYPE> *psrcMask, Operator<DTYPE> *ptgtMask, int  embedding_dim, int n_head, std::string pName = "No Name") : Module<DTYPE>(pName) {
#ifdef __DEBUG__
        std::cout << "TransformerDecoderLayer<DTYPE>::TransformerDecoderLayer(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, int, int, std::string)" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, pEncoder, psrcMask, ptgtMask, embedding_dim, n_head, pName);
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, Operator<DTYPE> *psrcMask, Operator<DTYPE> *ptgtMask, int  embedding_dim, int n_head, std::string pName) {
#ifdef __DEBUG__
        std::cout << "TransformerDecoderLayer<DTYPE>::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, int , int , std::string)" << '\n';
#endif  // __DEBUG__

        this->SetInput(4, pInput, pEncoder, psrcMask, ptgtMask);

        Operator<DTYPE> *out = NULL;
        Operator<DTYPE> *remember = pInput;
        Operator<DTYPE> *outQuery = pInput;
        Operator<DTYPE> *outKey   = pInput;
        Operator<DTYPE> *outValue = pInput;

        out = new MultiHeadAttention<DTYPE>(outKey, outQuery, outValue, ptgtMask, DROPRATE, embedding_dim, n_head, pName + "_DecoderSelfMultiHeadAttention");
        m_pSelfMultiHeadAttention = out;
        out = new TransformerSubLayerConnection<DTYPE>(out, remember, EPSILON, DROPRATE, pName+"_SubLayerConnection1");

        remember = out;
        outQuery = out;
        outKey   = pEncoder;
        outValue = pEncoder;

        out = new MultiHeadAttention<DTYPE>(outKey, outQuery, outValue, psrcMask, DROPRATE, embedding_dim, n_head, pName + "_DecoderSrcMultiheadAttention");
        m_pSrcMultiHeadAttention = out;
        out = new TransformerSubLayerConnection<DTYPE>(out, remember, EPSILON, DROPRATE, pName+"_SubLayerConnection2");

        remember = out;

        out = new TransformerFeedForwardLayer<DTYPE>(out, embedding_dim, DROPRATE, pName+"_FeedForward");
        out = new TransformerSubLayerConnection<DTYPE>(out, remember, EPSILON, DROPRATE, pName+"_SubLayerConnection3");

        this->AnalyzeGraph(out);

        return TRUE;
    }

    Operator<DTYPE> *GetSelfMultiHeadAttention() {
        return m_pSelfMultiHeadAttention;
    }

    Operator<DTYPE> *GetSrcMultiHeadAttention() {
        return m_pSrcMultiHeadAttention;
    }
};
#endif
