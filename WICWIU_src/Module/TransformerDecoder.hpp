#ifndef __TRANSFORMER_DECODER__
#define __TRANSFORMER_DECODER__

#include "../Module.hpp"

template<typename DTYPE>
class TransformerDecoder : public Decoder<DTYPE> {
private:
    Operator<DTYPE> **m_pDecoderLayers;
public:
    TransformerDecoder(Operator<DTYPE> *pInput, Operator<DTYPE> *pContext, Operator<DTYPE> *srcMask, Operator<DTYPE> *tgtMask, int nLayer, int embeddingDim, int nHead, std::string pName) : Decoder<DTYPE>(pName) {
        #ifdef __DEBUG__
        std::cout << "TransformerDecoder<DTYPE>::TransformerDecoder(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, int , int , int , std::string )" << '\n';
        #endif  // __DEBUG__

        Alloc(pInput, pContext, srcMask, tgtMask, nLayer, embeddingDim, nHead, pName);
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pContext, Operator<DTYPE> *srcMask, Operator<DTYPE> *tgtMask, int nLayer, int embeddingDim, int nHead, std::string pName) {
        #ifdef __DEBUG__
        std::cout << "TransformerDecoder<DTYPE>::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, int , int , int , std::string )" << '\n';
        #endif  // __DEBUG__

        this->SetInput(4, pInput, pContext, srcMask, tgtMask);

        Operator<DTYPE> *out = pInput;
        m_pDecoderLayers = new Operator<DTYPE>*[nLayer];
        for(int i=0 ; i<nLayer ; i++) {
            out = new TransformerDecoderLayer<DTYPE>(out, pContext, srcMask, tgtMask, embeddingDim, nHead, pName+"_DecoderLayer"+std::to_string(i));
            m_pDecoderLayers[i] = out;
        }

        out = new LayerNormalizeLayer<DTYPE>(out, TRUE, 1e-9, 3, pName+"_LayerNormalizeLayer");

        this->AnalyzeGraph(out);

        return TRUE;
    }

    Operator<DTYPE> *GetDecoderLayer(int idx) {
        return m_pDecoderLayers[idx];
    }

    ~TransformerDecoder(){
        delete[] m_pDecoderLayers;
    }
};

#endif //__TRANSFORMER_DECODER__
