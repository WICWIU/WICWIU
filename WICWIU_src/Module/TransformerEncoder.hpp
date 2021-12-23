#ifndef __TRANSFORMER_ENCODER__
#define __TRANSFORMER_ENCODER__

#include "../Module.hpp"

template<typename DTYPE>
class TransformerEncoder : public Encoder<DTYPE> {
private:
    Operator<DTYPE> **m_pEncoderLayers;
public:
    TransformerEncoder(Operator<DTYPE> *pInput, Operator<DTYPE> *srcMask, int nLayer, int embeddingDim, int nHead, std::string Name) : Encoder<DTYPE>(Name) {
#ifdef __DEBUG__
        std::cout << "TransformerEncoder<DTYPE>::TransformerEncoder(Operator<DTYPE> *, Operator<DTYPE> *, int , int , int , std::string )" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, srcMask, nLayer, embeddingDim, nHead, Name);
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *srcMask, int nLayer, int embeddingDim, int nHead, std::string Name) {
        #ifdef __DEBUG__
        std::cout << "TransformerEncoder<DTYPE>::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int , int , int , std::string )" << '\n';
        #endif  // __DEBUG__

        this->SetInput(2, pInput, srcMask);

        Operator<DTYPE> *out = pInput;
        m_pEncoderLayers = new Operator<DTYPE>*[nLayer];
        for (int i = 0; i < nLayer; i++) {
            out = new TransformerEncoderLayer<DTYPE>(out, srcMask, embeddingDim, nHead, Name+"_EncoderLayer"+std::to_string(i));
            m_pEncoderLayers[i] = out;
        }

        out = new LayerNormalizeLayer<DTYPE>(out, TRUE, 1e-9, 3, Name+"_LayerNormalizeLayer");

        this->AnalyzeGraph(out);

        return TRUE;
    }

    Operator<DTYPE> *GetEncoderLayer(int idx) {
        return m_pEncoderLayers[idx];
    }

    ~TransformerEncoder(){
        delete[] m_pEncoderLayers;
    }
};

#endif //__TRANSFORMER_ENCODER__
