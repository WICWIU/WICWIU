#ifndef TFENCODERWRAPPER_HPP_
#define TFENCODERWRAPPER_HPP_

#include "../Module.hpp"

template<typename DTYPE>
class TransformerEncoderModule : public Module<DTYPE> {
private:
    Operator<DTYPE> *m_pEncoder;
public:
    TransformerEncoderModule(Operator<DTYPE> *pInput, Operator<DTYPE> *pSrcMask, int nLayer, int vocabSize, int vocabLength, int embeddingDim, int nHead, std::string pName) : Module<DTYPE>(pName) {
#ifdef __DEBUG__
        std::cout << "TransformerEncoderModule<DTYPE>::TransformerEncoderModule(Operator<DTYPE> *, Operator<DTYPE> *, int , int , int , int , int , std::string )" << '\n';
#endif  // __DEBUG__

        Alloc(pInput, pSrcMask, nLayer, vocabSize, vocabLength, embeddingDim, nHead, pName);
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pSrcMask, int nLayer, int vocabSize, int vocabLength, int embeddingDim, int nHead, std::string pName) {
#ifdef __DEBUG__
        std::cout << "TransformerEncoderModule<DTYPE>::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int , int , int , std::string )" << '\n';
#endif  // __DEBUG__

        this->SetInput(2, pInput, pSrcMask);

        Operator<DTYPE> *out = pInput;
        out = new EmbeddingLayer<float>(out, vocabSize, embeddingDim, pName + "_SrcEmbeddingLayer");

        out = new Transpose<float>(out, 2, 3, pName + "_SrcEmbeddingTranspose");
        out = new TransformerPELayer<float>(out, pInput->GetResult()->GetBatchSize(), vocabLength, embeddingDim, pName + "_SrcPositionalEncoding");

        out = new TransformerEncoder<float>(out, pSrcMask, nLayer, embeddingDim, nHead, pName + "_Encoder");
        m_pEncoder = out;
        this->AnalyzeGraph(out);

        return TRUE;
    }
    
    Operator<DTYPE> *GetEncoder(){
        return m_pEncoder;
    }
};

#endif
