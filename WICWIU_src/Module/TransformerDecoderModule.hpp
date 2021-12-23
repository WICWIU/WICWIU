#ifndef TFDECODERWRAPPER_HPP_
#define TFDECODERWRAPPER_HPP_

#include "../Module.hpp"

template<typename DTYPE>
class TransformerDecoderModule : public Module<DTYPE> {
private:
    int m_nLayer;
    int m_vocabSize;
    int m_vocabLength;
    int m_embeddingDim;
    int m_nHead;
    Operator<DTYPE> *m_pDecoder;
public:
    TransformerDecoderModule(Operator<DTYPE> *pInput, Operator<DTYPE> *pContext, Operator<DTYPE> *pSrcMask, Operator<DTYPE> *pTgtMask, int nLayer, int vocabSize, int vocabLength, int embeddingDim, int nHead, std::string Name) : Module<DTYPE>(Name) {
        #ifdef __DEBUG__
        std::cout << "TransformerDecoderModule<DTYPE>::TransformerDecoderModule(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, int , int , int , std::string )" << '\n';
        #endif  // __DEBUG__

        m_nLayer = 0;
        m_vocabSize = 0;
        m_vocabLength = 0;
        m_embeddingDim = 0;
        m_nHead = 0;

        Alloc(pInput, pContext, pSrcMask, pTgtMask, nLayer, vocabSize, vocabLength, embeddingDim, nHead, Name);
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pContext, Operator<DTYPE> *pSrcMask, Operator<DTYPE> *pTgtMask, int nLayer, int vocabSize, int vocabLength, int embeddingDim, int nHead, std::string Name) {
        #ifdef __DEBUG__
        std::cout << "TransformerDecoderModule<DTYPE>::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, int , int , int , std::string )" << '\n';
        #endif  // __DEBUG__

        this->SetInput(4, pInput, pContext, pSrcMask, pTgtMask);

        m_nLayer = nLayer;
        m_vocabSize = vocabSize;
        m_vocabLength = vocabLength;
        m_embeddingDim = embeddingDim;
        m_nHead = nHead;


        Operator<DTYPE> *out = pInput;

        out = new EmbeddingLayer<float>(out, vocabSize, embeddingDim, "tgtEmbedding_Layer");
        out = new Transpose<float>(out, 2, 3, "tgtEmbeddingTranspose");
        out = new TransformerPELayer<float>(out, pInput->GetResult()->GetBatchSize(), vocabLength, embeddingDim, "tgtPositionalEncoding");

        out = new TransformerDecoder<float>(out, pContext, pSrcMask, pTgtMask, nLayer, embeddingDim, nHead, "Decoder");
        m_pDecoder = out;

        out = new TransformerGenerator<float>(out, embeddingDim, vocabSize, "Generator");
        this->AnalyzeGraph(out);

        return TRUE;
    }

    Operator<DTYPE> *MakeDeepCopy() {
        Container<Operator<DTYPE> *> *inputContainer = this->GetInputContainer();

        Module<DTYPE> *pCopy = new TransformerDecoderModule<DTYPE>((*inputContainer)[0], (*inputContainer)[1], (*inputContainer)[2], (*inputContainer)[3], m_nLayer, m_vocabSize, m_vocabLength, m_embeddingDim, m_nHead, "Copyed");

        // Container<Operator<DTYPE> *> *excutableOpContaine  r = this->GetExcutableOperatorContainer();
        // int numExcutableOpContainer = this-> GetNumOfExcutableOperator();

        Container<Operator<DTYPE> *> *paramContainer = this->GetParameterContainer();
        int numParamContainer = paramContainer->GetSize();

        // Container<Operator<DTYPE> *> *cpyExcutableOpContainer = this->GetExcutableOperatorContainer();

        Container<Operator<DTYPE> *> *thisParamContainer = this->GetParameterContainer();
        Container<Operator<DTYPE> *> *cpyParamContainer = pCopy->GetParameterContainer();

        for(int i=0; i<thisParamContainer->GetSize(); i++) {
            Tensor<DTYPE> *cpyTensor((*thisParamContainer)[i]->GetResult());
            (*cpyParamContainer)[i]->SetResult(cpyTensor);
        }

        return pCopy;
    }

    Operator<DTYPE> *MakeShallowCopy() {
        Container<Operator<DTYPE> *> *inputContainer = this->GetInputContainer();
        Module<DTYPE> *pCopy = new TransformerDecoderModule<DTYPE>((*inputContainer)[0], (*inputContainer)[1], (*inputContainer)[2], (*inputContainer)[3], m_nLayer, m_vocabSize, m_vocabLength, m_embeddingDim, m_nHead, this->GetName());

        pCopy->SetParamCopied();

        Container<Operator<DTYPE> *> *thisParamContainer = this->GetParameterContainer();
        Container<Operator<DTYPE> *> *cpyParamContainer = pCopy->GetParameterContainer();

        for(int i=0; i<thisParamContainer->GetSize(); i++)
            (*cpyParamContainer)[i]->SetResult((*thisParamContainer)[i]->GetResult());

        return pCopy;
    }

    Module<DTYPE> *GetThisPointer() {
        return this;
    }

    Operator<DTYPE> *GetDecoder(){
        return m_pDecoder;
    }
};

#endif
