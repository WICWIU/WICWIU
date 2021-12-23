#ifndef __TRANSFORMER_PE_LAYER__
#define __TRANSFORMER_PE_LAYER__    value

#include "../Module.hpp"
#include "../../WICWIU_src/PositionalEncoding.hpp"


template<typename DTYPE> class TransformerPELayer : public Module<DTYPE>{
private:
public:

    TransformerPELayer(Operator<DTYPE> *pInput, int batchSize, int vocabSize, int embeddingDim, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, batchSize, vocabSize, embeddingDim, pName);
    }

    virtual ~TransformerPELayer() {}


    int Alloc(Operator<DTYPE> *pInput, int batchSize, int vocabSize, int embeddingDim, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        PositionalEncoding<float>* pe = new TransformerPositionalEncoding<float>(batchSize, vocabSize, embeddingDim);
        Tensorholder<float>* peHolder = new Tensorholder<float>(1, batchSize, vocabSize, 1, embeddingDim, pName + "_TransformerPositionalEncoding", FALSE);
        peHolder->SetIsTensorholder(FALSE);
        peHolder->SetTensor(pe->GetPositionalEncoding());
        out = new Addall<float>(out, peHolder, "Add PostionalEncoding Value");

        this->AnalyzeGraph(out);

        return TRUE;
    }

};


#endif  // __TRANSFORMER_PE_LAYER__
