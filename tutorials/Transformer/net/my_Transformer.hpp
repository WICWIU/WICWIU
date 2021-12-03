#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

enum class DecoderMode : int {
    PARALLEL,
    GREEDY,
    BEAMSEARCH
};

template<typename DTYPE> class my_Transformer : public NeuralNetwork<float>{
private:
    Operator<DTYPE> *m_pEncoder;
    Operator<DTYPE> *m_pGenerator;
    Operator<DTYPE> *m_pSrcMask;
    Operator<DTYPE> *m_pTgtMask;

    int m_sequenceLength;
    int m_embeddingDim;
    int m_srcVocabSize;
    int m_tgtVocabSize;
    int m_nHead;
    int m_nLayer;

    int m_eosTok;

    DecoderMode m_decoderMode;

public:
    my_Transformer(Tensorholder<float> *indexedFeature, Tensorholder<float>* rightShiftedFeature, Tensorholder<float> *label, int sequenceLength, int embeddingDim, int srcVocabSize, int tgtVocabSize, int nHead, int nLayer, int eosTok = 2, DecoderMode mode = DecoderMode::PARALLEL) {
        SetInput(3, indexedFeature, rightShiftedFeature, label);

        m_sequenceLength = sequenceLength;
        m_embeddingDim   = embeddingDim;
        m_srcVocabSize   = srcVocabSize;
        m_tgtVocabSize   = tgtVocabSize;
        m_nHead          = nHead;
        m_nLayer         = nLayer;
        m_eosTok         = eosTok;
        m_decoderMode    = mode;

        Operator<float> *out      = NULL;
        Operator<float> *srcMask  = NULL;
        Operator<float> *tgtMask  = NULL;
        Operator<float> *context  = NULL;
        Operator<float> *eInput   = NULL;
        Operator<float> *dInput   = NULL;

        eInput = indexedFeature;
        int batchSize = indexedFeature->GetResult()->GetBatchSize();

        srcMask = new AttentionPaddingMask<float>(indexedFeature, sequenceLength, 0, FALSE, "srcMasking");
        tgtMask = new AttentionPaddingMask<float>(rightShiftedFeature, sequenceLength, 0, TRUE, "tgtMasking");
        m_pSrcMask = srcMask;
        m_pTgtMask = tgtMask;

        out = new TransformerEncoderModule<float>(eInput, srcMask, m_nLayer, m_srcVocabSize, m_sequenceLength, m_embeddingDim, m_nHead, "TransformerEncoderModule");
        m_pEncoder = out;
        // ======================== DECODER =================================

        context = out;
        dInput  = rightShiftedFeature;

        out = new TransformerDecoderModule<float>(dInput, context, srcMask, tgtMask, m_nLayer, m_tgtVocabSize, m_sequenceLength, m_embeddingDim, m_nHead, "TransformerDecoderModule");


        m_pGenerator = out;

        this->AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        SetLossFunction(new SmoothedKLDivLoss<float>(out, label, 0.1, m_tgtVocabSize, 2, "KLDiv"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new AdamOptimizer<float>(this->GetParameter(), 0.001, 0.9, 0.98, 1e-9f, MINIMIZE));

    }

    virtual ~my_Transformer() {

    }

    Operator<DTYPE> *GetEncoderWrapper(){
        return m_pEncoder;
    }

    Operator<DTYPE> *GetDecoderWrapper(){
        return m_pGenerator;
    }

    Tensor<DTYPE> *GetEncoderSelfAttention() {
        TransformerEncoderModule<float> *encoderWrapper         = (TransformerEncoderModule<float> *)this->GetEncoderWrapper();
        TransformerEncoder<float>        *encoder                = (TransformerEncoder<float> *)encoderWrapper->GetEncoder();
        TransformerEncoderLayer<float>   *encoderlayer1          = (TransformerEncoderLayer<float> *)encoder->GetEncoderLayer(0);
        MultiHeadAttention<float>        *encoderMultiheadAttn   = (MultiHeadAttention<float> *)encoderlayer1->GetMultiHeadAttention();
        DotProductAttentionWeight<float> *encoderAttentionWeight = (DotProductAttentionWeight<float> *)encoderMultiheadAttn->GetAttentionWeight();
        Operator<float>                  *encoderSrcAttn         = encoderAttentionWeight->GetAttention();
        return encoderSrcAttn->GetResult();
    }
    
    Tensor<DTYPE> *GetDecoderSelfAttention() {
        TransformerDecoderModule<float> *decoderWrapper         = (TransformerDecoderModule<float> *)this->GetDecoderWrapper();
        TransformerDecoder<float>        *decoder                = (TransformerDecoder<float> *)decoderWrapper->GetDecoder();
        TransformerDecoderLayer<float>   *decoderLayer1          = (TransformerDecoderLayer<float> *)decoder->GetDecoderLayer(0);
        MultiHeadAttention<float>        *decoderMultiheadAttn   = (MultiHeadAttention<float> *)decoderLayer1->GetSelfMultiHeadAttention();
        DotProductAttentionWeight<float> *decoderAttentionWeight = (DotProductAttentionWeight<float> *)decoderMultiheadAttn->GetAttentionWeight();
        Operator<float>                  *decoderSelfAttn        = decoderAttentionWeight->GetAttention();
        return decoderSelfAttn->GetResult();
    }

    Tensor<DTYPE> *GetDecoderSrcAttention() {
        TransformerDecoderModule<float> *decoderWrapper         = (TransformerDecoderModule<float> *)this->GetDecoderWrapper();
        TransformerDecoder<float>        *decoder                = (TransformerDecoder<float> *)decoderWrapper->GetDecoder();
        TransformerDecoderLayer<float>   *decoderLayer1          = (TransformerDecoderLayer<float> *)decoder->GetDecoderLayer(0);
        MultiHeadAttention<float>        *decoderMultiheadAttn   = (MultiHeadAttention<float> *)decoderLayer1->GetSrcMultiHeadAttention();
        DotProductAttentionWeight<float> *decoderAttentionWeight = (DotProductAttentionWeight<float> *)decoderMultiheadAttn->GetAttentionWeight();
        Operator<float>                  *decoderSrcAttn         = decoderAttentionWeight->GetAttention();
        return decoderSrcAttn->GetResult();
    }

    
    
    Tensor<DTYPE> *MakeInference(Tensor<DTYPE> *pInput, Tensor<DTYPE> *pLabel) {
        int timesize    = pInput->GetTimeSize();
        int batchsize   = pInput->GetBatchSize();
        int channelsize = pInput->GetChannelSize();
        int rowsize     = pInput->GetRowSize();
        int colsize     = pInput->GetColSize();

        Tensor<DTYPE> *output = Tensor<DTYPE>::Constants(timesize, batchsize, channelsize, rowsize, colsize, 1);
        Shape *shape = pInput->GetShape();

        int tensorsize = timesize * batchsize * channelsize * rowsize * colsize;
        Tensor<DTYPE> *originInput = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *originDInput = this->GetInput()[1]->GetResult();

        for (int i = 0; i < tensorsize; i++) {
            (*originInput)[i]  = (*pInput)[i];
            (*originDInput)[i] = (*output)[i];
        }
        
        m_pSrcMask->ResetResult();
        m_pEncoder->ResetResult();

#ifdef __CUDNN__
        m_pSrcMask->ForwardPropagateOnGPU(0);
        m_pEncoder->ForwardPropagateOnGPU(0);
#else
        m_pSrcMask->ForwardPropagate();
        m_pEncoder->ForwardPropagate();
#endif

        for (int co = 0; co < colsize-1; co++) {
            m_pTgtMask->ResetResult();
            m_pGenerator->ResetResult();
#ifdef __CUDNN__
            m_pTgtMask->ForwardPropagateOnGPU(0);
            m_pGenerator->ForwardPropagateOnGPU(0);
#else
            m_pTgtMask->ForwardPropagate();
            m_pGenerator->ForwardPropagate();
#endif
            Tensor<DTYPE> *generated = m_pGenerator->GetResult()->Argmax(4);
            for (int ba = 0; ba < batchsize; ba++) {
                int inference = (*generated)[Index5D(generated->GetShape(), 0, ba, 0, 0, co)];
                (*originDInput)[Index5D(shape, 0, ba, 0, 0, co+1)] = (DTYPE)inference;
                (*output)[Index5D(shape, 0, ba, 0, 0, co+1)] = (DTYPE)inference;
            }
            delete generated;
        }

        return output;
    }


};
