#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"


class my_AttentionSeqToSeq : public NeuralNetwork<float>{
private:
public:
    my_AttentionSeqToSeq(Tensorholder<float> *EncoderInput, Tensorholder<float> *DecoderInput, Tensorholder<float> *label, Tensorholder<float> *EncoderLengths, Tensorholder<float> *DecoderLengths, int vocabLength, int embeddingDim) {
        SetInput(5, EncoderInput, DecoderInput, label, EncoderLengths, DecoderLengths);

        Operator<float> *out = NULL;

        //out = new CBOW<float>(x(입력 배열), 아웃풋크기, "CBOW");
        //out = new OnehotVector<float>(x(입력 배열), 아웃풋크기, "OnehotVector");

        //중요!!! 여기에 operator를 추가하면 time이 안돌아감!....
        //그리고 embedding할때도.... EncoderInput하고 DecoderInput하고 dim이 다름....
        //embedding - 그래서 embedding은 encoder하고 decoder 내부로 들어가야 됨!!!
        //  -
        //  - pytorch에서도 그렇게 길이 다르게 해서 for문 돌림!

        Operator<float> *mask = new PaddingAttentionMaskRNN<float>(EncoderInput, 0, "srcMasking");

        // ======================= layer 1=======================
        out = new Encoder<float>(EncoderInput, vocabLength, embeddingDim, 64, TRUE, "Encoder");

        out = new AttentionDecoder_Module<float>(DecoderInput, out, mask, vocabLength, embeddingDim, 64, vocabLength, EncoderLengths, TRUE, "Decoder");

        // ContextVector = new AttentionModule<float>(enc, dec, mask, "attention");
        //
        // concate = new ConcatenateColumnWise<float>(dec, ContextVector);
        //
        // out = new Linear<float>(out, 1024, 10, TRUE, "Fully-connected_2");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        //SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        SetLossFunction(new SoftmaxCrossEntropy_padding<float>(out, label, DecoderLengths, "SCE"));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

        // ======================= Select Optimizer ===================
        // 1.0이 clipValue 값! 인자 하나가 더 생김!
        //현재 RMSprop clip값 = 0.5로 되어있음
        //SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, 0.9, 1.0, MINIMIZE));                      // Optimizer의 첫번째 인자로 parameter목록을 전달해주는거고!!!   즉 updateparameter를 할 때 넘겨주는 parameter에 대해서만 함!!!!!
        SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.01, 0.9, 1e-08, FALSE, MINIMIZE));
        //SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
        // SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        //SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));      //MAXIMIZE

    }

    virtual ~my_AttentionSeqToSeq() {}
};
