#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"


class my_SeqToSeq : public NeuralNetwork<float>{
private:
public:
    my_SeqToSeq(Tensorholder<float> *input1, Tensorholder<float> *input2, Tensorholder<float> *label, int vocabLength, int embeddingDim, int hiddenDim, Tensorholder<float> *EncoderLengths = NULL, Tensorholder<float> *DecoderLengths = NULL) {
        SetInput(5, input1, input2, label, EncoderLengths, DecoderLengths);

        Operator<float> *out = NULL;


        // ======================= layer 1=======================
        out = new Encoder<float>(input1, vocabLength, embeddingDim, hiddenDim, TRUE, "Encoder");

        //              Alloc(pInput, pEncoder, vocabLength, embeddingDim, hiddensize, outputsize,  teacheringforcing, pEncoderLengths, use_bias, pName);
        //out = new Decoder<float>(input2, out, vocabLength, embeddingDim, 64, vocabLength, TRUE, EncoderLengths, TRUE, "Decoder");       //teacher forcing
        //out = new Decoder<float>(input2, out, vocabLength, embeddingDim, 64, vocabLength, FALSE, EncoderLengths, TRUE, "Decoder");       //teacher forcing X

        //새로운 train함수를 위한 decoder
        out = new Decoder<float>(input2, out, vocabLength, embeddingDim, hiddenDim, vocabLength, FALSE, EncoderLengths, TRUE, "Decoder");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        //SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE", DecoderLengths));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

        // ======================= Select Optimizer ===================
        //SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.0001, 0.9, 1.0, MINIMIZE));
        SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.0009, 0.9, 1e-08, FALSE, MINIMIZE));
        //SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
        // SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        //SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));      //MAXIMIZE

    }

    virtual ~my_SeqToSeq() {}
};
