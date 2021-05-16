#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"


class my_GRU : public NeuralNetwork<float>{
private:
public:
    my_GRU(Tensorholder<float> *x, Tensorholder<float> *label, int vocabLength, int embeddingDim, int hiddenSize) {
        SetInput(2, x, label);

        Operator<float> *out = NULL;

        //Embedding Layer
        out = new EmbeddingLayer<float>(x, vocabLength, embeddingDim, "Embedding");

        // ======================= layer 1=======================
        out = new GRULayer<float>(out, embeddingDim, hiddenSize, NULL, TRUE, "Recur_1");

        // // ======================= layer 2=======================
        out = new Linear<float>(out, hiddenSize, vocabLength, TRUE, "Fully-Connected_2");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

        // ======================= Select Optimizer ===================
        //SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.009, 0.9, 1e-08, FALSE, MINIMIZE));
        //SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
        // SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        //SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));


    }

    virtual ~my_GRU() {}
};
