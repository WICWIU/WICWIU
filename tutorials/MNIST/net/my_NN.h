#include <iostream>

#include "../../../WICWIU_src/NeuralNetwork.h"

enum MODEL_OPTION {
    isSLP,
    isMLP
};

class my_NN : public NeuralNetwork<float>{
private:
public:
    my_NN(Tensorholder<float> *x, Tensorholder<float> *label, MODEL_OPTION pOption) {
        SetInput(2, x, label);

        if (pOption == isSLP) SLP(x, label);
        else if (pOption == isMLP) MLP(x, label);
    }

    void SLP(Tensorholder<float> *x, Tensorholder<float> *label) {
        Operator<float> *out = x;

        // ======================= layer 1======================
        out = new Linear<float>(out, 784, 10, TRUE, "1");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, 0.000001, "SCE"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.01, 0.9, MINIMIZE));
        // SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, MINIMIZE));
    }

    void MLP(Tensorholder<float> *x, Tensorholder<float> *label) {
        Operator<float> *out = x;

        // ======================= layer 1======================
        out = new Linear<float>(out, 784, 15, TRUE, "1");

        out = new Tanh<float>(out, "Tanh");

        // ======================= layer 2=======================
        out = new Linear<float>(out, 15, 10, TRUE, "2");

        AnalyzeGraph(out);


        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.0001, 0.9, MINIMIZE));
    }

    virtual ~my_NN() {}

};
