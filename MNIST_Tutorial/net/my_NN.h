#include <iostream>

#include "../../Header/NeuralNetwork.h"

enum MODEL_OPTION {
    isSLP,
    isMLP
};

class my_NN : public NeuralNetwork<float>{
private:
public:
    my_NN(Tensorholder<float> *x, Tensorholder<float> *label, MODEL_OPTION pOption) {
        if (pOption == isSLP) SLP(x, label);
        else if (pOption == isMLP) MLP(x, label);
    }

    void SLP(Tensorholder<float> *x, Tensorholder<float> *label) {
        Operator<float> *out = x;

        // ======================= layer 1======================
        out = AddOperator(new Linear<float>(out, 784, 10, TRUE, "1"));

        // ======================= Select LossFunction Function ===================
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, 0.000001, "SCE"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetTensorholder(), 0.01, 0.9, MINIMIZE));
        // SetOptimizer(new GradientDescentOptimizer<float>(GetTensorholder(), 0.001, MINIMIZE));

    }

    void MLP(Tensorholder<float> *x, Tensorholder<float> *label) {
        Operator<float> *out = x;

        // ======================= layer 1======================
        out = AddOperator(new Linear<float>(out, 784, 15, TRUE, "1"));

        out = AddOperator(new Sigmoid<float>(out, "Sigmoid"));

        // ======================= layer 2=======================
        out = AddOperator(new Linear<float>(out, 15, 10, TRUE, "2"));

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        SetLossFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetTensorholder(), 0.01, 0.9, MINIMIZE));

    }

    virtual ~my_NN() {}
};
