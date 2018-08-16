#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.h"

class my_CNN : public NeuralNetwork<float>{
private:
public:
    my_CNN(Tensorholder<float> *x, Tensorholder<float> *label) {
        SetInput(2, x, label);

        Operator<float> *out = NULL;

        out = new ReShape<float>(x, 3, 32, 32, "Flat2Image");

        // ======================= layer 1=======================
        out = new ConvolutionLayer2D<float>(out, 3, 32, 3, 3, 1, 1, 0, TRUE, "Conv_1");
        out = new BatchNormalizeLayer<float>(out, TRUE, "BN_1");
        out = new Relu<float>(out, "Relu_1");
        out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_1");

        // ======================= layer 2=======================
        out = new ConvolutionLayer2D<float>(out, 32, 64, 3, 3, 1, 1, 0, TRUE, "Conv_2");
        out = new BatchNormalizeLayer<float>(out, TRUE, "BN_2");
        out = new Relu<float>(out, "Relu_2");
        out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_2");

        // ======================= layer 3=======================
        out = new ReShape<float>(out, 1, 1, 6 * 6 * 64, "Image2Flat");

        // ======================= layer 3=======================
        out = new Linear<float>(out, 6 * 6 * 64, 1024, TRUE, "Fully-Connected_1");

        out = new Relu<float>(out, "Relu_3");
        //
        //// ======================= layer 4=======================
        out = new Linear<float>(out, 1024, 10, TRUE, "Fully-connected_2");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
    }

    virtual ~my_CNN() {}

};
