#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

class my_FaceNetNN : public NeuralNetwork<float>{
private:
public:
    my_FaceNetNN(Tensorholder<float> *x, Tensorholder<float> *label) {
        SetInput(2, x, label);

        Operator<float> *out = NULL;

        out = new ReShape<float>(x, 28, 28, "Flat2Image");

        // ======================= layer 1=======================
        out = new ConvolutionLayer2D<float>(out, 1, 32, 3, 3, 1, 1, 1, FALSE, "Conv_1");
        out = new BatchNormalizeLayer<float>(out, TRUE, "BasicBlock_BN_1");
        out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_1");
        out = new Relu<float>(out, "Relu_1");

        // ======================= layer 2=======================
        out = new ConvolutionLayer2D<float>(out, 32, 64, 3, 3, 1, 1, 1, FALSE, "Conv_2");
        out = new ConvolutionLayer2D<float>(out, 64, 64, 3, 3, 1, 1, 1, FALSE, "Conv_2_1");
        out = new BatchNormalizeLayer<float>(out, TRUE, "BasicBlock_BN_1");
        out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_2");
        out = new Relu<float>(out, "Relu_2");

        // ======================= layer 2=======================

        out = new ConvolutionLayer2D<float>(out, 64, 128, 3, 3, 1, 1, 1, FALSE, "Conv_3");
        out = new ConvolutionLayer2D<float>(out, 128, 128, 3, 3, 1, 1, 1, FALSE, "Conv_3_a");
        out = new BatchNormalizeLayer<float>(out, TRUE, "BasicBlock_BN_1");
        out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_3");
        out = new Relu<float>(out, "Relu_3");

        out = new ConvolutionLayer2D<float>(out, 128, 256, 3, 3, 1, 1, 1, FALSE, "Conv_4");
        out = new BatchNormalizeLayer<float>(out, TRUE, "BasicBlock_BN_1");
        // out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_4");
        out = new Relu<float>(out, "Relu_5");

        // ======================= layer 3=======================
        out = new ReShape<float>(out, 1, 1, 3 * 3 * 256, "Image2Flat");

        // ======================= layer 3=======================
        out = new Linear<float>(out, 3 * 3 * 256, 256, TRUE, "Fully-Connected_1");
        out = new Relu<float>(out, "Relu_4");
        //
        //// ======================= layer 4=======================
        // out = new Linear<float>(out, 256, 128, TRUE, "Fully-connected_2");
        // out = new Relu<float>(out, "Relu_5");
        out = new Linear<float>(out, 256, 64, TRUE, "Fully-connected_3");
        // out = new L2_normalize<float>(out, "L2_normalize");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        // SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));
        SetLossFunction(new TripletLoss<float>(out, label, 1.0, "TPL"));
        // ======================= Select Optimizer ===================
        // SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        // SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.001, 0.9, 1e-08, FALSE, MINIMIZE));
        SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.0001, 0.9, 0.999, 1e-08, MINIMIZE));
        // SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        // SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
    }

    virtual ~my_FaceNetNN() {}
};
