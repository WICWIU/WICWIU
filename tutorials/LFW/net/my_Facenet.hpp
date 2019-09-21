#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

template<typename DTYPE> class my_FaceNet : public NeuralNetwork<float>{
private:
public:
    my_FaceNet(Tensorholder<float> *x, Tensorholder<float> *label, int pNumOfClass) {
        SetInput(2, x, label);

        Operator<float> *out = NULL;

        out = new ReShape<float>(x, 3, 224, 224, "Flat2Image");

        // ======================= layer 1=======================
        out = new ConvolutionLayer2D<float>(out, 3, 64, 3, 3, 2, 2, 1, TRUE, "Conv_1");
        out = new Relu<float>(out, "Relu_1");
        out = new Maxpooling2D<float>(out, 3, 3, 2, 2, 1, "MaxPool_1");

        // ======================= layer 2=======================
        out = new ConvolutionLayer2D<float>(out, 64, 64, 1, 1, 1, 1, 0, TRUE, "Conv_2");
        out = new ConvolutionLayer2D<float>(out, 64, 192, 3, 3, 1, 1, 1, TRUE, "Conv_2-a");
        out = new BatchNormalizeLayer<float>(out, TRUE, "BN_1");
        out = new Relu<float>(out, "Relu_2");
        out = new Maxpooling2D<float>(out, 3, 3, 2, 2, 1, "MaxPool_2");

        // ======================= Layer 3========================
        out = new ConvolutionLayer2D<float>(out, 192, 192, 1, 1, 1, 1, 0, TRUE, "Conv_3");
        out = new ConvolutionLayer2D<float>(out, 192, 384, 3, 3, 1, 1, 1, TRUE, "Conv_3-a");
        out = new BatchNormalizeLayer<float>(out, TRUE, "BN_2");
        out = new Relu<float>(out, "Relu_3");
        out = new Maxpooling2D<float>(out, 3, 3, 2, 2, 1, "MaxPool_3");

        // ======================= Layer 4========================
        out = new ConvolutionLayer2D<float>(out, 384, 384, 1, 1, 1, 1, 0, TRUE, "Conv_4");
        out = new ConvolutionLayer2D<float>(out, 384, 256, 3, 3, 1, 1, 1, TRUE, "Conv_4-a");
        out = new Relu<float>(out, "Relu_4");

        // ======================= Layer 5========================
        out = new ConvolutionLayer2D<float>(out, 256, 256, 1, 1, 1, 1, 0, TRUE, "Conv_5");
        out = new ConvolutionLayer2D<float>(out, 256, 256, 3, 3, 1, 1, 1, TRUE, "Conv_5-a");
        out = new Relu<float>(out, "Relu_5");

        // ======================= Layer 6========================
        out = new ConvolutionLayer2D<float>(out, 256, 256, 1, 1, 1, 1, 0, TRUE, "Conv_4");
        out = new ConvolutionLayer2D<float>(out, 256, 256, 3, 3, 1, 1, 1, TRUE, "Conv_4-a");
        out = new Relu<float>(out, "Relu_6");
        out = new Maxpooling2D<float>(out, 3, 3, 2, 2, 1, "MaxPool_4");

        // ======================= Layer 7========================
        out = new ReShape<float>(out, 1, 1, 7 * 7 * 256, "Image2Flat");

        out = new Linear<float>(out, 7 * 7 * 256, 1 * 32 * 128, TRUE, "Fully-Connected_1");

        out = new Linear<float>(out, 1 * 32 * 128, 1 * 1 * 128, TRUE, "Fully-Connected_2");

        // out = new L2_norm<float>(out, "L2_norm");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        SetLossFunction(new TripletLoss<float>(out, label, 1.0, "TripletLoss"));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
    }

    virtual ~my_FaceNet() {}
};
