#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.h"

class my_CNN : public NeuralNetwork<float>{
private:
public:

    my_CNN(Tensorholder<float> *x, Tensorholder<float> *label) {
        SetInput(2, x, label);

        Operator<float> *out = NULL;

        out = new ReShape<float>(x, 28, 28, "Flat2Image");

        // ======================= layer 1=======================
        out = new ConvolutionLayer2D<float>(out, 1, 10, 3, 3, 1, 1, 0, TRUE, "Conv_1");

        //std::cout<< "slope1: "<< slope1 <<std::endl;
        out = new Relu<float>(out, "Relu_1");
        //out = new LRelu<float>(out, 0.01, "LRelu_1");
        //out = new PReluLayer<float>(out, "PRelu_1");

        out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_1");

        // ======================= layer 2=======================
        out = new ConvolutionLayer2D<float>(out, 10, 20, 3, 3, 1, 1, 0, TRUE, "Conv_2");

        out = new Relu<float>(out, "Relu_2");
        //out = new LRelu<float>(out, 0.01, "LRelu_2");
        //out = new PReluLayer<float>(out, "PRelu_2");

        out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_2");


        //out = new TransposedConvolutionLayer2D<float>(out, 20, 10, 6, 6, 1, 1, 0, TRUE, "Trans_Conv_3");
        //out = new ConvolutionLayer2D<float>(out, 10, 20, 6, 6, 1, 1, 0, TRUE, "Conv_3");



        // ======================= layer 3=======================
        out = new ReShape<float>(out, 1, 1, 5 * 5 * 20, "Image2Flat");

        // ======================= layer 3=======================
        out = new Linear<float>(out, 5 * 5 * 20, 1024, TRUE, "Fully-Connected_1");

        out = new Relu<float>(out, "Relu_3");
        //out = new LRelu<float>(out, 0.01, "LRelu_3");
        //out = new PReluLayer<float>(out, "PRelu_3");

        // ======================= layer 4=======================
        out = new Linear<float>(out, 1024, 10, TRUE, "Fully-connected_2");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.04, MINIMIZE));
        //SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.04, 0.5, MINIMIZE));

    }

    virtual ~my_CNN() {}

};
