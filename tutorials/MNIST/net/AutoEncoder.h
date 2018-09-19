#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.h"

class AutoEncoder : public NeuralNetwork<float>{
private:
public:

    AutoEncoder(Tensorholder<float> *x, Tensorholder<float> *label) {
        SetInput(2, x, label);
        Operator<float> *out = NULL;

        out = new ReShape<float>(x, 28, 28, "Flat2Image");

        // ======================= layer 1=======================
        out = new ConvolutionLayer2D<float>(out, 1, 10, 3, 3, 1, 1, 0, TRUE, "Conv_1");

        out = new Relu<float>(out, "Relu_1");

        out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_1");

        // ======================= layer 2=======================
        out = new ConvolutionLayer2D<float>(out, 10, 20, 7, 7, 1, 1, 0, TRUE, "Conv_2");

        out = new Relu<float>(out, "Relu_2");

        // ======================= layer 3=======================
        out = new TransposedConvolutionLayer2D<float>(out, 20, 10, 8, 8, 1, 1, 0, TRUE, "Trans_Conv_1");
        //out = new TransposedConvolutionLayer2D<float>(out, 20, 10, 2, 2, 2, 2, 0, TRUE, "Trans_Conv_1"); //stride test -> ok

        out = new Relu<float>(out, "Relu_3");

        // ======================= layer 4=======================
        out = new TransposedConvolutionLayer2D<float>(out, 10, 1, 15, 15, 1, 1, 0, TRUE, "Trans_Conv_2");

        out = new Relu<float>(out, "Relu_4");

        out = new ReShape<float>(out, 1, 1, 28 * 28 * 1, "Image2Flat");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        SetLossFunction(new MSE_backup<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.00005, MINIMIZE));

    }

    virtual ~AutoEncoder() {}

};
