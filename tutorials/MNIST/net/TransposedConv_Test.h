#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.h"

class TransposedConv_Test : public NeuralNetwork<float>{
private:
public:

    TransposedConv_Test(Tensorholder<float> *x, Tensorholder<float> *label) {
        SetInput(2, x, label);
        //std::cout<< label->GetResult() <<std::endl;
        //lable 과 마지막 out 의 result값 확인해본 결과 크기는(shape)는 784개 25batch둘

        Operator<float> *out = NULL;

        out = new ReShape<float>(x, 28, 28, "Flat2Image");

        //Operator<float> *original = new ReShape<float>(x, 28, 28, "Flat2Image");

        // ======================= layer 1=======================
        //out = new ConvolutionLayer2D<float>(out, 1, 32, 2, 2, 2, 2, 0, TRUE, "Conv_1");
        out = new ConvolutionLayer2D<float>(out, 1, 10, 3, 3, 1, 1, 0, TRUE, "Conv_1");

        out = new Relu<float>(out, "Relu_1");

        out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_1");

        out = new ConvolutionLayer2D<float>(out, 10, 20, 7, 7, 1, 1, 0, TRUE, "Conv_2");

        // ======================= layer 2=======================
        //out = new ConvolutionLayer2D<float>(out, 32, 64, 2, 2, 2, 2, 0, TRUE, "Conv_2");

        out = new Relu<float>(out, "Relu_2");

        // ======================= layer 3=======================
        //out = new TransposedConvolutionLayer2D<float>(out, 64, 32, 8, 8, 1, 1, 0, TRUE, "Trans_Conv_1");

        out = new TransposedConvolutionLayer2D<float>(out, 20, 10, 8, 8, 1, 1, 0, TRUE, "Trans_Conv_1");

        out = new Relu<float>(out, "Relu_3");

        //// ======================= layer 4=======================
        out = new TransposedConvolutionLayer2D<float>(out, 10, 1, 15, 15, 1, 1, 0, TRUE, "Trans_Conv_2");

        out = new Relu<float>(out, "Relu_4");

        out = new ReShape<float>(out, 1, 1, 28 * 28 * 1, "Image2Flat");
        //out = new Linear<float>(out, 28 * 28, 1024, TRUE, "Fully-Connected_1");
        //out = new Relu<float>(out, "Relu");
        //out = new Linear<float>(out, 1024, 10, TRUE, "Fully-Connected_2");


        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        SetLossFunction(new MSE_backup<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.00005, MINIMIZE));

    }

    virtual ~TransposedConv_Test() {}

};
