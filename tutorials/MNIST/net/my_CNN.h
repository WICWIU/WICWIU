#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.h"

class my_CNN : public NeuralNetwork<float>{
private:
public:
    my_CNN(Tensorholder<float> *x, Tensorholder<float> *label) {
        SetInput(2, x, label);

        Operator<float> *out = NULL;

        out = new ReShape<float>(x, 28, 28, "ReShape");

        // ======================= layer 1=======================
        // out = ConvolutionLayer2D(out, 1, 10, 3, 3, 1, 1, 0, TRUE, "1");
        out = new ConvolutionLayer2D<float>(out, 1, 10, 3, 3, 1, 1, 0, TRUE, "1");

        out = new Relu<float>(out, "Relu_1");
        out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_1");

        // ======================= layer 2=======================
        // out = ConvolutionLayer2D(out, 10, 20, 3, 3, 1, 1, 0, TRUE, "2");
        out = new ConvolutionLayer2D<float>(out, 10, 20, 3, 3, 1, 1, 0, TRUE, "2");

        out = new Relu<float>(out, "Relu_2");
        out = new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_2");

        out = new ReShape<float>(out, 1, 1, 5 * 5 * 20, "Flat");

        // ======================= layer 3=======================
        // out = Linear(out, 5 * 5 * 20, 1024, TRUE, "3");
        out = new Linear<float>(out, 5 * 5 * 20, 1024, TRUE, "3");

        out = new Relu<float>(out, "Relu_3");
        //
        //// ======================= layer 4=======================
        // out = Linear(out, 1024, 10, TRUE, "4");
        out = new Linear<float>(out, 1024, 10, TRUE, "4");

        AnalyseGraph(out);

        // ======================= Select LossFunction Function ===================
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.04, MINIMIZE));
        // SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, MINIMIZE));
    }

    virtual ~my_CNN() {}

    // Operator<float> * ConvolutionLayer2D(Operator<float> *pInput, int pNumInputChannel, int pNumOutputChannel, int pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int pPadding, int use_bias = FALSE, std::string pName = "NO NAME"){
    // Operator<float> *out = pInput;
    //
    // Tensorholder<float> *pWeight = new Tensorholder<float>(Tensor<float>::Random_normal(1, pNumOutputChannel, pNumInputChannel, pNumKernelRow, pNumKernelCol, 0.0, 0.1), "Convolution2D_Weight_" + pName);
    // out = new Convolution2D<float>(out, pWeight, pStrideRow, pStrideCol, pPadding, pPadding, "Convolution2D_Convolution2D_" + pName);
    //
    // if(use_bias){
    // Tensorholder<float> *pBias = new Tensorholder<float>(Tensor<float>::Constants(1, 1, pNumOutputChannel, 1, 1, 0), "Convolution2D_Bias_" + pName);
    // out = new AddChannelWise<float>(out, pBias, "Convolution2D_Add_" + pName);
    // }
    //
    // return out;
    // }
    //
    // Operator<float> * Linear(Operator<float> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias = FALSE, std::string pName = NULL){
    // Operator<float> *out = pInput;
    //
    // Tensorholder<float> *pWeight = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, pNumOutputCol, pNumInputCol, 0.0, 0.1), "Layer_Weight_" + pName);
    // out = new MatMul<float>(pWeight, out, "Layer_MatMul_" + pName);
    //
    // if (use_bias) {
    // Tensorholder<float> *pBias = new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, pNumOutputCol, 0.f), "Add_Bias_" + pName);
    // out = new AddColWise<float>(out, pBias, "Layer_Add_" + pName);
    // }
    //
    // return out;
    // }
};
