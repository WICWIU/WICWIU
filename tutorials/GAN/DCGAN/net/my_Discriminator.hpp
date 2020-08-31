#include <iostream>
#include <string>

#include "../../../../WICWIU_src/NeuralNetwork.hpp"
using namespace std;

template <typename DTYPE>
class my_Discriminator : public NeuralNetwork<DTYPE>
{
public:
    my_Discriminator(Operator<float>* x) { Alloc(x); }

    virtual ~my_Discriminator() {}

    int Alloc(Operator<float>* x)
    {
        this->SetInput(x);

        Operator<float>* out = x;

        // MNIST
        out = new ReShape<float>(out, 1, 28, 28, "Flat2Img");

        // input 28x28 output 14x14
        out = new ConvolutionLayer2D<float>(out, 1, 128, 3, 3, 2, 2, 1, 1, "D_Conv1");
        out = new BatchNormalizeLayer<float>(out, TRUE, "D_BN1");
        out = new LRelu<float>(out, 0.2, "D_Relu1");

        // input 14x14 output 7x7
        out = new ConvolutionLayer2D<float>(out, 128, 64, 3, 3, 2, 2, 1, 1, "D_Conv2");
        out = new BatchNormalizeLayer<float>(out, TRUE, "D_BN2");
        out = new LRelu<float>(out, 0.2, "D_Relu2");

        out = new ReShape<float>(out, 1, 1, 64 * 7 * 7, "D_ReShape3");
        out = new Linear<float>(out, 64 * 7 * 7, 1);
        out = new Sigmoid<float>(out, "D_Sigmoid3");

        this->AnalyzeGraph(out);
    }
};
