#include <iostream>
#include <string>

#include "../../../../WICWIU_src/NeuralNetwork.hpp"

using namespace std;

template<typename DTYPE> class my_Generator : public NeuralNetwork<DTYPE> {
private:
public:
    my_Generator(Operator<float> *z){
        Alloc(z);
    }

    virtual ~my_Generator() {
    }

    int Alloc(Operator<float> *z){
        this->SetInput(z);

        Operator<float> *out = z;
        const int D = 128;

        // MNIST
        out = new Linear<float>(out, 100, 4 * 4 * 512);
        out = new Relu<float>(out, "G_Relu0");
        out = new ReShape<float>(out, 512, 4, 4, "G_Reshape1");

        // input 4x4 output 7x7
        out = new TransposedConvolutionLayer2D<float>(out, 512, 256, 3, 3, 2, 2, 1, FALSE, "G_TranspoedConv1");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN1");
        out = new Relu<float>(out, "G_Relu1");

        // input 7x7 output 14x14
        out = new TransposedConvolutionLayer2D<float>(out, 256, 128, 4, 4, 2, 2, 1, FALSE, "G_TranspoedConv2");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN2");
        out = new Relu<float>(out, "G_Relu2");

        // input 14x14 output 28x28
        out = new TransposedConvolutionLayer2D<float>(out, 128, 1, 4, 4, 2, 2, 1, FALSE, "G_TransposedConv3");

        out = new Tanh<float>(out, "G_Tanh");
        out = new ReShape<float>(out, 1, 1, 28*28, "Img2Flat");

        this->AnalyzeGraph(out);
    }
};
